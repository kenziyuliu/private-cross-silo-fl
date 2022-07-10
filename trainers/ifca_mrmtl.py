import collections
import os
import functools
import pprint

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import dp_accounting
import jax_utils
import model_utils
import utils

from trainers.base import BaseTrainerLocal


class IFCA_MRMTL(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)

    if args['ifca_fedavg_frac'] is None:
      args['ifca_fedavg_frac'] = args['ifca_select_frac']

    select_frac, fedavg_frac = args['ifca_select_frac'], args['ifca_fedavg_frac']
    print(f'[IFCA Base]: select_frac={select_frac}, fedavg_frac={fedavg_frac}')
    self.num_select_rounds = int((self.num_rounds + 1) * select_frac)
    self.num_fedavg_rounds = int((self.num_rounds + 1) * fedavg_frac)
    assert 1 <= self.num_select_rounds <= self.num_fedavg_rounds

    self.iter_desc = '[IFCA Base]'
    self.noise_mults = dp_accounting.compute_silo_noise_mults(self.num_clients,
                                                              self.train_samples,
                                                              args,
                                                              num_selections=self.num_select_rounds,
                                                              selection_eps=args['selection_eps'])

    self.loss_sens_bound = args['ifca_loss_sens']

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    # Ensure random cluster inits for linear problems (by default they use 0 init).
    if self.args['is_linear']:
      cluster_model_fn = lambda inputs: model_utils.linear_model_fn(inputs, zero_init=False)
    else:
      cluster_model_fn = self.model_fn
    cluster_model = hk.without_apply_rng(hk.transform(cluster_model_fn))
    cluster_params = []
    for k in range(self.num_clusters):
      cluster_params.append(cluster_model.init(key, data_batch))
      key = random.fold_in(key, k)

    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optimizers.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, batch, centroid_params):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      model_diff = flat_params - jax_utils.model_flatten(centroid_params)
      prox_term = 0.5 * self.lam * (model_diff @ model_diff)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + prox_term + l2_term

    def batch_update(key, batch_idx, opt_state, centroid_params, batch, __):
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, batch, centroid_params)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    def private_batch_update(key, batch_idx, opt_state, centroid_params, batch, noise_mult):
      # Add fake batch dimension to data for vmapping
      batch = (batch[0][:, None], batch[1][:, None])
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      grad_fn = vmap(grad(loss_fn), in_axes=(None, 0, None), out_axes=0)
      example_grads = grad_fn(params, batch, centroid_params)
      mean_grad = jax_utils.privatize_grad_hk(example_grads, key, self.l2_clip, noise_mult)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    @functools.partial(jit, static_argnums=(1, 2))
    def _client_eval_model(params, client_idx, private=True):
      # Regression (use loss as metric) or Clasification with loss
      if self.args['is_regression'] or self.args['ifca_metric'] == 'loss':
        client_dataset = (self.x_train[client_idx], self.y_train[client_idx])
        train_metric = self.data_loss_fn(params, client_dataset)
        if private:
          # Bound sensitivity of training loss
          train_metric = jnp.minimum(train_metric, self.loss_sens_bound)
      # Classification
      else:
        # Use error_rate = 1 - accuracy to take min metric
        train_preds = self.pred_fn(params, self.x_train[client_idx])
        train_metric = 1.0 - jnp.mean(train_preds == self.y_train[client_idx])

      return train_metric

    def select_cluster(cluster_metrics, eps, sens, key):
      # Metrics are the lower the better.
      return jnp.argmin(cluster_metrics)

    def private_select_cluster(cluster_metrics, eps, sens, key):
      """Exponential mechanism implemented with `Report Gumbel-noisy Min`.
      With noise scale sens * 2 / eps, the mechanism is (eps^2/8)-zCDP.
      https://differentialprivacy.org/exponential-mechanism-bounded-range/
      Metrics are the lower the better.
      """
      scale = sens * 2 / eps
      gumbel_noise = random.gumbel(key, shape=cluster_metrics.shape) * scale
      noisy_metrics = cluster_metrics - gumbel_noise
      return jnp.argmin(noisy_metrics)

    select_cluster = private_select_cluster if self.use_dp else select_cluster
    select_cluster = jit(select_cluster)
    batch_update = private_batch_update if self.use_dp else batch_update
    batch_update = jit(batch_update)

    ############################################################################

    # NOTE: Maintain cluster assignments for stage 1, freeze for stage 2.
    client_2_cluster = None
    cluster_2_client = None
    cluster_sizes = []

    print(f'[IFCA stages] Select / Fedavg / Tot: '
          f'{self.num_select_rounds} / {self.num_fedavg_rounds} / {self.num_rounds + 1}')

    # Training loop
    progress_bar = tqdm(range(self.num_rounds + 1),
                        desc=f'{self.iter_desc} {cluster_sizes} Round',
                        disable=(self.args['repeat'] != 1))
    for i in progress_bar:
      select_stage = (i < self.num_select_rounds)
      fedavg_stage = (i < self.num_fedavg_rounds)

      # NOTE: clear cluster assignments during stage 1.
      if select_stage:
        client_2_cluster = [None] * self.num_clients
        cluster_2_client = collections.defaultdict(list)

      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))  # NOTE: use all clients for cross-silo.
      for t in selected_clients:
        key = random.fold_in(key, t)
        if self.inner_mode == 'iter':
          batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
        else:
          batches = utils.epochs_generator(self.x_train[t],
                                           self.y_train[t],
                                           self.batch_sizes[t],
                                           epochs=self.inner_epochs,
                                           seed=int(key[0]))

        #### NOTE: Stage 1/2: IFCA with/without selection
        if fedavg_stage:
          if select_stage:
            cluster_metrics = np.array(
                [_client_eval_model(params, t, self.use_dp) for params in cluster_params])
            # Clients take centriod with lowest metric
            if self.args['is_regression']:
              selection_sens = self.loss_sens_bound
            else:
              # Sensitivity of accuracy is 1/(N-1) across add/remove/replace
              selection_sens = 1 / (self.train_samples[t] - 1)
            cluster_idx = select_cluster(cluster_metrics,
                                         eps=self.args['selection_eps'],
                                         sens=selection_sens,
                                         key=key)
            cluster_idx = cluster_idx.tolist()

            client_2_cluster[t] = cluster_idx
            cluster_2_client[cluster_idx].append(t)

          # While in FedAvg stage, client's models are the cluster-fedavg models
          local_params[t] = cluster_params[client_2_cluster[t]]

        ##### NOTE: Stage 3, MR-MTL training (for stage 1/2, local model is centroid)
        centroid_params = cluster_params[client_2_cluster[t]]

        # Local batches
        opt_state = opt.init_fn(local_params[t])
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, batch_idx, opt_state, centroid_params, batch,
                                        self.noise_mults[t])

        # Record new model and model diff
        # NOTE: in either stage, you send back the model diff; you just keep different models.
        new_local_params = opt.params_fn(opt_state)
        local_updates[t] = jax_utils.model_subtract(new_local_params, local_params[t])
        local_params[t] = new_local_params

      #### NOTE: All stages: update cluster model, FedAvg style
      # Difference: cluster is frozen after stage 1; stage 3 keeps local model.
      for k in range(self.num_clusters):
        if len(cluster_2_client[k]) > 0:
          model_updates = [local_updates[t] for t in cluster_2_client[k]]
          update_weights = [self.update_weights[t] for t in cluster_2_client[k]]
          average_update = jax_utils.model_average(model_updates, weights=jnp.asarray(update_weights))
          cluster_params[k] = jax_utils.model_add(cluster_params[k], average_update)

      local_updates = [0] * self.num_clients
      cluster_sizes = [len(cnts) for idx, cnts in cluster_2_client.items()]
      progress_bar.set_description(f'{self.iter_desc} {cluster_sizes} Round')

      if not self.args['quiet']:
        print(f'Cluster sizes = {cluster_sizes}', end=' ')

      if i % self.args['eval_every'] == 0:
        train_metric, test_metric = self.eval(local_params, i)

    return train_metric, test_metric
