import os
import functools

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import dp_accounting
import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class MRMTL(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Mean-regularized MTL')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optimizers.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    def loss_fn(params, batch, global_params):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      model_diff = flat_params - jax_utils.model_flatten(global_params)
      prox_term = 0.5 * self.lam * (model_diff @ model_diff)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + prox_term + l2_term

    def batch_update(key, batch_idx, opt_state, global_params, batch, __):
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, batch, global_params)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    def private_batch_update(key, batch_idx, opt_state, global_params, batch, noise_mult):
      # Add fake batch dimension to data for vmapping
      batch = (batch[0][:, None], batch[1][:, None])
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      grad_fn = vmap(grad(loss_fn), in_axes=(None, 0, None), out_axes=0)
      example_grads = grad_fn(params, batch, global_params)
      mean_grad = jax_utils.privatize_grad_hk(example_grads, key, self.l2_clip, noise_mult)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    batch_update = private_batch_update if self.use_dp else batch_update
    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[MRMTL] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))  # NOTE: use all clients for cross-silo.
      for t in selected_clients:
        key = random.fold_in(key, t)
        # Batch generator
        if self.inner_mode == 'iter':
          batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
        else:
          batches = utils.epochs_generator(self.x_train[t],
                                           self.y_train[t],
                                           self.batch_sizes[t],
                                           epochs=self.inner_epochs,
                                           seed=int(key[0]))
        # Local batches
        opt_state = opt.init_fn(local_params[t])
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, batch_idx, opt_state, global_params, batch,
                                        self.noise_mults[t])

        # Record new model and model diff
        new_local_params = opt.params_fn(opt_state)
        local_updates[t] = jax_utils.model_subtract(new_local_params, local_params[t])
        local_params[t] = new_local_params

      # Update global model
      average_update = jax_utils.model_average(local_updates, weights=self.update_weights)
      global_params = jax_utils.model_add(global_params, average_update)
      local_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval(local_params, i)

    return train_accu, test_accu