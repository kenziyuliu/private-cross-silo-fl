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
from utils import gen_batch, client_selection
import utils

from trainers.base import BaseTrainerLocal


class Ditto(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Ditto')

    # Since Ditto take multiple inner iters, we overwrite noise & data generator
    self.global_iters = 1
    self.local_iters = 1
    step_factor = self.global_iters + self.local_iters
    self.noise_mults = dp_accounting.compute_silo_noise_mults(self.num_clients,
                                                              self.train_samples,
                                                              args,
                                                              steps_factor=step_factor)
    for t in range(self.num_clients):
      self.batch_gen[t] = gen_batch(self.x_train[t],
                                    self.y_train[t],
                                    self.batch_sizes[t],
                                    num_iter=step_factor * (self.num_rounds + 1) * self.inner_iters)

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_global_updates = [0] * self.num_clients

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
      key = random.fold_in(key, batch_idx)
      params = opt.params_fn(opt_state)
      mean_grad = grad(loss_fn)(params, batch, global_params)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    def private_batch_update(key, batch_idx, opt_state, global_params, batch, noise_mult):
      key = random.fold_in(key, batch_idx)
      # Add fake batch dimension to data for vmapping
      batch = (batch[0][:, None], batch[1][:, None])
      params = opt.params_fn(opt_state)
      grad_fn = vmap(grad(loss_fn), in_axes=(None, 0, None), out_axes=0)
      example_grads = grad_fn(params, batch, global_params)
      mean_grad = jax_utils.privatize_grad_hk(example_grads, key, self.l2_clip, noise_mult)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    batch_update = private_batch_update if self.use_dp else batch_update
    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Ditto] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))  # NOTE: use all clients for cross-silo.
      for t in selected_clients:
        key = random.fold_in(key, t)
        # Batch generator
        if self.inner_mode == 'iter':
          global_batches = (next(self.batch_gen[t])
                            for _ in range(self.inner_iters * self.global_iters))
          local_batches = (next(self.batch_gen[t])
                           for _ in range(self.inner_iters * self.local_iters))
        else:
          epoch_gen_fn = functools.partial(utils.epochs_generator, self.x_train[t], self.y_train[t],
                                           self.batch_sizes[t])
          global_batches = epoch_gen_fn(epochs=self.inner_epochs * self.global_iters,
                                        seed=int(key[0]))
          local_batches = epoch_gen_fn(epochs=self.inner_epochs * self.local_iters,
                                       seed=int(key[1]))

        # Global updates
        opt_state = opt.init_fn(global_params)
        for batch_idx, batch in enumerate(global_batches):
          prox_params = opt.params_fn(opt_state)  # We want prox term to be 0 for global update
          key, opt_state = batch_update(key, batch_idx, opt_state, prox_params, batch,
                                        self.noise_mults[t])
        new_global_params = opt.params_fn(opt_state)

        # Local updates
        opt_state = opt.init_fn(local_params[t])
        for batch_idx, batch in enumerate(local_batches):
          prox_params = global_params
          key, opt_state = batch_update(key, batch_idx, opt_state, prox_params, batch,
                                        self.noise_mults[t])
        new_local_params = opt.params_fn(opt_state)

        # Record new *local* model and *global* model diff
        local_global_updates[t] = jax_utils.model_subtract(new_global_params, global_params)
        local_params[t] = new_local_params

      # Update global model
      average_update = jax_utils.model_average(local_global_updates, weights=self.update_weights)
      global_params = jax_utils.model_add(global_params, average_update)
      local_global_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        train_accu, test_accu = self.eval(local_params, i)

    return train_accu, test_accu
