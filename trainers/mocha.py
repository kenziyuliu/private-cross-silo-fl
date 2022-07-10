import os
import functools
from typing import List

import numpy as np
from tqdm import tqdm

import haiku as hk
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
from scipy import linalg as LA

import dp_accounting
import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class Mocha(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Mocha')
    # Recompute noise multiplier in case Mocha uses multiple outer iterations
    step_factor = self.args['mocha_outer']
    self.noise_mults = dp_accounting.compute_silo_noise_mults(self.num_clients,
                                                              self.train_samples,
                                                              args,
                                                              steps_factor=step_factor)

  def train(self):
    # Mocha Primal, since we need to add item-level privacy.
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_updates = [0] * self.num_clients
    Sigma = np.eye(self.num_clients) * (1.0 / self.num_clients)

    # Optimizer shared for every client (re-init before client work)
    opt = optimizers.sgd(self.lr)  # provides init_fn, update_fn, params_fn

    @jit
    def _stack_params(local_params: List[hk.Params]):
      return jnp.array([jax_utils.model_flatten(p) for p in local_params]) # (n, d)

    @jit
    def update_sigma(local_params: List[hk.Params]):
      epsil = 1e-8
      params_mat = _stack_params(local_params)  # (n, d)
      A = params_mat @ params_mat.T
      # D, V = LA.eigh(A)
      D, V = jnp.linalg.eigh(A)
      D = (D * (D > epsil)) + epsil * (D <= epsil)
      sqm = jnp.sqrt(D)
      sqm = sqm / jnp.sum(sqm)
      Sigma = V @ jnp.diag(sqm) @ V.T
      return Sigma

    def loss_fn(params, params_mat, client_idx, Sigma, batch):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      params_mat = params_mat.at[client_idx].set(flat_params) # NOTE: ensure current param included for autodiff
      reg_term = 0.5 * self.lam * jnp.trace(params_mat.T @ Sigma @ params_mat)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + reg_term + l2_term

    def batch_update(key, batch_idx, opt_state, params_mat, client_idx, Sigma, batch, __):
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, params_mat, client_idx, Sigma, batch)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    def private_batch_update(key, batch_idx, opt_state, params_mat, client_idx, Sigma, batch, noise_mult):
      # Add fake batch dimension to data for vmapping
      batch = (batch[0][:, None], batch[1][:, None])
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      grad_fn = vmap(grad(loss_fn), in_axes=(None, None, None, None, 0), out_axes=0)
      example_grads = grad_fn(params, params_mat, client_idx, Sigma, batch)
      mean_grad = jax_utils.privatize_grad_hk(example_grads, key, self.l2_clip, noise_mult)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    batch_update = private_batch_update if self.use_dp else batch_update
    batch_update = jit(batch_update)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Mocha] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      # Mocha outer loop
      for j in range(self.args['mocha_outer']):
        # NOTE: Jax/Haiku is functional; so `local_params` can be accessed
        # by all clients simultaneously as long as we dont overwrite it.
        new_local_params = [None] * self.num_clients
        key = random.fold_in(key, j)
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
          params_mat = _stack_params(local_params)
          opt_state = opt.init_fn(local_params[t])
          for batch_idx, batch in enumerate(batches):
            key, opt_state = batch_update(key, batch_idx, opt_state, params_mat, t,
                                          Sigma, batch, self.noise_mults[t])
          # Record new model and model diff
          new_local_params[t] = opt.params_fn(opt_state)

        # After every Mocha outer iteration, updates all weights together
        local_params = new_local_params

      # Update Sigma after `mocha_outer` iterations of simultaneous client updates
      # (i.e. communication rounds)
      Sigma = update_sigma(local_params)

      if i % self.args['eval_every'] == 0:
        train_metric, test_metric = self.eval(local_params, i)

    return train_metric, test_metric