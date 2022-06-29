import functools
import os

import numpy as np
from tqdm import tqdm
from jax import jit
import jax.numpy as jnp
import haiku as hk

from utils import gen_batch, print_log
import utils
import jax_utils
import model_utils
import dp_accounting


class BaseTrainer:
  def __init__(self, args, data):
    x_train, y_train, x_test, y_test = data
    self.args = args
    self.seed = args['seed']

    ###### Client #######
    self.data = data
    self.num_clients = len(x_train)  # number of clients / tasks
    # self.clients_per_round = args['clients_per_round']
    # if self.clients_per_round == -1:
    #   self.clients_per_round = self.num_clients

    ###### Training configs #######
    self.lam = args['lambda']
    self.num_rounds = args['num_rounds']
    self.inner_mode = args['inner_mode']
    self.inner_epochs = args['inner_epochs']
    self.inner_iters = args['inner_iters']
    self.lr = args['learning_rate']
    self.l2_reg = args['l2_reg']
    self.dataset = args['dataset']
    self.train_samples = np.asarray([len(x) for x in x_train])
    self.test_samples = np.asarray([len(x) for x in x_test])
    self.l2_clip = self.args['ex_clip']
    self.num_clusters = args['num_clusters']

    if args['unweighted_updates']:
      self.update_weights = np.ones_like(self.train_samples)
    else:
      self.update_weights = self.train_samples / np.sum(self.train_samples)

    self.batch_size = args['batch_size']
    if self.batch_size == -1:
      # Full batch gradient descent if needed
      self.batch_sizes = [len(x_train[i]) for i in range(self.num_clients)]
    else:
      # Limit batch size to the dataset size, so downstream calculations (e.g. sample rate) don't break
      self.batch_sizes = [min(len(x_train[i]), self.batch_size) for i in range(self.num_clients)]

    ###### DP configs #######
    self.use_dp = self.args['example_dp']
    # NOTE: hack: trying to save compute as private selection needs to recompute noise_mults.
    if 'ifca' not in self.args['trainer'].lower():
      self.noise_mults = dp_accounting.compute_silo_noise_mults(self.num_clients, self.train_samples, args)

    # Set loss function and model. For now, fixed model arch for every task.
    if self.dataset in ('vehicle', 'gleam'):
      self.data_loss_fn = functools.partial(jax_utils.hinge_loss_hk, reg=self.args['lam_svm'])
      self.pred_fn = jax_utils.linear_svm_classify
      self.model_fn = model_utils.linear_model_fn

    elif self.dataset == 'school':
      self.data_loss_fn = jax_utils.l2_loss_hk
      self.pred_fn = jax_utils.regression_pred
      self.model_fn = model_utils.linear_model_fn

    elif self.dataset == 'adni':
      self.data_loss_fn = jax_utils.l2_loss_hk
      self.pred_fn = jax_utils.regression_pred
      self.model_fn = model_utils.adni_model_fn

    elif self.dataset in ('rotated_mnist', 'rotated_patched_mnist'):
      self.data_loss_fn = jax_utils.sce_loss_hk
      self.pred_fn = jax_utils.multiclass_classify
      self.model_fn = model_utils.mnist_model_fn

    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

    # Create model architecture & compile prediction/loss function
    self.model = hk.without_apply_rng(hk.transform(self.model_fn))
    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))

    # (DEPRECATED) Batch-wise local data generators; deprecated to use local epochs.
    self.batch_gen = {}
    for i in range(self.num_clients):
      self.batch_gen[i] = gen_batch(x_train[i], y_train[i], self.batch_size,
                                    num_iter=(self.num_rounds + 1) * self.inner_iters)
    self.train_data = self.batch_gen  # Legacy.

    self.x_train, self.y_train = x_train, y_train
    self.x_test, self.y_test = x_test, y_test

    with np.printoptions(precision=4):
      print('[DEBUG] client update weights', self.update_weights)

  def train(self):
    raise NotImplementedError(f'BaseTrainer train() needs to be implemented')

  def eval(self, local_params, round_idx):
    # Regression
    if self.args['is_regression']:
      train_losses, test_losses = [], []
      for t in range(self.num_clients):
        train_preds = self.pred_fn(local_params[t], self.x_train[t])
        train_losses.append(np.mean((train_preds - self.y_train[t])**2))
        test_preds = self.pred_fn(local_params[t], self.x_test[t])
        test_losses.append(np.mean((test_preds - self.y_test[t])**2))
      avg_train_metric = np.average(train_losses, weights=self.train_samples)
      avg_test_metric = np.average(test_losses, weights=self.test_samples)

    # Classification
    else:
      train_losses, test_losses = [], []
      num_correct_train, num_correct_test = [], []
      for t in range(self.num_clients):
        # Train
        train_preds = self.pred_fn(local_params[t], self.x_train[t])
        num_correct_train.append(jnp.sum(train_preds == self.y_train[t]))
        train_loss = self.data_loss_fn(local_params[t], (self.x_train[t], self.y_train[t]))
        train_losses.append(train_loss)
        # Test
        test_preds = self.pred_fn(local_params[t], self.x_test[t])
        num_correct_test.append(jnp.sum(test_preds == self.y_test[t]))
        test_loss = self.data_loss_fn(local_params[t], (self.x_test[t], self.y_test[t]))
        test_losses.append(test_loss)

      avg_train_metric = np.sum(np.array(num_correct_train)) / np.sum(self.train_samples)
      avg_test_metric = np.sum(np.array(num_correct_test)) / np.sum(self.test_samples)
      avg_train_loss = np.average(train_losses, weights=self.train_samples)
      avg_test_loss = np.average(test_losses, weights=self.test_samples)

    if not self.args['quiet']:
      print(f'Round {round_idx}, avg train metric: {avg_train_metric:.5f},'
            f'avg test metric: {avg_test_metric:.5f}')

    # Save only 5 decimal places
    utils.print_log(np.round([avg_train_metric, avg_test_metric], 5).tolist(),
                    stdout=False,
                    fpath=os.path.join(self.args['outdir'], 'output.txt'))
    if not self.args['is_regression']:
      utils.print_log(np.round([avg_train_loss, avg_test_loss], 5).tolist(),
                      stdout=False,
                      fpath=os.path.join(self.args['outdir'], 'losses.txt'))

    return avg_train_metric, avg_test_metric


class BaseTrainerLocal(BaseTrainer):
  def __init__(self, args, data):
    super().__init__(args, data)


class BaseTrainerGlobal(BaseTrainer):
  def __init__(self, params, data):
    super().__init__(params, data)

  def eval(self, params, round_idx):
    local_params = [params] * self.num_clients
    return super().eval(local_params, round_idx)
