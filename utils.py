import os
import random
import time

import numpy as np
import scipy.io
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_utils import rotated_mnist, adni_dataset


def gen_batch(data_x, data_y, batch_size, num_iter):
  """NOTE: Deprecated in favor of `epoch_generator`."""
  index = len(data_y)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data_y)):
      index = 0
      data_x, data_y = sklearn.utils.shuffle(data_x, data_y, random_state=i + 1)

    batched_x = data_x[index:index + batch_size]
    batched_y = data_y[index:index + batch_size]

    yield (batched_x, batched_y)


def epochs_generator(data_x, data_y, batch_size, epochs=1, seed=None):
  for ep in range(epochs):
    gen = epoch_generator(data_x, data_y, batch_size, seed=seed + ep)
    for batch in gen:
      yield batch


def epoch_generator(data_x, data_y, batch_size, seed=None):
  """Generate one epoch of batches."""
  data_x, data_y = sklearn.utils.shuffle(data_x, data_y, random_state=seed)
  # Drop last by default
  epoch_iters = len(data_x) // batch_size
  for i in range(epoch_iters):
    left, right = i * batch_size, (i + 1) * batch_size
    yield (data_x[left:right], data_y[left:right])


def client_selection(seed, total, num_selected, weights=None):
  rng = np.random.default_rng(seed=seed)
  indices = rng.choice(range(total), num_selected, replace=False, p=weights)
  return indices


def print_log(message, fpath=None, stdout=True, print_time=False):
  if print_time:
    timestr = time.strftime('%Y-%m-%d %a %H:%M:%S')
    message = f'{timestr} | {message}'
  if stdout:
    print(message)
  if fpath is not None:
    with open(fpath, 'a') as f:
      print(message, file=f)


#########################
#### Dataset loading ####
#########################


def read_vehicle_data(data_dir='data/vehicle', seed=None, bias=False, density=1.0, standardize=True):
  """Read Vehicle dataset.

  Args:
    data_dir: directory that stores the `vehicle.mat` file
    seed: random seed for generating the train/test split
    bias: whether to insert a column of 1s to the dataset (after standardizing)
        so that a model bias term is implicitly included.
    density: fraction of the training data on each client to keep; this does not
        affect test examples.
  """
  x_trains, y_trains, x_tests, y_tests = [], [], [], []
  mat = scipy.io.loadmat(os.path.join(data_dir, 'vehicle.mat'))
  raw_x, raw_y = mat['X'], mat['Y']  # y in {-1, 1}
  print('Vehicle dataset:')
  print('\tnumber of clients:', len(raw_x), len(raw_y))
  print('\tnumber of examples:', [len(raw_x[i][0]) for i in range(len(raw_x))])
  print('\tnumber of features:', len(raw_x[0][0][0]))
  print('\tSeed of dataset:', seed)
  print(f'\tUsing {density * 100:.2f}% of training data on each client')
  print(f'\tStandardizing using (density adjusted) training statistics: {standardize}')

  for i in range(len(raw_x)):
    features, label = raw_x[i][0], raw_y[i][0].flatten()
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.25, random_state=seed)

    if density != 1:
      num_train_examples = int(density * len(x_train))
      # Randomness should be set for different workers (no explicit seed)
      train_mask = np.random.permutation(len(x_train))[:num_train_examples]
      x_train = x_train[train_mask]  # Mask before fitting standard scaler.
      y_train = y_train[train_mask]
    if standardize:
      # Preprocessing using mean/std from training examples, within each silo.
      scaler = StandardScaler().fit(x_train)
      x_train = scaler.transform(x_train)
      x_test = scaler.transform(x_test)
    if bias:
      x_train = np.c_[x_train, np.ones(len(x_train))]
      x_test = np.c_[x_test, np.ones(len(x_test))]
    x_trains.append(x_train)
    x_tests.append(x_test)

    # binary label can either be ints or floats (float32 suffice)
    y_trains.append(y_train.astype(float))
    y_tests.append(y_test.astype(float))

  # Since different tasks have differnet data, this is a ragged array
  return (np.array(x_trains, dtype=object), np.array(y_trains, dtype=object),
          np.array(x_tests, dtype=object), np.array(y_tests, dtype=object))


def read_gleam_data(data_dir='data/gleam', seed=None, bias=False, density=1.0, standardize=True):
  """Read GLEAM dataset.

  Args:
    data_dir: directory that stores the `vehicle.mat` file
    seed: random seed for generating the train/test split
    bias: whether to insert a column of 1s to the dataset (after standardizing)
        so that a model bias term is implicitly included.
    density: fraction of the training data on each client to keep; this does not
        affect test examples.
  """
  x_trains, y_trains, x_tests, y_tests = [], [], [], []
  mat = scipy.io.loadmat(os.path.join(data_dir, 'gleam.mat'))
  raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y in {-1, 1}
  print('Google glass (GLEAM) dataset:')
  print('number of clients:', len(raw_x), len(raw_y))
  print('number of examples:', [len(raw_x[i]) for i in range(len(raw_x))])
  print('number of features:', len(raw_x[0][0]))
  print('Seed of dataset:', seed)
  print(f'Keeping {density * 100:.2f}% of training data on each client')
  print(f'Standardizing using (density adjusted) training statistics: {standardize}')

  for i in range(len(raw_x)):
    features, label = raw_x[i], raw_y[i].flatten()
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.25, random_state=seed)

    if density != 1:
      # Randomness should be set for different workers (no explicit seed)
      num_train_examples = int(density * len(x_train))
      train_mask = np.random.permutation(len(x_train))[:num_train_examples]
      x_train = x_train[train_mask]
      y_train = y_train[train_mask]
    if standardize:
      # Preprocessing using mean/std from training examples, within each silo
      scaler = StandardScaler().fit(x_train)
      x_train = scaler.transform(x_train)
      x_test = scaler.transform(x_test)
    if bias:
      # Append a column of ones to implicitly include bias
      x_train = np.c_[x_train, np.ones(len(x_train))]
      x_test = np.c_[x_test, np.ones(len(x_test))]
    x_trains.append(x_train)
    x_tests.append(x_test)

    # binary label can either be ints or floats (float32 suffice)
    y_trains.append(y_train.astype(float))
    y_tests.append(y_test.astype(float))

  # Since different tasks have differnet data, this is a ragged array
  return (np.array(x_trains, dtype=object), np.array(y_trains, dtype=object),
          np.array(x_tests, dtype=object), np.array(y_tests, dtype=object))


def read_school_data(data_dir='data/school', test_frac=0.3, seed=None, bias=False, standardize=True, **__kwargs):
  """Read School dataset."""
  x_trains, y_trains, x_tests, y_tests = [], [], [], []
  mat = scipy.io.loadmat(os.path.join(data_dir, 'school.mat'))
  # Note that the raw data structure is different from vehicles
  raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y is exam score
  print('School dataset:')
  print('number of clients:', len(raw_x), len(raw_y))
  print('number of examples:', [len(raw_x[i]) for i in range(len(raw_x))])
  print('number of features:', len(raw_x[0][0]))

  for i in range(len(raw_x)):   # For each client
    features, label = raw_x[i], raw_y[i].flatten()
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=test_frac, random_state=seed)

    if standardize:
      # Preprocessing using mean/std from training examples, within each silo
      scaler = StandardScaler().fit(x_train)
      x_train = scaler.transform(x_train)
      x_test = scaler.transform(x_test)
      # For y (scores), use min/max normalization
      min_y, max_y = 1, 70    # Hardcode stats from dataset.
      y_train = (y_train - min_y) / (max_y - min_y)
      y_test = (y_test - min_y) / (max_y - min_y)
    if bias:
      x_train = np.c_[x_train, np.ones(len(x_train))]
      x_test = np.c_[x_test, np.ones(len(x_test))]

    # features / exam scores should be float (if not standardized)
    x_trains.append(x_train.astype(float))
    x_tests.append(x_test.astype(float))
    y_trains.append(y_train.astype(float))
    y_tests.append(y_test.astype(float))

  # Since different tasks have differnet data, this is a ragged array
  return (np.array(x_trains, dtype=object), np.array(y_trains, dtype=object),
          np.array(x_tests, dtype=object), np.array(y_tests, dtype=object))


def read_rotated_mnist_data(data_dir='data/rotated_mnist', **__kwargs):
  """Read rotated MNIST data."""
  try:
    x_trains = np.load(os.path.join(data_dir, 'train_images.npy'))
    y_trains = np.load(os.path.join(data_dir, 'train_labels.npy'))
    x_tests = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_tests = np.load(os.path.join(data_dir, 'test_labels.npy'))
  except FileNotFoundError:
    x_trains, y_trains, x_tests, y_tests = rotated_mnist.rotated_mnist(save_dir=data_dir)

  assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
  num_clients = len(x_trains)
  print('Rotated MNIST dataset:')
  print('\tnumber of clients:', num_clients)
  print('\tnumber of train examples:', [len(x_trains[i]) for i in range(num_clients)])
  print('\tnumber of test examples:', [len(x_tests[i]) for i in range(num_clients)])
  print('\tnumber of features:', x_trains[0][0].shape)
  return x_trains, y_trains, x_tests, y_tests



def read_rotated_patched_mnist_data(data_dir='data/rotated_patched_mnist',
                                    noise_level=0.5, patch_size=7,
                                    **__kwargs):
  """Read rotated and patched MNIST data."""
  suffix = f'noise{noise_level}_patch{patch_size}_area'
  try:
    x_trains = np.load(os.path.join(data_dir, f'train_images_{suffix}.npy'))
    y_trains = np.load(os.path.join(data_dir, f'train_labels_{suffix}.npy'))
    x_tests = np.load(os.path.join(data_dir, f'test_images_{suffix}.npy'))
    y_tests = np.load(os.path.join(data_dir, f'test_labels_{suffix}.npy'))
  except FileNotFoundError:
    x_trains, y_trains, x_tests, y_tests = rotated_mnist.rotated_patched_mnist(
        noise_level=noise_level, save_dir=data_dir)

  assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
  num_clients = len(x_trains)
  print(f'Rotated + Patched MNIST dataset ({suffix}):')
  print('\tnumber of clients:', num_clients)
  print('\tnumber of train examples:', [len(x_trains[i]) for i in range(num_clients)])
  print('\tnumber of test examples:', [len(x_tests[i]) for i in range(num_clients)])
  print('\tnumber of features:', x_trains[0][0].shape)
  return x_trains, y_trains, x_tests, y_tests


def read_adni_data(data_dir='data/adni', seed=None, density=1.0, **__kwargs):
  print(f'ADNI dataset:')
  try:
    tag = f'_seed{seed}' if seed is not None else ''
    print(f'Loading dataset with tag "{tag}"...')
    # `allow_pickle` since we saved np object arrays (for ragged arrays)
    x_trains = np.load(os.path.join(data_dir, f'train_images{tag}.npy'), allow_pickle=True)
    y_trains = np.load(os.path.join(data_dir, f'train_labels{tag}.npy'), allow_pickle=True)
    x_tests = np.load(os.path.join(data_dir, f'test_images{tag}.npy'), allow_pickle=True)
    y_tests = np.load(os.path.join(data_dir, f'test_labels{tag}.npy'), allow_pickle=True)
    print('Loaded preprocessed ADNI dataset')
  except FileNotFoundError:
    print('Preprocessing ADNI dataset...')
    x_trains, y_trains, x_tests, y_tests = adni_dataset.read_data(seed=seed, save_dir=data_dir)

  # Check number of clients
  assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
  num_clients = len(x_trains)

  # Do deterministic subsampling since dataset was shuffled during construction.
  if density < 1:
    print(f'Subsampling training sets to {density}')
    for i in range(num_clients):
      x_trains[i] = x_trains[i][:int(len(x_trains[i]) * density)]
      y_trains[i] = y_trains[i][:int(len(y_trains[i]) * density)]

  print('\tnumber of clients:', num_clients)
  print('\tnumber of train examples:', [len(x_trains[i]) for i in range(num_clients)])
  print('\tnumber of test examples:', [len(x_tests[i]) for i in range(num_clients)])
  print('\tnumber of features:', x_trains[0][0].shape)
  return x_trains, y_trains, x_tests, y_tests
