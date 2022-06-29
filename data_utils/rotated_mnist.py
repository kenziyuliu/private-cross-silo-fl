"""
Create rotated MNIST dataset for cross-silo FL.
Data loading code based on https://github.com/google/jax/blob/main/examples/datasets.py.
"""
import array  # Python array std library
import gzip
import os
from os import path
import struct
import urllib

import matplotlib.pyplot as plt
import numpy as np


# CVDF mirror of http://yann.lecun.com/exdb/mnist/
MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# SAVE_DIR = './data/rotated_mnist'
DATA_DIR = '/tmp/datasets/rotated_mnist'
ROOT_SEED = int((np.e ** np.euler_gamma) ** np.pi * 1000)


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
  out_file = path.join(DATA_DIR, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, DATA_DIR))


def mnist(normalize=True, permute_train=False):
  """Download parse and process MNIST data to unit scale."""
  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, numDATA_DIR, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(numDATA_DIR, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(MNIST_URL + filename, filename)

  train_images = parse_images(path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))

  if normalize:
    train_images = train_images / np.float32(255.)
    test_images = test_images / np.float32(255.)

  if permute_train:
    perm = np.random.RandomState(ROOT_SEED).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def rotated_mnist(num_rotations=4, num_clients=40, inrot_shuffle=True, seed=ROOT_SEED,
                  save_dir=None):
  """Apply rotation to MNIST (no augmentation), split it into data siloes, and save for later."""
  train_images, train_labels, test_images, test_labels = mnist(permute_train=False)
  # Assume the examples can be evenly divided among clients
  assert len(train_images) % num_clients == len(train_labels) % num_clients == 0
  assert len(test_images) % num_clients == len(test_labels) % num_clients == 0
  # Also assume clients can be evenly divided into clusters; this ensures that
  # each client only have data from a certain rotation.
  assert num_clients % num_rotations == 0

  # Shuffle MNIST before rotations
  if inrot_shuffle:
    train_perm = np.random.RandomState(seed).permutation(len(train_images))
    test_perm = np.random.RandomState(seed + 1).permutation(len(test_images))
    train_images = train_images[train_perm]
    train_labels = train_labels[train_perm]
    test_images = test_images[test_perm]
    test_labels = test_labels[test_perm]

  # Cut images into `num_rotations` chunks, and rotate each chunk differently
  for k in range(num_rotations):
    left = k * len(train_images) // num_rotations
    right = (k + 1) * len(train_images) // num_rotations
    train_rot = np.rot90(train_images[left:right], axes=(1, 2), k=k)
    train_images[left:right] = train_rot

    left = k * len(test_images) // num_rotations
    right = (k + 1) * len(test_images) // num_rotations
    test_rot = np.rot90(test_images[left:right], axes=(1, 2), k=k)
    test_images[left:right] = test_rot

  # Split data into clients directly (we ensured that each client can have
  # data from the same rotation).
  # NOTE: also add a channel dimension for compatibility with Haiku conv.
  train_images = train_images.reshape(num_clients, -1, *train_images.shape[-2:], 1)
  train_labels = train_labels.reshape(num_clients, -1)
  test_images = test_images.reshape(num_clients, -1, *test_images.shape[-2:], 1)
  test_labels = test_labels.reshape(num_clients, -1)

  if save_dir is not None:
    if not path.exists(save_dir):
      os.makedirs(save_dir)
    np.save(path.join(save_dir, 'train_images'), train_images)
    np.save(path.join(save_dir, 'train_labels'), train_labels)
    np.save(path.join(save_dir, 'test_images'), test_images)
    np.save(path.join(save_dir, 'test_labels'), test_labels)
    print(f'Saved rotated MNIST to {save_dir}')

  return train_images, train_labels, test_images, test_labels


def rotated_patched_mnist(noise_level=0.5, patch_size=7, seed=ROOT_SEED, save_dir=None):
  # Shapes: data (K, n, 28, 28, 1), label (K, n)
  train_images, train_labels, test_images, test_labels = rotated_mnist()
  assert train_images.ndim == test_images.ndim == 5
  K, n, w, h, c = train_images.shape
  # For each client, add a client-specific heterogeneity (to make clients within
  # the same cluster further from each other).
  assert len(train_images) == len(train_labels) == len(test_images) == len(test_labels)

  rand = np.random.default_rng(seed=seed)
  mask_shape = (K, int(np.ceil(w / patch_size)), int(np.ceil(h / patch_size)), c)
  ### Noise level as fraction of area masked away ###
  noise_masks = rand.choice([0, 1], size=mask_shape, p=[1 - noise_level, noise_level])
  # Make patches via kronecker product
  noise_patches = np.kron(noise_masks, np.ones((patch_size, patch_size, 1)))
  noise_patches = noise_patches[:, :w, :h, :]   # Crop if necessary
  train_images = np.maximum(train_images, noise_patches[:, None])
  test_images = np.maximum(test_images, noise_patches[:, None])

  # Normalize again
  def min_max_normalize(images):
    min_val = np.min(images, axis=(2, 3, 4), keepdims=True)
    max_val = np.max(images, axis=(2, 3, 4), keepdims=True)
    return (images - min_val) / (max_val - min_val)

  train_images = min_max_normalize(train_images)
  test_images = min_max_normalize(test_images)

  if save_dir is not None:
    if not path.exists(save_dir):
      os.makedirs(save_dir)
    suffix = f'noise{noise_level}_patch{patch_size}_area'
    np.save(path.join(save_dir, f'train_images_{suffix}'), train_images)
    np.save(path.join(save_dir, f'train_labels_{suffix}'), train_labels)
    np.save(path.join(save_dir, f'test_images_{suffix}'), test_images)
    np.save(path.join(save_dir, f'test_labels_{suffix}'), test_labels)
    print(f'Saved rotated + patched MNIST to {save_dir}')

  return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
  xtr, ytr, xte, yte = rotated_mnist()
