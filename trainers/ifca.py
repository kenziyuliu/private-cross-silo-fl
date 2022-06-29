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

from trainers.ifca_mrmtl import IFCA_MRMTL


class IFCA(IFCA_MRMTL):
  def __init__(self, args, data):
    print(f'[INFO] Running IFCA: overwriting ifca_select_frac = ifca_fedavg_frac = 1')
    args['ifca_select_frac'] = args['ifca_fedavg_frac'] = 1
    self.lam = 0.0  # NOTE: IFCA should not use lambda.
    self.iter_desc = '[IFCA]'
    super().__init__(args, data)

