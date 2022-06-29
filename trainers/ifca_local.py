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


class IFCA_Local(IFCA_MRMTL):
  def __init__(self, args, data):
    print(f"""
    [INFO] Running IFCA + Local:
      (1) overwriting ifca_fedavg_frac = ifca_select_frac
      (2) setting lam = 0 to reduce MR-MTL to local training
    """)
    # `ifca_select_frac` should be specified as part of input.
    args['ifca_fedavg_frac'] = args['ifca_select_frac']
    self.lam = 0.0  # Reduce to local training after IFCA pre-conditioning
    self.iter_desc = '[IFCA+Local]'
    super().__init__(args, data)

