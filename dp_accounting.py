import json
import math
import os
import pprint
import pickle
from typing import Optional

import numpy as np
from scipy import optimize
from filelock import FileLock

import tensorflow_privacy as tfp


# RDP orders from tensorflow privacy
RDP_ORDERS = np.array(
    [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) +
    [128, 256, 512])


def compute_eps(noise_mult: float, sampling_rate: float, steps: int, target_delta: float,
                num_selections: Optional[int] = 0, selection_eps: Optional[float] = None,
                orders=RDP_ORDERS):
  """Combines accounting of DP-SGD training with Private Selection."""
  if noise_mult == 0:
    return float('inf')

  rdp = tfp.compute_rdp(q=sampling_rate,
                        noise_multiplier=noise_mult,
                        steps=steps,
                        orders=orders)

  if num_selections > 0 and selection_eps is not None:
    # Exponential mechanism is (eps^2/8)-zCDP ==> (a, a * eps^2 / 8)-RDP:
    # https://differentialprivacy.org/exponential-mechanism-bounded-range/
    # Just compose the selection cost to the RDP.
    # Note that the accounting does not care about sensitivity.
    selection_rdp = orders / 8 * (selection_eps**2) * num_selections
    assert selection_rdp.shape == rdp.shape
    rdp = rdp + selection_rdp

  eps, delta, opt_order = tfp.get_privacy_spent(orders, rdp, target_delta=target_delta)
  return eps


def compute_noise(target_eps: float,
                  sampling_rate: float,
                  steps: int,
                  target_delta: float,
                  num_selections: Optional[int] = 0,
                  selection_eps: Optional[float] = None,
                  cache_path='data/dp_accounting_cache.pickle'):
  """Computes eps for a given delta using RDP."""
  assert sampling_rate <= 1, f'sampling rate must be in [0, 1]; found {sampling_rate}'

  # Read cache if available
  inp = (target_eps, target_delta, sampling_rate, steps, num_selections, selection_eps)
  cache = {}
  if cache_path is not None:
    cache = _read_cache(cache_path)
    if inp in cache:
      return cache[inp]

  def opt_fn(noise_mult):
    noise_mult = max(1e-5, noise_mult)  # Lower bound the noise to avoid division by 0
    # Upper bound the noise to avoid numerical errors in TFP. This means that
    # the eps is too small that you need a HUGE noise. Capping the noise means
    # we won't actually satisfy the eps, but then at this point the utility ~= 0.
    noise_mult = min(1e6, noise_mult)
    return (target_eps - compute_eps(noise_mult, sampling_rate, steps, target_delta,
                                     num_selections, selection_eps))**2

  # chosen_noise = optimize.minimize_scalar(opt_fn, tol=1e-5)
  chosen_noise = optimize.minimize_scalar(opt_fn)
  chosen_noise, success = chosen_noise.x, chosen_noise.success

  if success:
    # Write to cache if path provided
    if cache_path is not None:
      cache[inp] = chosen_noise
      _write_cache(cache, cache_path)
    return chosen_noise

  raise ValueError(f'Cannot find suitable noise for the input params: eps={target_eps}, '
                   f'delta={target_delta}, q={sampling_rate}, steps={steps}')


def compute_silo_noise_mults(num_clients: int, train_samples: np.ndarray, params: dict, steps_factor=1,
                             num_selections: Optional[int] = 0, selection_eps: Optional[float] = None):
  """DP accounting: compute accounting stats for all clients at once."""
  # `example_dp` just refers to whether or not we use DP (including private selection).
  if not params['example_dp']:
    print('Not using DP, return 0.0 noise mults for all clients')
    return [0.0] * num_clients

  train_samples = np.asarray(train_samples)
  if num_selections or selection_eps:
    print(f'Accounting for private selection: num_select={num_selections}, select_eps={selection_eps}')

  if params['batch_size'] == -1:
    batch_sizes = train_samples
  else:
    batch_sizes = np.minimum(params['batch_size'], train_samples)

  num_rounds = params['num_rounds']
  sampling_rates = batch_sizes / train_samples
  if params['inner_mode'] == 'iter':
    num_steps = np.array([(num_rounds + 1) * params['inner_iters']] * num_clients)
  else:
    tot_epochs = (num_rounds + 1) * params['inner_epochs']
    num_steps = train_samples // batch_sizes * tot_epochs

  # Additional factor for methods that take more steps (e.g. Ditto)
  num_steps *= steps_factor

  target_eps, target_delta = params['ex_eps'], params['ex_delta']
  print(f'Target (eps, delta) = ({target_eps}, {target_delta}), Total steps:', num_steps)

  if target_eps is not None:
    print(f'Target epsilon set to {target_eps} for all silos; computing noise mults...')
    noise_mults = [
        compute_noise(target_eps, q, steps, target_delta, num_selections, selection_eps)
        for q, steps in zip(sampling_rates, num_steps)
    ]
    print(f'Noise mults for eps={target_eps}, delta={target_delta}: {np.round(noise_mults, 4)}')

  else:
    noise_mult = params['ex_noise_mult']
    noise_mults = [noise_mult] * num_clients
    if noise_mult is not None:
      print(f'Fixing noise mults to nm={noise_mult}')
      epsilons = [
          compute_eps(noise_mult, q, steps, target_delta)
          for q, steps in zip(sampling_rates, num_steps)
      ]
      print(f'Resulting eps with noise_mult={noise_mult}, delta={target_delta}: {np.round(epsilons, 4)}')
    else:
      print(f'NO privacy added since target_eps=None and ex_noise_mult not provided.')

  return noise_mults


def _read_cache(cache_path: str):
  """Cache the accounting results with locks as a simple solution for concurrent IO."""
  cache = {}
  if os.path.exists(cache_path):
    with FileLock(cache_path + '.lock') as lock:
      with open(cache_path, 'rb') as f:
        try:
          cache = pickle.load(f)
        except EOFError:
          cache = {}
  return cache


def _write_cache(cache: dict, cache_path: str):
  """Cache the accounting results with locks as a simple solution for concurrent IO."""
  with FileLock(cache_path + '.lock') as lock:
    with open(cache_path, 'wb') as f:
      pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
