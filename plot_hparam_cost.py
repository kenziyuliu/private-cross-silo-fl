import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import tensorflow_privacy as tfp
from num2tex import num2tex

import dp_accounting

RDP_ORDERS = dp_accounting.RDP_ORDERS

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', type=str)
parser.add_argument('-K', type=int, default=20)
parser.add_argument('-n', type=int, default=200)
parser.add_argument('-eps', type=float, default=0.25)
parser.add_argument('-sig', type=float, default=0.4)
parser.add_argument('-tau', type=float, default=0.08)
parser.add_argument('-delta', type=float, default=1e-5)
parser.add_argument('-etas', nargs='+', type=float, default=[-0.5, 0, 0.5, 1, 2, 5])
parser.add_argument('-cf', '--closed_form', action='store_true')
parser.add_argument('--nolegend', action='store_true')
args = parser.parse_args()


def plt_setup(legendsize=12,
              figsize=(5, 4),
              labelspacing=0.3,
              tick_size=12,
              axes_size=13,
              markersize=5):
  matplotlib.rcParams['font.family'] = "sans-serif"
  matplotlib.rcParams['font.sans-serif'] = "Arial"
  matplotlib.rc('text', usetex=True)
  if markersize:
    matplotlib.rcParams['lines.markersize'] = markersize

  plt.rc('font', size=14)  # controls default text sizes
  plt.rc('axes', titlesize=16)  # fontsize of the axes title
  plt.rc('axes', labelsize=axes_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=tick_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=tick_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=legendsize)  # legend fontsize
  if labelspacing:
    plt.rc('legend', labelspacing=labelspacing)
  plt.rc('figure', titlesize=16)  # fontsize of the figure title
  plt.rc('figure', figsize=figsize)


def errorfill(x,
              y,
              yerr,
              fmt=None,
              color=None,
              alpha_fill=0.15,
              ax=None,
              label=None,
              **line_kwargs):
  ax = ax if ax is not None else plt.gca()
  x, y, yerr = map(np.array, [x, y, yerr])
  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr

  opts, kwopts = [], {}
  if fmt is not None:
    opts.append(fmt)
  if color is not None:
    kwopts['color'] = color

  ax.plot(x, y, *opts, **kwopts, **line_kwargs, label=label)
  ax.fill_between(x, ymax, ymin, **kwopts, alpha=alpha_fill, linewidth=0)


def compute_noise(target_eps, target_delta=args.delta, sampling_rate=1, steps=1):
  return dp_accounting.compute_noise(target_eps=target_eps,
                                     target_delta=target_delta,
                                     sampling_rate=sampling_rate,
                                     steps=steps)


def compute_eps(noise_mult, target_delta=args.delta, sampling_rate=1, steps=1):
  return dp_accounting.compute_eps(noise_mult=noise_mult,
                                   target_delta=target_delta,
                                   sampling_rate=sampling_rate,
                                   steps=steps)


def trunc_neg_binom_expectation(eta, gamma):
  """Definition 1 of https://arxiv.org/pdf/2110.03620.pdf."""
  if eta == 0:
    return (1.0 / gamma - 1) / np.log(1.0 / gamma)
  else:
    return eta * (1 - gamma) / (gamma * (1 - gamma**eta))


def trunc_neg_binom_gamma(expectation, eta):
  """
  Numerically solve for gamma of the truncated negative binomial distribution,
  given eta and the target expectation (number of runs).
  """
  def opt_fn(gamma):
    cur_expectation = trunc_neg_binom_expectation(eta, gamma)
    return (expectation - cur_expectation)**2

  chosen_gamma = optimize.minimize_scalar(opt_fn, method='bounded', bounds=(1e-10, 1 - 1e-10))
  if not chosen_gamma.success:
    raise ValueError(f'Cannot find gamma for expectation={expectation}, eta={eta}')
  return chosen_gamma.x


def trunc_neg_binom_tuning_rdp(eta, gamma, orders, rdps):
  """
  Negative Binomial Distribution considers 2 RDP guarantees at a time;
  Do a naive nested loop thru the (order, rdp) pairs, and obtain a list of
  (order, min(rdp)) pairs.
  Theorem 2 of https://arxiv.org/pdf/2110.03620.pdf.
  """
  assert len(orders) == len(rdps)
  expected_k = trunc_neg_binom_expectation(eta, gamma)
  final_rdps = []
  for order1, rdp in zip(orders, rdps):
    rdp_primes = []
    for order2, rdp2 in zip(orders, rdps):
      rdp_prime = rdp + (1 + eta) * (1 - 1 / order2) * rdp2
      rdp_prime += (1 + eta) * np.log(1 / gamma) / order2
      rdp_prime += np.log(expected_k) / (order1 - 1)
      rdp_primes.append(rdp_prime)
    final_rdps.append(min(rdp_primes))

  return orders, final_rdps


def compute_eps_tuning(noise_mult, target_delta, eta=None, gamma=None, orders=RDP_ORDERS):
  if noise_mult == 0:
    return float('inf')
  # Base RDPs (privacy without tuning)
  rdps = tfp.compute_rdp(q=1, steps=1, noise_multiplier=noise_mult, orders=orders)
  # Update RDP with tuning cost
  orders, rdps = trunc_neg_binom_tuning_rdp(eta, gamma, orders, rdps)
  # Convert back to final epsilon in approx DP
  eps, delta, opt_order = tfp.get_privacy_spent(orders, rdps, target_delta=target_delta)
  return eps


def compute_noise_tuning(target_eps,
                         target_delta=args.delta,
                         eta=None,
                         gamma=None,
                         orders=RDP_ORDERS):
  def opt_fn(noise_mult):
    cur_eps = compute_eps_tuning(noise_mult, target_delta, eta=eta, gamma=gamma, orders=orders)
    return (target_eps - cur_eps)**2

  chosen_nm = optimize.minimize_scalar(opt_fn, method='bounded', bounds=(1e-5, 1e5))

  if not chosen_nm.success:
    raise ValueError(f'Cannot find suitable noise for the input params: eps={target_eps}, '
                     f'delta={target_delta}, q={sampling_rate}, steps={steps}')
  return chosen_nm.x


######################
### Helper classes ###
######################

# class Estimator():
#   def __init__(self, w_local, lam):
#     self.w_bar = np.mean(w_local)  # w_bar is the optimal global model
#     self.w_local = w_local  # (K,)
#     self.K = len(w_local)
#     self.lam = lam  # lam can be ndarray for vectorization
#     self.w_mtl = (1 / (1 + lam)) * self.w_local + (lam / (1 + lam)) * self.w_bar

#   def get_mtl_loss(self, w_true):
#     return (self.w_mtl - w_true)**2


class VectorizedEstimator():
  def __init__(self, w_local, lam):
    # w_local: (num_trials, K), lam: scalar
    self.w_bar = np.mean(w_local, axis=1)  # w_bar is client average; (num_trials,)
    self.w_local = w_local
    self.K = w_local.shape[1]
    # w_mtl: (num_trials, K)
    self.w_mtl = (1 / (1 + lam)) * self.w_local + (lam / (1 + lam)) * self.w_bar[:, None]

  def get_mtl_loss(self, w_true):
    # w_true: (num_trials, K), out: (num_trials, K); squared norm difference as loss
    return (self.w_mtl - w_true)**2


class Simulator():
  def __init__(self,
               theta=0,
               tau=0.1,
               sigma=1,
               sigma_dp=0.0,
               clip=float('inf'),
               K=20,
               n=20,
               num_trials=1000):
    print(f'Running simulator with theta={theta}, sig={sigma}, sig_dp={sigma_dp}',
          f'clip={clip}, K={K}, n={n}, num_trials={num_trials}')
    self.num_trials = num_trials
    self.sigma_loc2 = (sigma**2 + sigma_dp**2 / n) / n
    self.K = K
    self.n = n
    self.tau = tau
    self.optimal_lambda = self.sigma_loc2 / tau**2

    # true client data centers w_k, shape (num_trials, K)
    self.w_underlying = np.random.normal(loc=theta, scale=tau, size=(num_trials, K))

    data_shape = (n, ) + self.w_underlying.shape  # (n, num_trials, K)
    self.data = np.random.normal(loc=self.w_underlying, scale=sigma, size=data_shape)
    # (n, num_trials, K)
    self.clipped_data = self.data * np.minimum(1, clip / np.abs(self.data))
    # (num_trials, K)
    self.clipped_sum = np.sum(self.clipped_data, axis=0)
    self.noisy_sum = self.clipped_sum + np.random.normal(scale=sigma_dp,
                                                         size=self.clipped_sum.shape)
    self.w_hat = self.noisy_sum / n  # (num_trials, K)

  def run(self, lambdas):
    silo_error_avg = []
    silo_error_std = []
    self.lam_losses = []
    for lam in lambdas:
      est = VectorizedEstimator(self.w_hat, lam)
      losses = est.get_mtl_loss(self.w_underlying)  # (num_trials, K)
      self.lam_losses.append(losses)
      silo_error_avg.append(np.mean(losses))  # Mean across trials and silos
      silo_error_std.append(np.std(np.mean(losses, axis=0)))  # mean across trials then take std
    return silo_error_avg, silo_error_std


def plot_main():

  lambdas = np.array([0] + list(np.logspace(-3, 2, num=100)))

  def plot_sim(sim, label, ax=None, color=None, plot_low_endpoint=False):
    print('[plot_sim]', label)
    if args.closed_form:
      mse = (1 - 1 / sim.K) * (sim.sigma_loc2 + lambdas**2 * sim.tau**2) / (lambdas + 1)**2
      mse = mse + sim.sigma_loc2 / K
      std = np.zeros_like(mse)
    else:
      mse, std = sim.run(lambdas=lambdas)

    errorfill(lambdas, mse, std, ax=ax, label=label, color=color)
    if plot_low_endpoint:
      local_mse, fedavg_mse = mse[0], mse[1]
      endpoint_mse = min(local_mse, fedavg_mse)

      ax.axhline(y=nonpriv_mean,
                 color=last_color,
                 linestyle=linestyle,
                 linewidth=1,
                 label=nonpriv_label)
      ax.axhspan(nonpriv_mean - nonpriv_std,
                 nonpriv_mean + nonpriv_std,
                 alpha=0.15,
                 color=last_color)

  # Configs
  num_trials = 500
  K = args.K or 20
  n = args.n or 200
  sigma = args.sig or 0.4
  tau = args.tau or 0.08
  target_eps = args.eps or 0.25
  target_delta = args.delta or 1e-5
  clip = 1

  # Private without tuning
  target_delta_tex = str(num2tex(args.delta)).strip('\\times ')
  noise_mult = compute_noise(target_eps, target_delta, steps=1, sampling_rate=1)
  sigma_dp = clip * noise_mult
  print(f'Params: sigma={sigma}, tau={tau}, n={n}, '
        f'sigma_dp={sigma_dp}, clip={clip}, eps={target_eps}, delta={target_delta}')

  # Private with tuning; use expectation = 10
  expected_tune = 10
  etas = args.etas
  gammas = [trunc_neg_binom_gamma(expected_tune, eta) for eta in etas]
  sigma_tunes = [
      clip * compute_noise_tuning(target_eps, target_delta, eta=eta, gamma=gamma)
      for eta, gamma in zip(etas, gammas)
  ]

  # Non-private
  nonpriv_sim = Simulator(tau=tau, sigma=sigma, K=K, n=n, num_trials=num_trials)
  # Private without tuning
  priv_sim = Simulator(tau=tau,
                       sigma=sigma,
                       sigma_dp=sigma_dp,
                       clip=clip,
                       K=K,
                       n=n,
                       num_trials=num_trials)
  # Private with tuning
  tune_sims = [
      Simulator(tau=tau,
                sigma=sigma,
                sigma_dp=sigma_tune,
                clip=clip,
                K=K,
                n=n,
                num_trials=num_trials) for sigma_tune in sigma_tunes
  ]

  plt_setup(figsize=(3, 3))
  plot_sim(nonpriv_sim, rf'Non-private', color='tab:blue')
  plot_sim(priv_sim,
           rf'Private ($\varepsilon={target_eps}, \delta={target_delta_tex}$)',
           color='tab:green')

  tune_colors = plt.cm.OrRd(np.linspace(0.4, 0.8, len(etas)))

  for sim, eta, gamma, color in zip(tune_sims, etas, gammas, tune_colors):
    plot_sim(sim, rf'Private, TNB ($\eta={eta}, \gamma={gamma:.3f}$)', color=color)

  plt.xlabel(r'$\lambda$', fontsize=16)
  plt.ylabel('Generalization Error', fontsize=16)

  plt.xscale('log')
  plt.yscale('log')
  plt.xlim(min(lambdas[lambdas > 0]), max(lambdas))

  if not args.nolegend:
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)

  plt.grid(which='both', alpha=0.3)

  fname = args.out or f'test_hparam_cost_tau{tau}'
  os.makedirs('figures/', exist_ok=True)
  plt.savefig(f'figures/{fname}.pdf', bbox_inches='tight')
  plt.savefig(f'figures/{fname}.png', bbox_inches='tight', dpi=300)
  print(f'Plots saved to figures/{fname}.png and figures/{fname}.pdf')
  plt.show()


if __name__ == '__main__':
  plot_main()
