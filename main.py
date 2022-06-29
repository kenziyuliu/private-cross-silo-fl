import argparse
import collections
import gc
import multiprocessing as mp
import os
import pprint
import random
import sys
import time

import numpy as np

import utils
from trainers.finetune import Finetune
from trainers.fedavg import FedAvg
from trainers.local import Local
from trainers.ifca import IFCA
from trainers.mrmtl import MRMTL
from trainers.ditto import Ditto
from trainers.ifca_mrmtl import IFCA_MRMTL
from trainers.ifca_fedavg import IFCA_FedAvg
from trainers.ifca_finetune import IFCA_Finetune
from trainers.ifca_local import IFCA_Local
from trainers.mocha import Mocha


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer',
                        help='algorithm to run',
                        type=str,
                        choices=('local', 'finetune', 'fedavg', 'mocha',
                                 'mrmtl', 'ditto', 'ifca', 'ifca_mrmtl',
                                 'ifca_fedavg', 'ifca_local', 'ifca_finetune'),
                        default='fedavg')
    # Datasets
    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=('vehicle', 'school', 'gleam', 'adni',
                                 'rotated_mnist', 'rotated_patched_mnist'),
                        type=str,
                        required=True)
    parser.add_argument('--density',
                        type=float,
                        help='Fraction of the local training data to use (for each silo)',
                        default=1.0)
    parser.add_argument('--no_std',
                        help='Disable dataset standardization (vehicle and gleam only)',
                        action='store_true')
    # Learning
    parser.add_argument('-t', '--num_rounds',
                        help='number of communication rounds',
                        type=int,
                        default=400)
    parser.add_argument('--seed',
                        help='root seed for randomness',
                        type=int,
                        default=0)
    parser.add_argument('-lr', '--learning_rate',
                        help='client learning rate for local training',
                        type=float,
                        default=0.01)
    parser.add_argument('--lrs',
                        help='sweep client learning rate',
                        nargs='+',
                        type=float)
    parser.add_argument('--lambda',
                        help='parameter for personalization',
                        type=float,
                        default=0.0)
    parser.add_argument('--lambdas',
                        help='sweep lambda values',
                        nargs='+',
                        type=float)
    parser.add_argument('--l2_reg',
                        help='L2 regularization',
                        type=float,
                        default=0.0)
    parser.add_argument('--lam_svm',
                        help='regularization parameter for linear SVM',
                        type=float,
                        default=0.0)  # this param is kept the same for all methods and for all runs
    parser.add_argument('-ee', '--eval_every',
                        help='evaluate every `eval_every` rounds;',
                        type=int,
                        default=1)
    # parser.add_argument('--clients_per_round',
    #                     help='number of clients trained per round; -1 means use all clients',
    #                     type=int,
    #                     default=-1)
    parser.add_argument('--batch_size',
                        help='batch size for client optimization',
                        type=int,
                        default=32)
    parser.add_argument('--inner_mode',
                        help='How to run inner loop (fixed no. of batches or epochs)',
                        type=str,
                        choices=('iter', 'epoch'),
                        default='epoch')
    parser.add_argument('--inner_epochs',
                        help='number of epochs per communication round',
                        type=int,
                        default=1)
    parser.add_argument('--inner_iters',
                        help='number of inner iterations per communication round',
                        type=int,
                        default=1)
    parser.add_argument('--unweighted_updates',
                        help='Disable weighing client model updates by their example counts',
                        action='store_true')
    # School dataset flags
    parser.add_argument('--school_testfrac',
                      type=float,
                      help='Fraction of local datasets to use as test data for School dataset',
                      default=0.2)
    # MNIST dataset flags
    parser.add_argument('--mnist_patch_size',
                        type=int,
                        help='Noisy patch size for rotated + patched MNIST dataset',
                        default=2)
    parser.add_argument('--mnist_patch_noise',
                        type=float,
                        help='Noisy patch noise level for rotated + patched MNIST dataset',
                        default=0.5)
    # Mocha
    parser.add_argument('--mocha_outer',
                        help='number of inner rounds to runs per server update',
                        type=int,
                        default=1)
    # Finetuning
    parser.add_argument('--finetune_frac',
                        type=float,
                        help='Fraction of rounds for fedavg training',
                        default=0.5)
    # IFCA
    parser.add_argument('--ifca_select_frac',
                        type=float,
                        help='Fraction of rounds for IFCA with cluster selection',
                        default=0.1)
    parser.add_argument('--ifca_fedavg_frac',
                        type=float,
                        help='Fraction of rounds for IFCA with fixed clusters (and switch to MRMTL after)',
                        default=None)
    parser.add_argument('--ifca_metric',
                        type=str,
                        choices=('loss', 'acc'),
                        help='Metric to use for cluster selection; original papers use loss, but privacy is better under example-level DP',
                        default='acc')
    parser.add_argument('--ifca_loss_sens',
                        type=float,
                        default=0.1,
                        help='Clip bound for loss values when choosing clusters for regression datasets')
    # Clustering
    parser.add_argument('-k', '--num-clusters',
                        help='Number of clusters for FedAvg (IFCA)',
                        type=int,
                        default=1)
    parser.add_argument('-crp', '--cluster-randprob',
                        help='probability of randomly chooosing a cluster',
                        type=float,
                        default=0)
    parser.add_argument('--crp-gamma',
                        help='probability of randomly chooosing a cluster',
                        type=float,
                        default=0.999)
    # DP args
    parser.add_argument('-edp', '--example-dp',
                        help='Enable example-level DP',
                        action='store_true')
    parser.add_argument('-ec', '--ex-clip',
                        help='L2 norm clipping for example-level DP',
                        type=float,
                        default=None)   # orders of 10s e.g. 20 for example-dp
    parser.add_argument('-enm', '--ex-noise-mult',
                        help='Noise multiplier for example-level DP',
                        type=float,
                        default=None)
    parser.add_argument('-eps', '--ex-eps',
                        help='Epsilon for example-level DP',
                        type=float,
                        default=None)
    parser.add_argument('-del', '--ex-delta',
                        help='Delta for example-level DP',
                        type=float,
                        default=1e-7)
    parser.add_argument('--selection_eps',
                        help='Epsilon for private selection',
                        type=float,
                        default=None)
    # Misc args
    parser.add_argument('-o', '--outdir',
                        help=('Directory to store artifacts, under `logs/`.'),
                        type=str)
    parser.add_argument('-r', '--repeat',
                        help=('Number of times to repeat the experiment'),
                        type=int,
                        default=1)
    parser.add_argument('-q', '--quiet',
                        help='Try not to print things',
                        action='store_true')
    parser.add_argument('--no_per_round_log',
                        help='Disable storing eval metrics',
                        action='store_true')
    parser.add_argument('--num_procs',
                        help='number of parallel processes for mp.Pool()',
                        type=int)
    parser.add_argument('--downsize_pool',
                        help='Downsize the multiprocessing pool',
                        action='store_true')

    args = parser.parse_args()
    print(f'Command executed: python3 {" ".join(sys.argv)}')

    if args.outdir is None:
      print(f'Outdir not provided.', end=' ')
      args.outdir = f'logs/{args.trainer}-{time.strftime("%Y-%m-%d--%H-%M-%S")}'
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Storing outputs to {args.outdir}')

    if args.seed is None or args.seed < 0:
      print(f'Random seed not provided.', end=' ')
      args.seed = random.randint(0, 2**32 - 1)
    print(f'Using {args.seed} as global seed.')

    # Privacy
    if args.example_dp and args.selection_eps is None and 'ifca' in args.trainer:
      args.selection_eps = args.ex_eps * 0.03
      print(f'NOTE: Using DP-SGD, but selection_eps=None.'
            f'Default to selection_eps = 0.03 * ex_eps = {args.selection_eps}.')

    # Problem types
    args.is_regression = (args.dataset in ('school', 'adni'))
    args.is_linear = (args.dataset in ('vehicle', 'school', 'gleam'))

    # Record flags and input command.
    # NOTE: the `args.txt` would NOT include the parallel sweep hparams (e.g. lambdas).
    parsed = vars(args)
    with open(os.path.join(args.outdir, 'args.txt'), 'w') as f:
      pprint.pprint(parsed, stream=f)
    with open(os.path.join(args.outdir, 'command.txt'), 'w') as f:
      print(' '.join(sys.argv), file=f)

    print(parsed)
    return parsed


def main(options, run_idx=None):
    options['run_idx'] = run_idx
    # set worker specific config.
    if run_idx is not None:
      options['seed'] += 1000 * run_idx
      options['outdir'] = os.path.join(options['outdir'], f'run{run_idx}')
      os.makedirs(options['outdir'], exist_ok=True)
      print(f'Run {run_idx} uses master seed {options["seed"]}')

    ###########################
    ##### Create Datasets #####
    ###########################

    seed = options['seed']
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    dataset_args = dict(seed=seed, bias=False, density=options['density'],
                        standardize=(not options['no_std']))

    # Read data as ragged arrays with (K, n_i, ...).
    if options['dataset'] == 'vehicle':
      dataset = utils.read_vehicle_data(**dataset_args)
    elif options['dataset'] == 'gleam':
      dataset = utils.read_gleam_data(**dataset_args)
    elif options['dataset'] == 'school':
      dataset = utils.read_school_data(test_frac=options['school_testfrac'], **dataset_args)

    # Image datasets do not take dataset seed; randomness is for params/SGD.
    # The seed for datasets are fixed at data generation time.
    elif options['dataset'] == 'adni':
      dataset = utils.read_adni_data(**dataset_args)
    elif options['dataset'] == 'rotated_mnist':
      dataset = utils.read_rotated_mnist_data(**dataset_args)
    elif options['dataset'] == 'rotated_patched_mnist':
      dataset = utils.read_rotated_patched_mnist_data(
          noise_level=options['mnist_patch_noise'],
          patch_size=options['mnist_patch_size'],
          **dataset_args)
    else:
        raise ValueError(f'Unknown dataset `{options["dataset"]}`')

    ###########################
    ##### Create Trainers #####
    ###########################

    if options['trainer'] == 'fedavg':
      t = FedAvg(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ifca':  # clustered fedavg
      t = IFCA(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ifca_fedavg':  # IFCA + FedAvg (freeze cluster)
      t = IFCA_FedAvg(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ifca_local':  # IFCA + Local (freeze cluster)
      t = IFCA_Local(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ifca_finetune':  # IFCA + Finetune (freeze cluster)
      t = IFCA_Finetune(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ifca_mrmtl':  # IFCA Base
      t = IFCA_MRMTL(options, dataset)
      result = t.train()

    elif options['trainer'] == 'finetune':
      t = Finetune(options, dataset)
      result = t.train()

    elif options['trainer'] == 'mocha':
      t = Mocha(options, dataset)
      result = t.train()

    elif options['trainer'] == 'local':  # train independent local models
      t = Local(options, dataset)
      result = t.train()

    elif options['trainer'] == 'mrmtl':
      t = MRMTL(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ditto':
      t = Ditto(options, dataset)
      result = t.train()

    else:
      raise ValueError(f'Unknown trainer `{options["trainer"]}`')

    # Run garbage collection to ensure finished runs don't keep unnecessary memory
    gc.collect()
    print(f'Outputs stored at {options["outdir"]}')
    return result


def repeat_main(options):
  num_repeats = options['repeat']
  with mp.Pool(num_repeats + 1) as pool:
    results = [pool.apply_async(main, (options, run_idx))
               for run_idx in range(num_repeats)]
    results = [r.get() for r in results]
    return results  # (num_repeats,)


def sweep_main(options):
    """Handles repeats, LR sweeps, and lambda sweeps."""
    options['no_per_round_log'] = True  # Disable per-round log since file size is too large.
    num_repeats = options['repeat']
    print(f'Sweeping over lams={options["lambdas"]}, lr={options["lrs"]}, repeat={num_repeats}')
    results = collections.defaultdict(list)

    def runner(lr, lam):
        cur_dir = f'{options["outdir"]}/lam{lam}_lr{lr}'
        cur_options = {**options, 'lambda': lam, 'learning_rate': lr, 'outdir': cur_dir}
        return [pool.apply_async(main, (cur_options, run_idx))
                for run_idx in range(num_repeats)]

    if options['downsize_pool']:
        print('Note: downsizing the multiprocessing pool along the lambda axis.')
        for lam in options['lambdas']:
            with mp.Pool(options['num_procs']) as pool:
                for lr in options['lrs']:
                    results[lr, lam] = runner(lr, lam)
                for lr in options['lrs']:
                    results[lr, lam] = [r.get() for r in results[lr, lam]]
    else:
        with mp.Pool(options['num_procs']) as pool:
            for lam in options['lambdas']:
                for lr in options['lrs']:
                    results[lr, lam] = runner(lr, lam)
            for lam in options['lambdas']:
                for lr in options['lrs']:
                    results[lr, lam] = [r.get() for r in results[lr, lam]]

    print(f'Sweep outputs stored at {options["outdir"]}')
    return results  # ((lrs, lams, repeats) of (train, test))

if __name__ == '__main__':
    options = read_options()
    print(f'outdir: {options["outdir"]}')

    # Handle sweeping separately
    if options['lambdas'] is not None or options['lrs'] is not None:
        # Populate a sweep list if doesn't exist
        options['lrs'] = options['lrs'] or [options['learning_rate']]
        options['lambdas'] = options['lambdas'] or [options['lambda']]
        # Perform sweep and take stats over repertition
        out = sweep_main(options)
        for (lr, lam), repeat_vals in out.items():
            # Axis=0 to ensure taking stats for train/test separately
            out[lr, lam] = [np.mean(repeat_vals, axis=0), np.std(repeat_vals, axis=0)]

        # Rank best results differently for regression
        rank_fn = min if options['is_regression'] else max
        # Stats over lambda sweep
        lr_out, lam_out = {}, {}
        for lr in options['lrs']:   # output result for each LR.
            res = [out[lr, lam] for lam in options['lambdas']]
            lr_out[lr] = rank_fn(res, key=lambda x: x[0][1])   # Best run by the mean of test runs.
        for lam in options['lambdas']:   # output result for each lam.
            res = [out[lr, lam] for lr in options['lrs']]
            lam_out[lam] = rank_fn(res, key=lambda x: x[0][1])
        # Stats over all sweep; best run by the mean of test runs.
        best_hparams, best_run = rank_fn(dict(out).items(), key=lambda x: x[1][0][1])
        assert (np.array(best_run) ==
                np.array(rank_fn(lam_out.values(), key=lambda x: x[0][1]))).all()
        # Save results
        with open(os.path.join(options['outdir'], 'full_result.txt'), 'w') as f:
            pprint.pprint(dict(out), stream=f)
        with open(os.path.join(options['outdir'], 'best_result.txt'), 'w') as f:
            pprint.pprint({best_hparams: best_run}, stream=f)
        with open(os.path.join(options['outdir'], 'lr_sweep_lam_result.txt'), 'w') as f:
            pprint.pprint(lr_out, stream=f)
        with open(os.path.join(options['outdir'], 'lam_sweep_lr_result.txt'), 'w') as f:
            pprint.pprint(lam_out, stream=f)

    # No sweeping
    else:
        if options['repeat'] == 1:
            out = main(options)
        else:
            out = repeat_main(options)

        out = np.atleast_2d(out)
        stats = [np.mean(out, axis=0), np.std(out, axis=0)]
        print(f'final output:\n{pprint.pformat(out)}')
        print(f'mean, std:\n{stats}')
        if options['is_regression']:
            print(f'final test metric: {stats[0][1]:.5f} ± {stats[1][1]:.5f}')
        else:
            print(f'final test metric: {stats[0][1] * 100:.3f} ± {stats[1][1] * 100:.3f}')

        with open(os.path.join(options['outdir'], 'final_result.txt'), 'w') as f:
            pprint.pprint(out, stream=f)
            print(stats, file=f)

    print(f'Final outputs stored at {options["outdir"]}')
