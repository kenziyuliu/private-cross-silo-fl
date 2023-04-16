# On Privacy and Personalization in Cross-Silo Federated Learning

This repo hosts the code for the paper "On Privacy and Personalization in Cross-Silo Federated Learning". The implemention is written in Python with [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku). A short version of this work appears at [TPDP 2022](https://tpdp.journalprivacyconfidentiality.org/2022).

[[PDF](https://arxiv.org/pdf/2206.07902)][[arXiv](https://arxiv.org/abs/2206.07902)]

## Directory structure

* `main.py`: the main driver script
* `utils.py`: helper functions for reading datasets, data loading, logging, etc.
* `jax_utils.py`: helper functions for JAX (loss, prediction, structs, etc)
* `model_utils.py`: model definitions in JAX/Haiku
* `dp_accounting.py` helper functions for privacy accounting
* `data_utils/`: scripts for preprocessing datasets (mainly rotated + masked MNIST and ADNI)
* `trainers/`: implementations of the benchmark algorithms
* `runners/`: scripts for starting an experiment
* `data/`: directory of datasets

## Dependencies

Python 3.9 with dependencies in `requirements.txt` (an older version of Python would probably work too).

## Running an experiment

The general template commands for running an experiment are:

* Without DP

```bash
bash runners/<dataset>/run_<algorithm>.sh --repeat 5 [other flags]
```

* With DP

```bash
bash runners/<dataset>/dp/run_<algorithm>.sh --repeat 5 -eps <epsilon value> [other flags]
```

Below are examples commands for running experiments with some hyperparameters.
See `main.py` for the full list of hyperparameters.

### Example: single run

For example, to reproduce the results on School of Fig. 3(c) at $\varepsilon = 6$, run

```bash
bash runners/school/dp/run_fedavg.sh -r 5 -t 200 -eps 6 -lr 0.01  # gives ~ 0.02564
bash runners/school/dp/run_local.sh -r 5 -t 200 -eps 6 -lr 0.01   # gives ~ 0.02628
bash runners/school/dp/run_mrmtl.sh -r 5 -t 200 -eps 6 --lambda 1 -lr 0.01  # gives ~ 0.02394
```

### Example: hyperparameter sweep

For convenience, hyperparameter grid search is implemented for `--lambda` / `--lambdas` (personalization hyperparam for MR-MTL, Ditto, etc.) and `--learning_rate` / `--lrs` (client learning rates).

For example, we can sweep the client LRs and the lambdas of MR-MTL for a range of epsilons via

```bash
for eps in 0.1 0.3 0.5 0.7 1; do
  bash runners/vehicle/dp/run_mrmtl.sh \
    -r 5 --lrs "0.001 0.003 0.01" --lambdas "0.1 0.3 1 3" \
    -t 400 --quiet -eps $eps \
    -o logs/vehicle/mrmtl_eps$eps
done
```

The results will then be stored in `logs/vehicle/mrmtl_eps<eps>`.

## Dataset access

See appendix of the paper for full details on dataset access.

* School, Vehicle, and GLEAM datasets can be downloaded [here](https://www.dropbox.com/s/gxgnu3imufuoddj/private_silos_data.zip?dl=0). A copy of the School dataset is included for convenience.
* Rotated/Patched MNIST can be downloaded/preprocessed automatically from the `data_utils/rotated_mnist.py` script.
* ADNI dataset requires approval; see appendix for more details.
* Put datasets in the following structure: `data/<dataset_name>/<content>`; e.g. `data/vehicle/vehicle.mat`.

## Mean Estimation and Privacy Cost of Hyperparameter Tuning

To generate Fig. 7 of the paper, run

```bash
python3 plot_hparam_cost.py -eps 0.25 -tau 0.03 -o lowhet  # Low heterogeneity
python3 plot_hparam_cost.py -eps 0.25 -tau 0.08 -o midhet  # Moderate heterogeneity
python3 plot_hparam_cost.py -eps 0.25 -tau 0.2 -o highhet  # High heterogeneity
```

## Citation

Please consider citing our work if you find this repo useful:

```BibTeX
@article{liu2022privacy,
  title={On privacy and personalization in cross-silo federated learning},
  author={Liu, Ken and Hu, Shengyuan and Wu, Steven Z and Smith, Virginia},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5925--5940},
  year={2022}
}
```
