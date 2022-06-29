import jax
import haiku as hk


def mnist_model_fn(inputs, num_classes=10, **kwargs):
  return hk.Sequential([
      hk.Conv2D(32, (3, 3), padding='VALID'),
      jax.nn.relu,
      hk.MaxPool(2, strides=2, padding='VALID'),
      hk.Conv2D(64, (3, 3), padding='VALID'),
      jax.nn.relu,
      hk.MaxPool(2, strides=2, padding='VALID'),
      hk.Flatten(),
      hk.Linear(num_classes),
  ])(inputs)


def adni_model_fn(inputs, **kwargs):
  """A simple 2 layer convnet (32, 32, 1) -> (5, 5, 64) -> (512,) -> (1,)."""
  return hk.Sequential([
    # Conv layers
    hk.Conv2D(32, (5, 5), padding='VALID'),
    jax.nn.relu,
    hk.MaxPool(2, strides=2, padding='VALID'),
    hk.Conv2D(64, (5, 5), padding='VALID'),
    jax.nn.relu,
    hk.MaxPool(2, strides=2, padding='VALID'),
    hk.Flatten(),
    hk.Linear(1),
  ])(inputs)


def linear_model_fn(inputs, zero_init=True, **kwargs):
  w_init = hk.initializers.Constant(0) if zero_init else None
  return hk.Sequential([
      hk.Flatten(),
      # Linear models initialized as zero; turns out to work better.
      hk.Linear(1, w_init=w_init)
  ])(inputs)

