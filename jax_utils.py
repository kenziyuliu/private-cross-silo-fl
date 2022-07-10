import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, device_put
import optax
import haiku as hk
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.scipy.special import expit as sigmoid
import numpy as np


############################################
############## Training utils ##############
############################################


def hinge_loss_hk(model, params, batch, reg=0.1):
  inputs, targets = batch  # (n, ...), (n,)
  param_vec = model_flatten(params)
  # Scalar output's last dimension needs to be squeezed.
  preds = model.apply(params=params, inputs=inputs).squeeze()
  losses = jax.nn.relu(1.0 - targets * preds) + 0.5 * reg * (param_vec @ param_vec)
  return jnp.mean(losses)


def bce_loss_hk(model, params, batch):
  inputs, targets = batch
  # Scalar output's last dimension needs to be squeezed.
  logits = model.apply(params=params, inputs=inputs).squeeze()
  per_example_loss = optax.sigmoid_binary_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss)


def sce_loss_hk(model, params, batch):
  inputs, targets = batch
  logits = model.apply(params=params, inputs=inputs)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # Conver to one_hot
  per_example_loss = optax.softmax_cross_entropy(logits, targets)
  return jnp.mean(per_example_loss)  # `mean` implicitly means away the batch dim too.


def l2_loss_hk(model, params, batch):
  inputs, targets = batch
  # Scalar output's last dimension needs to be squeezed.
  preds = model.apply(params=params, inputs=inputs).squeeze()
  per_example_loss = 0.5 * (preds - targets)**2
  return jnp.mean(per_example_loss)


def global_l2_norm_sq(tensor_struct):
  # NOTE: Apparently you can get NaNs from `jnp.linalg.norm`; the gist is
  # that `norm` is not differentiable at 0, but `squared-norm` is indeed
  # differentiable at 0 (needed for l2 regularization).
  # https://github.com/google/jax/issues/3058
  # https://github.com/google/jax/issues/6484
  flat_vec = model_flatten(tensor_struct)
  return flat_vec @ flat_vec


def global_l2_clip(tensor_struct, clip: float):
  t_list, tree_def = tree_flatten(tensor_struct)
  global_norm = jnp.linalg.norm([jnp.linalg.norm(t.reshape(-1), ord=2) for t in t_list])
  norm_factor = jnp.minimum(clip / (global_norm + 1e-15), 1.0)
  clipped_t_list = [t * norm_factor for t in t_list]
  return tree_unflatten(tree_def, clipped_t_list)


def privatize_grad_hk(example_grads, key, clip, noise_mult):
  # Clipping
  clip_fn = vmap(global_l2_clip, in_axes=(0, None), out_axes=0)
  example_grads = clip_fn(example_grads, clip)
  # Sum
  flat_example_grads, tree_def = tree_flatten(example_grads)
  batch_size = len(flat_example_grads[0])  # 1st dim of per-example grad tensors
  flat_sum_grads = [g.sum(axis=0) for g in flat_example_grads]
  # Noise & mean
  keys = random.split(key, len(flat_sum_grads))
  flat_mean_noisy_grads = [(g + clip * noise_mult * random.normal(k, g.shape)) / batch_size
                           for k, g in zip(keys, flat_sum_grads)]
  return tree_unflatten(tree_def, flat_mean_noisy_grads)


def multiclass_classify(model, params, batch_inputs):
  logits = model.apply(params=params, inputs=batch_inputs)
  pred_class = jnp.argmax(logits, axis=1)
  return pred_class


def linear_svm_classify(model, params, batch_inputs):
  preds = model.apply(params=params, inputs=batch_inputs).squeeze()
  return jnp.sign(preds)


def logreg_classify(model, params, batch_inputs, temperature=1.0):
  # data_x: x: (n, d), w: (d,), b: (1) --> out: (n,)
  preds = model.apply(params=params, inputs=batch_inputs).squeeze()
  preds = sigmoid(preds / temperature)
  return jnp.round(preds)


def regression_pred(model, params, batch_inputs):
  return model.apply(params=params, inputs=batch_inputs).squeeze()


##########################################
############## Struct utils ##############
##########################################


def num_params(tensor_struct):
  param_list, _ = tree_flatten(tensor_struct)
  return np.sum([w.size for w in param_list])  # Use numpy since shape is static.


@jit
def model_add(params_1, params_2):
  return tree_map(jnp.add, params_1, params_2)


@jit
def model_subtract(params_1, params_2):
  return tree_map(jnp.subtract, params_1, params_2)


@jit
def model_multiply(params_1, params_2):
  return tree_map(jnp.multiply, params_1, params_2)


@jit
def model_sqrt(params):
  return tree_map(jnp.sqrt, params)


@jit
def model_divide(params_1, params_2):
  return tree_map(jnp.divide, params_1, params_2)


@jit
def model_add_scalar(params, value):
  t_list, tree_def = tree_flatten(params)
  new_t_list = [t + value for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


@jit
def model_multiply_scalar(params, factor):
  t_list, tree_def = tree_flatten(params)
  new_t_list = [t * factor for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


@jit
def model_average(params_list, weights=None):
  def average_fn(*tensor_list):
    return jnp.average(jnp.asarray(tensor_list), axis=0, weights=weights)

  return tree_map(average_fn, *params_list)


@jit
def model_flatten(params):
  tensors, tree_def = tree_flatten(params)
  flat_vec = jnp.concatenate([t.reshape(-1) for t in tensors])
  return flat_vec


@jit
def model_unflatten(flat_vec, params_template):
  t_list, tree_def = tree_flatten(params_template)
  pointer, split_list = 0, []
  for tensor in t_list:
    length = np.prod(tensor.shape)  # Shape is static so np is fine
    split_vec = flat_vec[pointer:pointer + length]
    split_list.append(split_vec.reshape(tensor.shape))
    pointer += length
  return tree_unflatten(tree_def, split_list)


@jit
def model_concat(params_list):
  flat_vecs = [model_flatten(params) for params in params_list]
  return jnp.concatenate(flat_vecs)


@jit
def model_zeros_like(params):
  return tree_map(jnp.zeros_like, params)