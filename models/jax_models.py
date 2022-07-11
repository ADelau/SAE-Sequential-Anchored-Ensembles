# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File reproduced from https://github.com/izmailovpavel/neurips_bdl_starter_kit"""

"""CNN haiku models."""

from typing import Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import functools


Batch = Tuple[jnp.ndarray, jnp.ndarray]
_DEFAULT_BN_CONFIG = {
  'decay_rate': 0.9,
  'eps': 1e-5,
  'create_scale': True,
  'create_offset': True
}


def make_cifar_alexnet(data_info):
  num_classes = data_info["num_classes"]
  # act = jax.nn.relu
  act = jax.nn.swish

  def forward(x, is_training=True):
    x = x.astype(jnp.float32)
    net = hk.Sequential([  # [32, 32, 3]
        hk.Conv2D(output_channels=64, kernel_shape=3,
                  padding='SAME'),  # [32, 32, 64]
        act,
        hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1),
                   padding='VALID'),  # [16, 16, 64]
        hk.Conv2D(output_channels=128, kernel_shape=3,
                  padding='SAME'),  # [16, 16, 128]
        act,
        hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1),
                   padding='VALID'),  # [8, 8, 128]
        hk.Conv2D(output_channels=256, kernel_shape=2,
                  padding='SAME'),  # [8, 8, 256]
        act,
        hk.Conv2D(output_channels=128, kernel_shape=2,
                  padding='SAME'),  # [8, 8, 128]
        act,
        hk.Conv2D(output_channels=64, kernel_shape=2,
                  padding='SAME'),  # [8, 8, 64]
        act,
        hk.Flatten(),
        hk.Linear(256),
        act,
        hk.Linear(256),
        act,
        hk.Linear(num_classes),
    ])
    return net(x)

  return forward


def make_retinopathy_cnn(data_info):
  num_classes = data_info["num_classes"]
  act = jax.nn.relu

  def forward(x, is_training=True):
    """Simple CNN for diabetic retinopathy classification."""
    x = x.astype(jnp.float32)
    cnn = hk.Sequential([
        hk.Conv2D(output_channels=32, kernel_shape=3, padding='SAME'),
        act,
        hk.Conv2D(output_channels=32, kernel_shape=3, padding='SAME'),
        act,
        hk.MaxPool(window_shape=2, strides=2, padding='SAME'),
        hk.Conv2D(output_channels=32, kernel_shape=3, padding='SAME'),
        act,
        hk.Conv2D(output_channels=32, kernel_shape=3, padding='SAME'),
        act,
        hk.MaxPool(window_shape=2, strides=2, padding='SAME'),
        hk.Conv2D(output_channels=16, kernel_shape=3, padding='SAME'),
        act,
        hk.MaxPool(window_shape=2, strides=2, padding='SAME'),
        hk.Conv2D(output_channels=16, kernel_shape=3, padding='SAME'),
        act,
        hk.MaxPool(window_shape=2, strides=2, padding='SAME'),
        hk.Flatten(),
        hk.Linear(128),
        act,
        hk.Linear(num_classes),
    ])
    return cnn(x)

  return forward


def make_lenet5_fn(data_info):
    num_classes = data_info["num_classes"]
    act = jax.nn.relu

    def lenet_fn(x, is_training):
        """Network inspired by LeNet-5."""

        cnn = hk.Sequential([
            hk.Conv2D(output_channels=6, kernel_shape=5, padding="SAME"),
            act,
            hk.AvgPool(window_shape=2, strides=2, padding="VALID"),
            hk.Conv2D(output_channels=16, kernel_shape=5, padding="VALID"),
            act,
            hk.AvgPool(window_shape=2, strides=2, padding="VALID"),
            hk.Flatten(),
            hk.Linear(120),
            act,
            hk.Linear(84),
            act,
            hk.Linear(num_classes),
        ])
        return cnn(x)

    return lenet_fn


he_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')


class FilterResponseNorm(hk.Module):
  def __init__(self, eps=1e-6, name='frn'):
    super().__init__(name=name)
    self.eps = eps
  
  def __call__(self, x, **unused_kwargs):
    del unused_kwargs
    par_shape = (1, 1, 1, x.shape[-1])  # [1,1,1,C]
    tau = hk.get_parameter('tau', par_shape, x.dtype, init=jnp.zeros)
    beta = hk.get_parameter('beta', par_shape, x.dtype, init=jnp.zeros)
    gamma = hk.get_parameter('gamma', par_shape, x.dtype, init=jnp.ones)
    nu2 = jnp.mean(jnp.square(x), axis=[1, 2], keepdims=True)
    x = x * jax.lax.rsqrt(nu2 + self.eps)
    y = gamma * x + beta
    z = jnp.maximum(y, tau)
    return z
  

def _resnet_layer(
    inputs, num_filters, normalization_layer, kernel_size=3, strides=1,
    activation=lambda x: x, use_bias=True, is_training=True
):
  x = inputs
  x = hk.Conv2D(
      num_filters, kernel_size, stride=strides, padding='same',
      w_init=he_normal, with_bias=use_bias)(x)
  x = normalization_layer()(x, is_training=is_training)
  x = activation(x)
  return x


def make_resnet_fn(
    num_classes: int,
    depth: int,
    normalization_layer,
    width: int = 16,
    use_bias: bool = True,
    activation=jax.nn.relu,
):
  num_res_blocks = (depth - 2) // 6
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')
  
  def forward(x, is_training):
    num_filters = width
    x = _resnet_layer(
        x, num_filters=num_filters, activation=activation, use_bias=use_bias,
        normalization_layer=normalization_layer
    )
    for stack in range(3):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
          strides = 2  # downsample
        y = _resnet_layer(
            x, num_filters=num_filters, strides=strides, activation=activation,
            use_bias=use_bias, is_training=is_training,
            normalization_layer=normalization_layer)
        y = _resnet_layer(
            y, num_filters=num_filters, use_bias=use_bias,
            is_training=is_training, normalization_layer=normalization_layer)
        if stack > 0 and res_block == 0:  # first layer but not first stack
          # linear projection residual shortcut connection to match changed dims
          x = _resnet_layer(
              x, num_filters=num_filters, kernel_size=1, strides=strides,
              use_bias=use_bias, is_training=is_training,
              normalization_layer=normalization_layer)
        x = activation(x + y)
      num_filters *= 2
    x = hk.AvgPool((8, 8, 1), 8, 'VALID')(x)
    x = hk.Flatten()(x)
    logits = hk.Linear(num_classes, w_init=he_normal)(x)
    return logits
  return forward


def make_resnet20_fn(data_info, activation=jax.nn.relu):
  num_classes = data_info["num_classes"]
  def normalization_layer(): hk.BatchNorm(**_DEFAULT_BN_CONFIG)
  return make_resnet_fn(
      num_classes, depth=20, normalization_layer=normalization_layer,
      activation=activation)


def make_resnet20_frn_fn(data_info, activation=jax.nn.relu):
  num_classes = data_info["num_classes"]
  return make_resnet_fn(
      num_classes, depth=20, normalization_layer=FilterResponseNorm,
      activation=activation)


def make_cnn_lstm(
    data_info, max_features=20000, embedding_size=128, cell_size=128,
    num_filters=64, kernel_size=5, pool_size=4,
    use_swish=False, use_maxpool=True
):
  """CNN LSTM architecture for the IMDB dataset."""

  num_classes = data_info["num_classes"]
  def forward(x, is_training):
    batch_size = x.shape[0]
    x = hk.Embed(vocab_size=max_features, embed_dim=embedding_size)(x)
    x = hk.Conv1D(output_channels=num_filters, kernel_shape=kernel_size,
                  padding="VALID")(x)
    if use_swish:
        x = jax.nn.swish(x)
    else:
        x = jax.nn.relu(x)
    if use_maxpool:
        x = hk.MaxPool(
            window_shape=pool_size, strides=pool_size, padding='VALID',
            channel_axis=2)(x)
    x = jnp.moveaxis(x, 1, 0)[:, :] #[T, B, F]
    lstm_layer = hk.LSTM(hidden_size=cell_size)
    init_state = lstm_layer.initial_state(batch_size)
    x, state = hk.static_unroll(lstm_layer, x, init_state)
    x = x[-1]
    logits = hk.Linear(num_classes)(x)
    return logits
  
  return forward


def make_smooth_cnn_lstm(
    data_info, max_features=20000, embedding_size=128, cell_size=128,
    num_filters=64, kernel_size=5, pool_size=4
):
  num_classes = data_info["num_classes"]
  return make_cnn_lstm(num_classes, max_features, embedding_size, cell_size,
    num_filters, kernel_size, pool_size, use_swish=True, use_maxpool=False)


def make_mlp(layer_dims, output_dim):
  def forward(x, is_training):
    x = hk.Flatten()(x)
    for layer_dim in layer_dims:
      x = hk.Linear(layer_dim)(x)
      x = jax.nn.relu(x)
    x = hk.Linear(output_dim)(x)
    return x
  return forward


def make_mlp_regression(data_info, output_dim=2, layer_dims=[100, 100]):
  return make_mlp(layer_dims, output_dim)


def make_mlp_regression_small(data_info):
  return make_mlp([50], 2)


def make_mlp_classification(data_info, layer_dims=[256, 256]):
  num_classes = data_info["num_classes"]
  return make_mlp(layer_dims, num_classes)

def make_mlp_classification_toy(data_info, layer_dims=[3, 3]):
  num_classes = data_info["num_classes"]
  return make_mlp(layer_dims, num_classes)


def make_logistic_regression(data_info):
  num_classes = data_info["num_classes"]
  return make_mlp([], num_classes)

  
def get_model(model_name, data_info, **kwargs):
  _MODEL_FNS = {
    "retinopathy_cnn": make_retinopathy_cnn,
    "cifar_alexnet": make_cifar_alexnet,
    "medmnist_lenet": make_lenet5_fn,
    "uci_mlp": make_mlp_regression_small,
    "resnet20": make_resnet20_fn,
    "resnet20_frn": make_resnet20_frn_fn,
    "resnet20_frn_swish": functools.partial(
      make_resnet20_frn_fn, activation=jax.nn.swish),
    "cnn_lstm": make_cnn_lstm,
    "smooth_cnn_lstm": make_smooth_cnn_lstm,
    "mlp_regression": make_mlp_regression,
    "mlp_classification": make_mlp_classification,
    "mlp_classification_toy": make_mlp_classification_toy,
    "logistic_regression": make_logistic_regression,
  }
  net_fn = _MODEL_FNS[model_name](data_info, **kwargs)
  net = hk.transform(net_fn)
  return net
