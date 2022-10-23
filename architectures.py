from typing import Callable
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from chex import Array, Scalar, PRNGKey


def conv_block(x: Array, features: int) -> Array:

    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    return x


def conv_block_with_bn(x: Array, features: int) -> Array:

    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=False)(x)
    x = nn.relu(x)
    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=False)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    return x


class MinCNN(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = nn.Conv(features=5, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.Conv(features=5, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dense(features=10)(x)
        
        return x


class MinCNNPlus(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = nn.Conv(features=20, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.Conv(features=20, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dense(features=10)(x)
        
        return x


class DoubleBlockCNN(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = conv_block(x, 20)
        x = conv_block(x, 40)
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dense(features=10)(x)
        
        return x


class QuadBlockCNN(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = conv_block(x, 20)
        x = conv_block(x, 40)
        x = conv_block(x, 80)
        x = conv_block(x, 160)
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dense(features=10)(x)
        
        return x


class BatchNormCNN(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = conv_block_with_bn(x, 20)
        x = conv_block_with_bn(x, 40)
        x = conv_block_with_bn(x, 80)
        x = conv_block_with_bn(x, 160)
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dense(features=10)(x)
        
        return x


def create_MinCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = MinCNN()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_MinCNNPlus(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = MinCNNPlus()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_DoubleBlockCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = DoubleBlockCNN()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_QuadBlockCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = QuadBlockCNN()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_BatchNormCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = BatchNormCNN()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


ARCHITECTURES = {
    "min_cnn": create_MinCNN,
    "min_cnn_plus": create_MinCNNPlus,
    "double_block_cnn": create_DoubleBlockCNN,
    "quad_block_cnn": create_QuadBlockCNN,
    "batch_norm_cnn": create_BatchNormCNN
}
