import jax.numpy as jnp
import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState as RawTrainState
from flax.core import FrozenDict
from chex import Array, Scalar, PRNGKey


class TrainState(RawTrainState):
    batch_stats: FrozenDict


def conv_block(x: Array, features: int) -> Array:

    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    return x


def conv_block_with_bn(x: Array, features: int, training: bool) -> Array:

    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = nn.relu(x)
    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=not training)(x)
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
    def __call__(self, batch: Array, training: bool):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = conv_block_with_bn(x, 20, training)
        x = conv_block_with_bn(x, 40, training)
        x = conv_block_with_bn(x, 80, training)
        x = conv_block_with_bn(x, 160, training)
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
    variables = cnn.init(init_key, dummy_batch, training=False)

    return TrainState.create(
        apply_fn=cnn.apply,
        params=variables['params'],
        tx=optax.sgd(learning_rate=lr, momentum=momentum),
        batch_stats=variables['batch_stats'])


@jax.jit
def eval_BatchNormCNN(state: TrainState, batch: Array, labels: Array):
    
    logits = BatchNormCNN().apply(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch,
        training=False
    )
    one_hot = jax.nn.one_hot(labels, 10)

    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

    return loss, accuracy


ARCHITECTURES = {
    "min_cnn": create_MinCNN,
    "min_cnn_plus": create_MinCNNPlus,
    "double_block_cnn": create_DoubleBlockCNN,
    "quad_block_cnn": create_QuadBlockCNN,
    "batch_norm_cnn": create_BatchNormCNN
}
