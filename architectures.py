import jax.numpy as jnp
import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState as RawTrainState
from flax.training.checkpoints import restore_checkpoint
from flax.core import FrozenDict
from chex import Array, Scalar, PRNGKey


IMG_RANDOM = jax.random.randint(jax.random.PRNGKey(42), (32, 32, 3), 0, 255, jnp.uint8)
IMG_BLACK = jnp.zeros((32, 32, 3), jnp.uint8)
IMG_WHITE = 255 * jnp.ones((32, 32, 3), jnp.uint8)


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


def conv_block_with_dropout(x: Array, features: int, dropout_rate: float, training: bool) -> Array:

    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = nn.relu(x)
    x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)

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


class DropoutCNN(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array, training: bool):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = conv_block_with_dropout(x, 20, 0.1, training)
        x = conv_block_with_dropout(x, 40, 0.2, training)
        x = conv_block_with_dropout(x, 80, 0.3, training)
        x = conv_block_with_dropout(x, 160, 0.4, training)
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dense(features=10)(x)
        
        return x


def gap_pool(x):

    return jnp.max(x, axis=(-3, -2))


class GAPCNN(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array, training: bool):
        
        x = batch / 255
        x = conv_block_with_dropout(x, 20, 0.1, training)
        x = conv_block_with_dropout(x, 40, 0.2, training)
        x = conv_block_with_dropout(x, 80, 0.3, training)
        x = conv_block_with_dropout(x, 160, 0.4, training)
        x = gap_pool(x)
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

    return RawTrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_MinCNNPlus(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = MinCNNPlus()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return RawTrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_DoubleBlockCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = DoubleBlockCNN()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return RawTrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


def create_QuadBlockCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = QuadBlockCNN()
    params = cnn.init(init_key, dummy_batch)['params']
    sgd = optax.sgd(learning_rate=lr, momentum=momentum)

    return RawTrainState.create(apply_fn=cnn.apply, params=params, tx=sgd)


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


def create_DropoutCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = DropoutCNN()
    variables = cnn.init(init_key, dummy_batch, training=False)

    return TrainState.create(
        apply_fn=cnn.apply,
        params=variables['params'],
        tx=optax.sgd(learning_rate=lr, momentum=momentum),
        batch_stats=variables['batch_stats'])


@jax.jit
def eval_DropoutCNN(state: TrainState, batch: Array, labels: Array):
    
    logits = DropoutCNN().apply(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch,
        training=False,
        rngs={'dropout': jax.random.PRNGKey(42)}
    )
    one_hot = jax.nn.one_hot(labels, 10)

    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

    return loss, accuracy


def create_GAPCNN(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> TrainState:

    cnn = GAPCNN()
    variables = cnn.init(init_key, dummy_batch, training=False)

    return TrainState.create(
        apply_fn=cnn.apply,
        params=variables['params'],
        tx=optax.sgd(learning_rate=lr, momentum=momentum),
        batch_stats=variables['batch_stats'])


@jax.jit
def eval_GAPCNN(state: TrainState, batch: Array, labels: Array):
    
    logits = GAPCNN().apply(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch,
        training=False,
        rngs={'dropout': jax.random.PRNGKey(42)}
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
    "batch_norm_cnn": create_BatchNormCNN,
    "dropout_cnn": create_DropoutCNN,
    "gap_cnn": create_GAPCNN
}


DUMMY_MODEL = create_GAPCNN(jnp.zeros((1, 32, 32, 3)), jax.random.PRNGKey(42))

MIN_CNN_MODEL = restore_checkpoint(
    "../results/min_cnn", 
    create_MinCNN(jnp.zeros((1, 32, 32, 3)), jax.random.PRNGKey(42))
)

GAP_MODEL = restore_checkpoint(
    "../results/gap_cnn", 
    create_GAPCNN(jnp.zeros((1, 32, 32, 3)), jax.random.PRNGKey(42))
)
