from typing import Callable, Tuple
from absl import logging

import os
import time
import datetime
from PIL import Image
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as jkey
from chex import Scalar, Array, PRNGKey, Shape
import flax
from flax import linen as nn
from flax.training.train_state import TrainState as RawTrainState
from flax.training.checkpoints import save_checkpoint
import optax
import matplotlib.pyplot as plt
import tensorflow as tf

from architectures import *

logging.set_verbosity(logging.WARN)


LABEL_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]



class Metrices:

	def __init__(self, epochs):
		
		self.idx = 0
		self.train_loss_trace: Array		= jnp.zeros(epochs)
		self.train_accuracy_trace: Array 	= jnp.zeros(epochs)
		self.test_loss_trace: Array			= jnp.zeros(epochs)
		self.test_accuracy_trace: Array 	= jnp.zeros(epochs)
	
	def update(self, train_loss, train_accuracy, test_loss, test_accuracy):

		self.train_loss_trace = self.train_loss_trace.at[self.idx].set(train_loss)
		self.train_accuracy_trace = self.train_accuracy_trace.at[self.idx].set(train_accuracy)
		self.test_loss_trace = self.test_loss_trace.at[self.idx].set(test_loss)
		self.test_accuracy_trace = self.test_accuracy_trace.at[self.idx].set(test_accuracy)
		self.idx += 1


def get_CIFAR10(key: PRNGKey, chunk_size: float = 1.0):

    assert 0.0 < chunk_size <= 1.0

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_size = int(x_train.shape[0] * chunk_size)
    test_size = int(x_test.shape[0] * chunk_size)

    x_train = jax.random.choice(key, x_train, (train_size,), replace=False)
    y_train = jax.random.choice(key, y_train, (train_size, ), replace=False)
    x_test = jax.random.choice(key, x_test, (test_size, ), replace=False)
    y_test = jax.random.choice(key, y_test, (test_size, ), replace=False)

    return (x_train, jnp.squeeze(y_train)), (x_test, jnp.squeeze(y_test))


def predict_probs(state: TrainState, batch: Array) -> Array:
  
    return nn.softmax(state.apply_fn({'params': state.params}, batch))


@jax.jit
def apply_model(state: TrainState, batch: Array, labels: Array):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params, batch_stats):

        logits, mutated_vars = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            batch,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': jax.random.PRNGKey(42)}
        )
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

        return loss, (logits, mutated_vars)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, mutated_vars)), grads = grad_fn(state.params, state.batch_stats)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    new_state = TrainState(
        step=state.step,
        apply_fn=state.apply_fn,
        params=state.params,
        tx=state.tx,
        opt_state=state.opt_state,
        batch_stats=mutated_vars['batch_stats']
    )
    
    return grads, new_state, loss, accuracy


@jax.jit
def update_model(state: TrainState, grads: nn.FrozenDict):

    return state.apply_gradients(grads=grads)


def train_epoch(
	state: TrainState,
	x_train: Array,
	y_train: Array,
	batch_size: int,
	perm_key: PRNGKey
) -> Tuple[TrainState, Scalar, Scalar]:

	n_samples = x_train.shape[0]
	steps_per_epoch = n_samples // batch_size

	perms = jax.random.permutation(perm_key, n_samples)[:steps_per_epoch * batch_size]
	perms = jnp.reshape(perms, (steps_per_epoch, batch_size))

	epoch_loss = []
	epoch_accuracy = []

	for perm in perms:

		x_batch = x_train[perm, ...]
		y_batch = y_train[perm, ...]
		grads, state, loss, accuracy = apply_model(state, x_batch, y_batch)
		state = update_model(state, grads)
		epoch_loss.append(loss)
		epoch_accuracy.append(accuracy)
	
	return state, np.mean(epoch_loss), np.mean(epoch_accuracy)


def train_and_eval(
    seed: int,
	epochs: int,
	batch_size: int,
	create_state_fun: Callable,
	lr: Scalar = 0.001,
	momentum: Scalar = 0.9,
	ds_chunk_size = 1.0,
	log_every: int = 0,
    checkpoint_dir: str = "",
) -> Tuple[TrainState, Metrices, float]:

    # Create PRNG keys
    key = jkey(seed)
    key, ds_key, init_key = jax.random.split(key, 3)

    # Load CIFAE10 dataset
    (x_train, y_train), (x_test, y_test) = get_CIFAR10(ds_key, chunk_size=ds_chunk_size)

    # Create structures to accumulate metrices
    metrices = Metrices(epochs)

    # Get initial MinCNN training state
    state = create_state_fun(x_train, init_key, lr=lr, momentum=momentum)

    # Iterate through the dataset for epochs number of times
    start = time.time()
    for epoch in range(1, epochs + 1):

        key, epoch_key = jax.random.split(key)
        state, train_loss, train_accuracy = train_epoch(state, x_train, y_train, batch_size, epoch_key)
        test_loss, test_accuracy = eval_GAPCNN(state, x_test, y_test)
        metrices.update(train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        
        if log_every and (epoch % log_every == 0 or epoch in {1, epochs}):
            print(
                'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
                % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
            )
        
        checkpoint(checkpoint_dir, state, metrices, epoch, time.time() - start)

    return state, metrices, time.time() - start


def checkpoint(checkpoint_dir: str, state: TrainState, metrices: Metrices, epoch: int, time: float):

    try:
        save_checkpoint(checkpoint_dir, state, epoch)
        plot_metrices(metrices, checkpoint_dir)
        time_file = open(os.path.join(checkpoint_dir, f"elapsed_time.txt"), "w")
        time_file.write(f"{datetime.timedelta(seconds=time)}")
        time_file.close()
    except flax.errors.InvalidCheckpointError:
        return


def plot_metrices(metrices: Metrices, save_dir: str):

    phi = (1 + jnp.sqrt(5)) / 2
    height = 5
    n_epochs = metrices.test_accuracy_trace.shape[0]
    epochs = jnp.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(2 * phi * height, height)
    
    axes[0].plot(epochs, metrices.train_loss_trace, label="train loss")
    axes[0].plot(epochs, metrices.test_loss_trace, label="test loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss Value")
    axes[0].set_title("Loss Plot")
    axes[0].set_ybound(0, 2.5)
    axes[0].legend()

    axes[1].plot(epochs, metrices.train_accuracy_trace, label="train accuracy")
    axes[1].plot(epochs, metrices.test_accuracy_trace, label="test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Plot")
    axes[1].set_ybound(0, 100)
    axes[1].legend()

    plt.savefig(os.path.join(save_dir, f"metrices.png"))
    plt.close()


def plot_samples(batch: Array, batch_labels: Array, subplots_shape: Shape = (3, 5), seed: int = 42):

    rows = subplots_shape[0]
    cols = subplots_shape[1]
    num = rows * cols

    labels_names = LABEL_NAMES
    indeces = jax.random.choice(jax.random.PRNGKey(seed), batch.shape[0], (num,), replace=False)

    images = batch[indeces]
    labels = batch_labels[indeces]

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(rows * 2, cols * 2)
    for i in range(num):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i])
        ax.set_title(labels_names[labels[i]])
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def single_prediction(img: str | Array, model: TrainState, train_shape: Shape = (32, 32), verbose: bool = False):

    full_img = img
    
    if type(img) == str:
        full_img = Image.open(img)
        w_train, h_train = train_shape
        w_input, h_input = full_img.size
        if w_input < h_input:
            w_scaled, h_scaled = w_train, int(h_input * w_train / w_input)
        else:
            w_scaled, h_scaled = int(w_input * h_train / h_input), h_train

        img = jnp.array(full_img.resize((w_scaled, h_scaled)))
        full_img = jnp.array(full_img)

    if type(model) == TrainState:
        prediction = GAPCNN().apply(
            {'params': model.params, 'batch_stats': model.batch_stats},
            jnp.expand_dims(img, axis=0),
            training=False,
            rngs={'dropout': jax.random.PRNGKey(42)}
        )
    else:
        prediction = MinCNN().apply(
            {'params': model.params},
            jnp.expand_dims(img, axis=0)
        )
    
    prediction_probs = 100 * jnp.squeeze(jnp.exp(jax.nn.log_softmax(prediction)))

    if verbose:
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        axes[0].imshow(full_img)
        axes[0].set_axis_off()
        axes[1].bar(LABEL_NAMES, prediction_probs)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("Probability [%]")
        plt.show()
    
    return {LABEL_NAMES[i]: prediction_probs[i] for i in range(10)}


if __name__ == "__main__":

    cnn_code = "gap_cnn"

    final_state, metrices, elapsed_time = train_and_eval(
        seed=42,
        epochs=150,
        batch_size=32,
        create_state_fun=ARCHITECTURES[cnn_code],
        lr=0.005,
        momentum=0.9,
        ds_chunk_size=1.0,
        log_every=1,
        checkpoint_dir=os.path.join("checkpoints", cnn_code),
    )
    print(f"Total training time: {elapsed_time:.3f}")

