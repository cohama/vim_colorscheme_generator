from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint
from tqdm import tqdm

from vim_colorscheme_generator.data import BatchGen
from vim_colorscheme_generator.model import DiffusionModel


@partial(jax.jit, static_argnums=(4,))
def train_step(
    inputs: jax.Array,
    rng: jax.random.KeyArray,
    dropout_rng: jax.random.KeyArray,
    state: TrainState,
    is_training=True,
):
    def loss_fn(params):
        pred_noises, pred_inputs, noises, noisy_inputs = state.apply_fn(
            {
                "params": params,
            },
            inputs,
            rng=rng,
            rngs={"dropout": dropout_rng},
        )
        noise_loss = jnp.abs(noises - pred_noises).mean()
        input_loss = jnp.abs(inputs - pred_inputs).mean()
        return noise_loss, (input_loss, pred_inputs, noisy_inputs)

    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (noise_loss, (image_loss, pred_inputs, noisy_inputs)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
    else:
        noise_loss, (image_loss, pred_inputs, noisy_inputs) = loss_fn(state.params)
    return noise_loss, image_loss, pred_inputs, noisy_inputs, state


def _update_ema(p_cur, p_new, momentum: float = 0.999):
    return momentum * p_cur + (1 - momentum) * p_new


def run(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    log_dir: Path,
    model_checkpoint_dir: Path,
):
    batch_gen = BatchGen(Path("scripts/result.jsonl"), batch_size=32)
    model = DiffusionModel(
        max_signal_rates=0.95,
        min_signal_rates=0.02,
        input_length=40,
        dataset_mean=batch_gen.dataset_mean,
        dataset_std=batch_gen.dataset_std,
    )
    key_param, rng = jax.random.split(jax.random.PRNGKey(1))
    key_dropout, rng = jax.random.split(jax.random.PRNGKey(1))
    key2, rng = jax.random.split(rng)
    dummy_input = jnp.ones((batch_size, batch_gen.input_length, 3))
    variables = model.init({"params": key_param, "dropout": key_dropout}, dummy_input, key2)

    params = variables["params"]

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.lamb(learning_rate=learning_rate, weight_decay=weight_decay),
    )
    ema_params = state.params.copy(add_or_replace={})

    summary_writer = tf.summary.create_file_writer(str(log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")))
    for epoch in tqdm(range(epochs)):
        train_noise_loss, train_inputs_loss = 0.0, 0.0

        for inputs in batch_gen.ds.as_numpy_iterator():
            key_train, rng = jax.random.split(rng)
            key_dropout, rng = jax.random.split(rng)
            noise_loss, inputs_loss, pred_inputs, noisy_inputs, state = train_step(
                inputs, key_train, key_dropout, state, is_training=True
            )
            train_noise_loss += noise_loss
            train_inputs_loss += inputs_loss
            ema_params = jax.tree_map(_update_ema, ema_params, state.params)
        if epoch == 0:
            print("train_finished")

        train_noise_loss /= float(batch_gen.datasize)
        train_inputs_loss /= float(batch_gen.datasize)

        eval_key, rng = jax.random.split(rng)
        diff_key, rng = jax.random.split(rng)
        generated_inputs = model.apply(
            variables={
                "params": ema_params,
            },
            method=model.generate,
            num_inputs=10,
            diffusion_steps=20,
            rng=diff_key,
        )

        with summary_writer.as_default(step=epoch):
            tf.summary.scalar("train_noise_loss", train_noise_loss)
            tf.summary.scalar("train_input_loss", train_inputs_loss)
            tf.summary.image("generated_iamge", batch_gen.visualize(generated_inputs), max_outputs=10)
            tf.summary.image("train_inputs", batch_gen.visualize(inputs), max_outputs=10)
            tf.summary.image("train_noisy_inputs", batch_gen.visualize(noisy_inputs), max_outputs=10)
            tf.summary.image("train_denoised_inputs", batch_gen.visualize(pred_inputs), max_outputs=10)

        save_checkpoint(model_checkpoint_dir, target=state, step=epoch)


if __name__ == "__main__":
    run(
        batch_size=64,
        epochs=100,
        learning_rate=optax.piecewise_constant_schedule(5e-3, {80: 0.1}),
        weight_decay=1e-4,
        log_dir=Path("log"),
        model_checkpoint_dir=Path("model_checkpoint")
    )
