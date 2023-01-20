from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn


@partial(jax.jit, static_argnums=(0, 1))
def positional_encoding(length: int, depth: int) -> jax.Array:
    depth = depth / 2
    positions = jnp.arange(length).reshape((-1, 1))
    depths = jnp.arange(depth) / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = jnp.concatenate(
        [jnp.sin(angle_rads), jnp.cos(angle_rads)],
        axis=-1,
    )
    return pos_encoding


class SinusoidalEmbedding(nn.Module):
    embedding_dims: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        embedding_min_frequency = 1.0
        embedding_max_frequency = 1000.0
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(embedding_min_frequency),
                jnp.log(embedding_max_frequency),
                self.embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        return jnp.concatenate([jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)], axis=-1)


class EncoderBlock(nn.Module):
    features: int
    ff_features: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jax.Array, is_training: bool) -> jax.Array:
        mha_input = x
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.features,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=not is_training)
        x = mha_input + x
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)

        ff_input = x
        x = nn.Dense(features=self.ff_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        x = ff_input + x
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        return x


class Transformer(nn.Module):
    @nn.compact
    def __call__(self, noisy_input, noisy_variance, is_training: bool) -> jax.Array:
        emb = SinusoidalEmbedding(64)(noisy_variance)
        _b, input_length, _c = noisy_input.shape
        emb = jnp.tile(emb, (1, input_length, 1))

        x = nn.Conv(64, [1])(noisy_input)
        x = x + emb + positional_encoding(input_length, 64)

        x = EncoderBlock(64, 256, 4, 0.9)(x, is_training)
        x = EncoderBlock(64, 256, 4, 0.9)(x, is_training)
        x = EncoderBlock(64, 256, 4, 0.9)(x, is_training)
        x = EncoderBlock(64, 256, 4, 0.9)(x, is_training)

        x = nn.Conv(3, [1], kernel_init=nn.initializers.zeros)(x)
        return x


class DiffusionModel(nn.Module):
    max_signal_rates: float
    min_signal_rates: float

    input_length: int

    dataset_mean: jax.Array
    dataset_std: jax.Array

    def setup(self):
        self.network = Transformer()

    def _diffusion_schedule(self, diffusion_times: jax.Array) -> tuple[jax.Array, jax.Array]:
        # diffusion times -> angles
        start_angle = jnp.arccos(self.max_signal_rates)
        end_angle = jnp.arccos(self.min_signal_rates)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise raites
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)

        return noise_rates, signal_rates

    def _denoise(self, noisy_inputs, noise_rates, signal_rates, is_training: bool) -> tuple[jax.Array, jax.Array]:
        pred_noises = self.network(noisy_inputs, noise_rates**2, is_training)
        pred_images = (noisy_inputs - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def _reverse_diffusion(self, initial_noise, diffusion_steps: int) -> jax.Array:
        num_inputs = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_inputs = initial_noise
        for step in range(diffusion_steps):
            noisy_inputs = next_noisy_inputs

            diffusion_times = jnp.ones((num_inputs, 1, 1)) - step * step_size
            noise_rates, signal_rates = self._diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self._denoise(noisy_inputs, noise_rates, signal_rates, is_training=False)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self._diffusion_schedule(next_diffusion_times)
            next_noisy_inputs = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def _normalize(self, x: jax.Array) -> jax.Array:
        return (x - self.dataset_mean) / (self.dataset_std + 1e-5)

    def _denormalize(self, x: jax.Array) -> jax.Array:
        return self.dataset_std * x + self.dataset_mean

    def generate(self, num_inputs: int, diffusion_steps: int, rng: jax.random.KeyArray) -> jax.Array:
        """ノイズから列を生成する"""
        initial_noise = jax.random.normal(rng, shape=(num_inputs, self.input_length, 3))
        generate_inputs = self._reverse_diffusion(initial_noise, diffusion_steps)
        return self._denormalize(generate_inputs)

    def __call__(
        self, inputs: jax.Array, rng: jax.random.KeyArray
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """訓練用にノイズ付与された列をデノイズする

        Args:
            inputs: 訓練用の入力列 shape(B, L, 3)
            rng: 乱数

        Returns:
            pred_noises, pred_images
        """
        normed_inputs = self._normalize(inputs)
        rng_noise, rng = jax.random.split(rng)
        noises = jax.random.normal(key=rng_noise, shape=inputs.shape)
        rng_time, rng = jax.random.split(rng)
        diffusion_times = jax.random.uniform(key=rng_time, shape=(inputs.shape[0], 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self._diffusion_schedule(diffusion_times)
        noisy_inputs = signal_rates * normed_inputs + noise_rates * noises
        pred_noises, pred_inputs = self._denoise(noisy_inputs, noise_rates, signal_rates, is_training=True)
        pred_inputs = self._denormalize(pred_inputs)
        return pred_noises, pred_inputs, noises, noisy_inputs
