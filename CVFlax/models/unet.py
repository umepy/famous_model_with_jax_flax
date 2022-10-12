import jax
import jax.numpy as jnp
from flax import linen as nn


class UNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        pass
