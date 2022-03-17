import jax
import jax.numpy as jnp
from flax import linen as nn


class AlexNet(nn.Module):
    """This implementation is not "pure" AlexNet. See the details in "ImageNet Classification with Deep Convolutional Neural Networks" """

    @nn.compact
    def __call__(self, x, output_dim=1000):
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), padding=((0, 0), (0, 0)))(x)  # 55x55
        x = nn.relu(x)

        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # 27x27
        x = nn.Conv(features=256, kernel_size=(5, 5), padding=((2, 2), (2, 2)))(x)  # 27x27
        x = nn.relu(x)

        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # 13x13
        x = nn.Conv(features=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(x)  # 13x13
        x = nn.relu(x)

        x = nn.Conv(features=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(x)  # 13x13
        x = nn.relu(x)

        x = nn.Conv(features=256, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(x)  # 13x13
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # 6x6

        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)

        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)

        x = nn.Dense(features=output_dim)(x)
        x = nn.softmax(x)
        return x


if __name__ == "__main__":
    model = AlexNet()
    key = jax.random.PRNGKey(20220317)
    params = model.init(key, jnp.ones((1, 227, 227, 3)))["params"]
    pred = model.apply({"params": params}, jax.random.normal(key, (10, 227, 227, 3)))
    print(jnp.argmax(pred, axis=1))
