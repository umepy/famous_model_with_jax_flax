import jax
import jax.numpy as jnp
from flax import linen as nn


def conv3x3_relu(x, features):
    x = nn.Conv(features=features, kernel_size=(3, 3), strides=(1, 1), padding=((0, 0), (0, 0)))(x)
    x = nn.relu(x)
    return x


def center_crop(x, crop_size):
    # get height
    h = x.size()[1]
    start_idx = int((h - crop_size) / 2)
    end_idx = start_idx + crop_size
    return x[:, start_idx:end_idx, start_idx:end_idx, :]


class UNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # initial dimention 572x572, x are NHWC
        x = conv3x3_relu(x, 64)  # 570x570x64
        x = conv3x3_relu(x, 64)  # 568x568x64
        # save values for latter input
        x_392 = center_crop(x, crop_size=392)

        # max pooling with stride 2
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 284x284x64
        x = conv3x3_relu(x, 128)  # 282x282x64
        x = conv3x3_relu(x, 128)  # 280x280x64
        # save values for latter input
        x_200 = center_crop(x, crop_size=200)

        # max pooling with stride 2
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 140x140x128
        x = conv3x3_relu(x, 256)  # 138x138x256
        x = conv3x3_relu(x, 256)  # 136x136x256
        # save values for latter input
        x_104 = center_crop(x, crop_size=104)

        # max pooling with stride 2
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 68x68x256
        x = conv3x3_relu(x, 512)  # 66x66x512
        x = conv3x3_relu(x, 512)  # 64x64x512
        # save values for latter input
        x_56 = center_crop(x, crop_size=56)

        # max pooling with stride 2
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 32x32x512
        x = conv3x3_relu(x, 1024)  # 30x30x1024
        x = conv3x3_relu(x, 1024)  # 28x28x1024

        return x


if __name__ == "__main__":
    model = UNet()
    key = jax.random.PRNGKey(20221017)
    params = model.init(key, jnp.ones((1, 572, 572, 3)))["params"]
    pred = model.apply({"params": params}, jax.random.normal(key, (10, 572, 572, 3)))
    print(pred)
