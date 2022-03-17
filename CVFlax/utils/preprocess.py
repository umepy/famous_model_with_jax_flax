import json
import os
import shutil
from glob import glob

import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def jnp_transform(x):
    return jnp.array(x) / 255.0


def collate_fn(batch):
    batch = list(zip(*batch))
    x = jnp.stack(batch[0])
    y = jnp.array(batch[1])
    return x, y


def download_food101():
    datasets.Food101("./data/", download=True)

    if not os.path.exists("./data/food-101/train"):
        print("Splitting dataset to train and test...")
        images = glob("./data/food-101/images/*/*")
        train_files = json.load(open("./data/food-101/meta/train.json", "r"))
        test_files = json.load(open("./data/food-101/meta/test.json", "r"))
        images = [x.replace("./data/food-101/images/", "").replace(".jpg", "") for x in images]

        for i in tqdm(images):
            category = i.split("/")[0]
            if i in train_files[category]:
                os.makedirs(f"./data/food-101/train/{category}", exist_ok=True)
                shutil.copy(f"./data/food-101/images/{i}.jpg", f"./data/food-101/train/{i}.jpg")
            elif i in test_files[category]:
                os.makedirs(f"./data/food-101/test/{category}", exist_ok=True)
                shutil.copy(f"./data/food-101/images/{i}.jpg", f"./data/food-101/test/{i}.jpg")
            else:
                raise Exception()
    else:
        print("seems already splitted, so skipping to split dataset")


def calculate_mean_std_food101():
    alexnet_transform = transforms.Compose([jnp_transform, lambda x: jnp.transpose(x, (2, 0, 1))])
    trainset = datasets.ImageFolder("./data/food-101/train/", transform=alexnet_transform)
    train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    mean = jnp.zeros(3)
    std = jnp.zeros(3)
    bar = tqdm(total=len(trainset))
    for idx, (x, y) in enumerate(train_loader):
        bar.update(1)
        x = x.reshape(3, -1)
        mean += jnp.mean(x, axis=1)
        std += jnp.std(x, axis=1)
    mean /= len(trainset)
    std /= len(trainset)
    print(f"mean:{mean}, std:{std}")


def alexnet_dataloader(batch_size=128):
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(p=0.5),
            jnp_transform,
            transforms.Normalize((0.54498774, 0.4434933, 0.34360075), (0.23354167, 0.24430245, 0.24236338)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.TenCrop(227),
            jnp_transform,
            transforms.Normalize((0.54498774, 0.4434933, 0.34360075), (0.23354167, 0.24430245, 0.24236338)),
        ]
    )
    trainset = datasets.ImageFolder("./data/food-101/train/", transform=train_transform)
    testset = datasets.ImageFolder("./data/food-101/test/", transform=test_transform)
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    return train_loader, test_loader


if __name__ == "__main__":
    calculate_mean_std_food101()