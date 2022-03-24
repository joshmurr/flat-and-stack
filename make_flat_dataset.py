import argparse

from keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
from PIL import Image

datasets = {
    "mnist": mnist.load_data,  # noqa: F821
    "fashion": fashion_mnist.load_data,  # noqa: F821
    "cifar10": cifar10.load_data,  # noqa: F821
}


def get_dataset(name):
    (x_train, y_train), (x_test, y_test) = datasets[name]()
    return x_train, y_train, x_test, y_test


def flat_and_stack(arr):
    width = arr.shape[1] * arr.shape[2]
    height = arr.shape[0]
    channels = arr.shape[-1] if len(arr.shape) == 4 else 1
    stacked = np.ones([height, width, channels])

    for i in range(height):
        stacked[i] = arr[i].reshape([1, width, channels])

    return stacked.squeeze()


def save_labels(arr, name):
    with open(f"{name}_{arr.shape[0]}_labels.npy", "wb") as file:
        np.save(file, arr)


def save_as_png(arr, name):
    im = Image.fromarray(arr)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(f"{name}_{arr.shape[0]}.png")


def main(args):
    x_train, y_train, x_test, y_test = get_dataset(args.dataset)

    x_train_stack = flat_and_stack(x_train)
    x_test_stack = flat_and_stack(x_test)

    all_images = np.vstack((x_train_stack, x_test_stack))

    # Drop single dimension if of shape (k, 1)
    if len(y_train.shape) == 2:
        y_train = y_train.reshape((y_train.shape[0],))
    if len(y_test.shape) == 2:
        y_test = y_test.reshape((y_test.shape[0],))

    all_labels = np.hstack((y_train, y_test))

    total_images = all_images.shape[0]

    k = min(total_images, args.numImages)

    save_as_png(all_images[:k].astype("uint8"), args.outputName)
    save_labels(all_labels[:k], args.outputName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten a Dataset")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["mnist", "fashion", "cifar10"],
        default="mnist",
        help="Choose a dataset",
    )
    parser.add_argument(
        "-k",
        "--numImages",
        type=int,
        default="-1",
        help="Number of images...",
    )
    parser.add_argument(
        "-n",
        "--outputName",
        type=str,
        default="output",
        help="Filename prefix",
    )

    args = parser.parse_args()

    main(args)
