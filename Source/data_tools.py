"""Provides tools to work with the data."""

import torch
from torchvision import transforms, datasets
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_class_noniid, general_class_noniid
from torch.utils.data import Dataset


class DataWrapper:
    def __init__(self, Xy_list) -> None:
        self.data = []
        for i in range(len(Xy_list[0])):
            self.data.append(tuple([Xy_list[0][i], Xy_list[1][i]]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]




class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone(), torch.tensor(label)

def get_dataset(name="mnist", samp_type="iid", unequal_splits=False, num_devices=5):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if name == "mnist" or "fmnist":
        if name == "mnist":
            data_dir = "../data/mnist/"
        else:
            data_dir = "../data/fmnist/"

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst users
        if samp_type=="iid":
            # Sample IID user data from Mnist
            device_groups = mnist_iid(train_dataset, num_devices)
        elif samp_type=="class-noniid":
            # Sample Class Non-IID user data from Mnist
            # device_groups = mnist_class_noniid(train_dataset, num_devices)
            device_groups = general_class_noniid(train_dataset, num_devices)
        else:
            # Sample Non-IID user data from Mnist
            if unequal_splits:
                # Chose uneuqal splits for every user
                device_groups = mnist_noniid_unequal(train_dataset, num_devices)
            else:
                # Chose euqal splits for every user
                device_groups = mnist_noniid(train_dataset, num_devices)
    else:
        exit("Error: unrecognized dataset name")

    return train_dataset, test_dataset, device_groups

def prepare_sys_data(devices_list, data_name, samp_type, unequal_splits):
    """downloading the dataset for training and testing."""
    train_dataset, test_dataset, device_groups = get_dataset(
        data_name, samp_type, unequal_splits, len(devices_list)
    )
    """ preparing the data for devices """
    device_data_idxs = {}
    for i, d in enumerate(devices_list):
        device_data_idxs[d] = device_groups[i]

    return train_dataset, test_dataset, device_data_idxs
