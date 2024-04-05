import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import json

from models import locnet
from dataset import filterdataset, transforms

# from metrics import accuracy
from utils import constants, plot
import sl_utils


def evaluate(model, filter, level, train_dataset, val_dataset, batch_size=512):
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
        shuffle=False,
    )
    model.to(device)
    predictions = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        predictions = []
        true_labels = []
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.reshape(-1, 1)
            outputs = model(images).cpu()
            predictions.append(outputs)
            true_labels.append(labels)
        predictions = torch.vstack(predictions)
        true_labels = torch.vstack(true_labels)
        acc1, acc = sl_utils.accuracy(predictions, true_labels, topk=(1, 5))
        print(f"Validation accuracy for filter {filter} - level {level}: {acc1}")
    return acc1, acc


def load_results(file_path):
    with open(file_path, "r") as file:
        accuracies = json.load(file)
    return accuracies


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = vgg11_bn.CustomVGG11_BN()
    for params in model.parameters():
        params.requires_grad = False

    accuracies = {}
    start = time.time()
    for f, flevels in constants.filter_levels.items():
        accuracies[f] = {}
        for level_path in flevels:
            level = level_path.split("/")[-1]
            train_dataset = filterdataset.FilterDataset(
                data_dir=constants.train_dir,
                filter=f,
                level=level,
                transform=transforms.transform,
            )
            val_dataset = filterdataset.FilterDataset(
                data_dir=constants.val_dir,
                filter=f,
                level=level,
                transform=transforms.transform,
            )
            train_acc, val_acc = evaluate(model, f, level, train_dataset, val_dataset)
            accuracies[f][level] = {"train_acc": train_acc, "val_acc": val_acc}
    end = time.time()
    print(f"Time elapsed : {end - start}")
    print(accuracies)

    # plot.plot_levelwise_corr(accuracies, 'vgg_11_bn', constants.plot_dir)
