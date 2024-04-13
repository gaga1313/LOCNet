import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import os

from src.data.create_dataset import GeiDataset, get_image_transform

from src.sl_utils import accuracy
from src.models import locnet


def evaluate(model, filter, dataset, batch_size, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        shuffle=False,
    )
    predictions = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        predictions = []
        true_labels = []
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.reshape(-1, 1)
            _, predicted_class = model(images)
            predicted_class = predicted_class.cpu()
            predictions.append(predicted_class)
            true_labels.append(labels)
        predictions = torch.vstack(predictions)
        true_labels = torch.vstack(true_labels)
        acc1, acc5 = accuracy(predictions, true_labels, topk=(1, 5))
        print(f"Top 1 Validation accuracy for filter {filter} : {acc1}")
        print(f"Top 5 Validation accuracy for filter {filter} : {acc5}")
    return acc1, acc5


def load_results(file_path):
    with open(file_path, "r") as file:
        accuracies = json.load(file)
    return accuracies


def main():
    model_path = "/cifs/data/tserre_lrs/projects/prj_model_vs_human/LOCNet/checkpoints/trial10/model_best.pth.tar"
    data_dir = "/cifs/data/tserre_lrs/projects/prj_model_vs_human/model-vs-human/model-vs-human/datasets"

    trial = model_path.split("/")[-2]
    result_file = trial + ".json"
    batch_size = 128

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = locnet().to(device)
    model.load_state_dict(torch.load(model_path)["state_dict"])

    image_transform = get_image_transform()

    accuracies = {}
    filters = [
        "cue-conflict",
        "edge",
        "silhouette",
        "sketch",
        "stylized",
        "colour",
        "contrast",
        "eidolonI",
        "eidolonII",
        "eidolonIII",
        "false-colour",
        "high-pass",
        "low-pass",
        "phase-scrambling",
        "power-equalisation",
        "rotation",
        "uniform-noise",
    ]

    start = time.time()

    for f in filters:
        dataset = GeiDataset(
            root_dir=data_dir,
            filter=f,
            image_transform=image_transform,
        )
        print(f"Total Sample Images : {len(dataset)}")  # sanity check

        top1_acc, top5_acc = evaluate(model, f, dataset, batch_size, device)
        accuracies[f] = {"top1_acc": top1_acc.item(), "top5_acc": top5_acc.item()}

    end = time.time()

    print(f"Time elapsed : {end - start}")
    print(accuracies)

    os.makedirs("./result", exist_ok=True)
    result_file_path = os.path.join("./result", result_file)

    with open(result_file_path, "w") as file:
        json.dump(accuracies, file)
