# %%
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import device
from torch.optim import Optimizer, lr_scheduler
from yaml import safe_load

from chess_eval.constants import BASE_DIR, SCALING_PATH
from chess_eval.networks import Network
from chess_eval.schemas import CustomDataLoader, CustomDataset, TrainingParams
from chess_eval.utils import prep_data

logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to show
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train(
    model: nn.Module,
    device: device,
    X_train_dataloader: CustomDataLoader,
    y_train_dataloader: CustomDataLoader,
    optimizer: Optimizer,
    criterion1: nn.Module,
    criterion2: nn.Module,
    epoch: int,
    log_interval: int | None = None,
) -> tuple[float, float]:
    model.train()
    correct = 0
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(
        zip(X_train_dataloader, y_train_dataloader)
    ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion1(output, target)
        train_loss += criterion2(output, target).item()
        loss.backward()
        optimizer.step()

        pred_index = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        target_index = target.argmax(dim=1, keepdim=True)
        correct += (pred_index == target_index).sum().item()

        if log_interval:
            if batch_idx % log_interval == 0:
                print(
                    f"Epoch: {epoch} \tLoss: {loss.item():.6f} \
                    \t({batch_idx * len(data)}/{len(X_train_dataloader.dataset)})"
                )

    train_loss /= len(X_train_dataloader)
    train_accuracy = correct / len(X_train_dataloader.dataset)
    print(
        f"Train set: \tAverage loss:\t{train_loss:.4f} \
          \tAccuracy: {correct}/{len(X_train_dataloader.dataset)} ({100.0 * train_accuracy:.1f}%)"
    )
    return train_loss, train_accuracy


def validate(
    model: nn.Module,
    device: device,
    X_test_loader: CustomDataLoader,
    y_test_loader: CustomDataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in zip(X_test_loader, y_test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred_index = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            target_index = target.argmax(dim=1, keepdim=True)
            correct += (pred_index == target_index).sum().item()

    val_loss /= len(X_test_loader.dataset)
    val_accuracy = correct / len(X_test_loader.dataset)

    return val_loss, val_accuracy


def plot_graph(H: dict[str, list[float]], params: TrainingParams) -> None:
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H["train_loss"])), H["train_loss"], label="train loss")
    plt.plot(np.arange(0, len(H["val_loss"])), H["val_loss"], label="val loss")
    plt.plot(
        np.arange(0, len(H["train_accuracy"])),
        H["train_accuracy"],
        label="train accuracy",
    )
    plt.plot(
        np.arange(0, len(H["val_accuracy"])),
        H["val_accuracy"],
        label="val accuracy",
    )
    plt.title("Training and Validation Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(
        BASE_DIR
        / f"figures/epochs_{params['epochs']}__lr_{params['lr']}__gamma_{params['gamma']}__scheduler_{params['scheduler']}.png"
    )


def main(config_name: str, save_model: bool, log_interval: int) -> None:
    device = torch.device("cpu")
    config_path = BASE_DIR / "ml_config.yml"
    with open(config_path, encoding="utf-8") as f:
        config = safe_load(f)[config_name]

    params = TrainingParams(**config)
    data_dir = BASE_DIR / "data" / "processed"
    X_train, y_train = prep_data(data_dir / "train" / "train_d10.npy", SCALING_PATH)
    X_val, y_val = prep_data(data_dir / "val" / "val_d10.npy", SCALING_PATH)

    input_size = 70
    output_layer1 = 32
    output_layer2 = 16
    model = Network(
        input_size=input_size, output_layer1=output_layer1, output_layer2=output_layer2
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction="sum")

    scheduler: lr_scheduler.LRScheduler
    if params.get("scheduler", "") == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])
    else:
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=params["step_size"], gamma=params["gamma"]
        )

    X_train_dataloader = CustomDataLoader(
        CustomDataset(X_train[:, :]), batch_size=params["batch_size"]
    )
    y_train_dataloader = CustomDataLoader(
        CustomDataset(y_train[:, :]), batch_size=params["batch_size"]
    )
    X_val_dataloader = CustomDataLoader(
        CustomDataset(X_val[:, :]), batch_size=params["batch_size"]
    )
    y_val_dataloader = CustomDataLoader(
        CustomDataset(y_val[:, :]), batch_size=params["batch_size"]
    )

    H: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    for epoch in range(1, params["epochs"] + 1):
        print(f"Epoch: {epoch}")
        train_loss, train_accuracy = train(
            model,
            device,
            X_train_dataloader,
            y_train_dataloader,
            optimizer,
            criterion_train,
            criterion_test,
            epoch,
            log_interval,
        )
        val_loss, val_accuracy = validate(
            model, device, X_val_dataloader, y_val_dataloader, criterion_test
        )

        H["train_loss"].append(train_loss)
        H["train_accuracy"].append(train_accuracy)
        H["val_loss"].append(val_loss)
        H["val_accuracy"].append(val_accuracy)

        scheduler.step()

        if epoch % 5 == 0:
            plot_graph(H, params)

    if save_model:
        models_dir = BASE_DIR / "models"
        torch.save(model.state_dict(), models_dir / f"{config_name}_model.pt")  # nosec: CWE-502


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The name of the configuration from the ml_config.yml file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test",
        help="Navigate to ml_config.yml to choose your configuration and enter the name",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        metavar="l",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="For Saving the current Model",
    )

    args = parser.parse_args()
    main(args.config, args.save_model, args.log_interval)
