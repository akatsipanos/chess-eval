# %%
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from constants import BASE_DIR, SCALING_PATH
from torch import device
from torch.optim import Optimizer, lr_scheduler

from chess_eval.networks import Network
from chess_eval.schemas import CustomDataLoader, CustomDataset
from chess_eval.utils import prep_data


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

    # train_loss /= len(cast(Sized, X_train_dataloader.dataset))
    # dataset: SizedDataset =

    train_loss /= len(X_train_dataloader)  # .dataset)
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


def main() -> None:
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--epochs",
            type=int,
            default=14,
            metavar="N",
            help="number of epochs to train (default: 14)",
        )
        ap.add_argument(
            "--lr",
            type=float,
            default=1.0,
            metavar="LR",
            help="learning rate (default: 1.0)",
        )
        ap.add_argument(
            "--gamma",
            type=float,
            default=0.9,
            metavar="M",
            help="Learning rate step gamma (default: 0.7)",
        )
        ap.add_argument(
            "--log-interval",
            type=int,
            default=1000,
            metavar="N",
            help="how many batches to wait before logging training status",
        )
        ap.add_argument(
            "--scheduler",
            type=str,
            default="step",
            metavar="S",
            help="learning rate scheduler to use, step or exponential",
        )
        ap.add_argument(
            "--batch-size",
            type=int,
            default=256,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        ap.add_argument(
            "--save-model",
            type=str,
            default=False,
            metavar="S",
            help="For Saving the current Model",
        )
        ap.add_argument(
            "--step-size", type=int, default=10, help="Step size for lr scheduler"
        )
        args = vars(ap.parse_args())

    except SystemExit:
        args = {
            "epochs": 100,
            "lr": 0.1,
            "gamma": 0.1,
            "log_interval": 10000,
            "save_model": False,
        }

    device = torch.device("cpu")

    data_dir = BASE_DIR / "data" / "processed"
    X_train, y_train = prep_data(data_dir / "train" / "train_d10.npy", SCALING_PATH)
    X_val, y_val = prep_data(data_dir / "val" / "val_d10.npy", SCALING_PATH)

    # model = Conv()
    input_size = 70
    output_layer1 = 32
    output_layer2 = 16
    model = Network(
        input_size=input_size, output_layer1=output_layer1, output_layer2=output_layer2
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"])
    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction="sum")

    scheduler: lr_scheduler.LRScheduler
    if args.get("scheduler", "") == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args["gamma"])
    else:
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args["step_size"], gamma=args["gamma"]
        )

    X_train_dataloader = CustomDataLoader(
        CustomDataset(X_train[:, :]), batch_size=args["batch_size"]
    )
    y_train_dataloader = CustomDataLoader(
        CustomDataset(y_train[:, :]), batch_size=args["batch_size"]
    )
    X_val_dataloader = CustomDataLoader(
        CustomDataset(X_val[:, :]), batch_size=args["batch_size"]
    )
    y_val_dataloader = CustomDataLoader(
        CustomDataset(y_val[:, :]), batch_size=args["batch_size"]
    )

    H: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    for epoch in range(1, args["epochs"] + 1):
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
            args["log_interval"],
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
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(
                np.arange(0, len(H["train_loss"])), H["train_loss"], label="train loss"
            )
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
                f"figures/epochs_{args['epochs']}__lr_{args['lr']}__gamma_{args['gamma']}__scheduler_{args['scheduler']}.png"
            )

    if args["save_model"]:
        torch.save(model.state_dict(), "models/chess_model.pt")  # nosec: CWE-502


if __name__ == "__main__":
    main()
#  python train.py --epochs 65 --lr 0.1 --gamma 0.8 --schedular step --step-size 15# %%
