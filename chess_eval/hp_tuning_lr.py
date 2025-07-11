# %%
import random
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
from constants import BASE_DIR, SCALING_PATH
from torch.optim import Optimizer, lr_scheduler

from chess_eval.networks import Network
from chess_eval.schemas import CustomDataLoader, CustomDataset
from chess_eval.train import train, validate
from chess_eval.utils import prep_data


def main() -> None:
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    device = torch.device("cpu")

    data_dir = BASE_DIR / "data" / "processed"
    train_data_path = data_dir / "train" / "train.npy"
    val_data_path = data_dir / "val" / "val.npy"

    X_train, y_train = prep_data(train_data_path, SCALING_PATH)
    X_val, y_val = prep_data(val_data_path, SCALING_PATH)

    input_size = len(X_train[0, :])
    output_layer1 = 32
    output_layer2 = 16

    model = Network(input_size, output_layer1, output_layer2)

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run():
            params: dict[str, Any] = {
                "epochs": trial.suggest_categorical("epochs", [10, 50, 100]),
                "lr": trial.suggest_float("learning_rate", 1e-4, 0.8, log=True),
                "gamma": trial.suggest_float("gamma", 0.1, 0.9),
                "scheduler": trial.suggest_categorical(
                    "scheduler", ["step", "exponential"]
                ),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "step_size": trial.suggest_int("step_size", low=5, high=20, step=5),
                "optimizer": trial.suggest_categorical("optimiser", ["SGD", "Adam"]),
            }
            mlflow.log_params(params)

            optimizer: Optimizer
            if params["optimizer"] == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

            criterion_train = torch.nn.CrossEntropyLoss()
            criterion_test = torch.nn.CrossEntropyLoss(reduction="sum")

            scheduler: lr_scheduler.LRScheduler
            if params["scheduler"] == "exponential":
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
                train_loss, train_accuracy = train(
                    # params,
                    model,
                    device,
                    X_train_dataloader,
                    y_train_dataloader,
                    optimizer,
                    criterion_train,
                    criterion_test,
                    epoch,
                )
                val_loss, val_accuracy = validate(
                    model, device, X_val_dataloader, y_val_dataloader, criterion_test
                )

                H["train_loss"].append(train_loss)
                H["train_accuracy"].append(train_accuracy)
                H["val_loss"].append(val_loss)
                H["val_accuracy"].append(val_accuracy)

                scheduler.step()

            mlflow.log_metric("train_loss", H["train_loss"][-1])
            mlflow.log_metric("train_accuracy", H["train_accuracy"][-1])
            mlflow.log_metric("val_loss", H["val_loss"][-1])
            mlflow.log_metric("val_accuracy", H["val_accuracy"][-1])

            plt.style.use("ggplot")
            fig, ax = plt.subplots()
            ax.plot(
                np.arange(0, len(H["train_loss"])), H["train_loss"], label="train loss"
            )
            ax.plot(np.arange(0, len(H["val_loss"])), H["val_loss"], label="val loss")
            ax.plot(
                np.arange(0, len(H["train_accuracy"])),
                H["train_accuracy"],
                label="train accuracy",
            )
            ax.plot(
                np.arange(0, len(H["val_accuracy"])),
                H["val_accuracy"],
                label="val accuracy",
            )
            ax.set_title("Training and Validation Loss and Accuracy")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Loss/Accuracy")
            ax.legend()

            mlflow.log_figure(fig, "accuracy_loss_plot.png")

            mlflow.pytorch.log_model(model, "model")

        return val_accuracy

    tracking_uri = r"./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("chess_eval")

    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=50)


if __name__ == "__main__":
    main()
# %%
