# %%
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import Optimizer, lr_scheduler

from chess_eval.constants import BASE_DIR, SCALING_PATH
from chess_eval.networks import Network_1h, Network_2h, Network_3h
from chess_eval.schemas import CustomDataLoader, CustomDataset
from chess_eval.train import train, validate
from chess_eval.utils import prep_data


def main() -> None:
    torch.manual_seed(123)

    np.random.seed(123)

    device = torch.device("cpu")

    data_dir = BASE_DIR / "data" / "processed"
    train_data_path = data_dir / "train" / "train.npy"
    val_data_path = data_dir / "val" / "val.npy"

    X_train, y_train = prep_data(train_data_path, SCALING_PATH)
    X_val, y_val = prep_data(val_data_path, SCALING_PATH)

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run():
            params = {
                "n_hidden": trial.suggest_categorical("n_hidden", [1, 2, 3]),
                "h1": trial.suggest_categorical("h1", [32, 64, 128, 256]),
                "h2": trial.suggest_categorical("h2", [16, 32, 64, 128]),
                "h3": trial.suggest_categorical("h3", [8, 16, 32, 64]),
            }
            mlflow.log_params(params)

            input_size = len(X_train[0, :])
            neurons = [params["h1"], params["h2"], params["h3"]]

            model: nn.Module
            if params["n_hidden"] == 1:
                model = Network_1h(input_size, neurons[0])

            elif params["n_hidden"] == 2:
                model = Network_2h(input_size, neurons[0], neurons[1])

            else:
                model = Network_3h(input_size, neurons[0], neurons[1], neurons[2])

            optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

            criterion_train = torch.nn.CrossEntropyLoss()
            criterion_test = torch.nn.CrossEntropyLoss(reduction="sum")

            scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

            X_train_dataloader = CustomDataLoader(CustomDataset(X_train[:, :]), 256)
            y_train_dataloader = CustomDataLoader(CustomDataset(y_train[:, :]), 256)
            X_val_dataloader = CustomDataLoader(CustomDataset(X_val[:, :]), 256)
            y_val_dataloader = CustomDataLoader(CustomDataset(y_val[:, :]), 256)

            H: dict[str, list[float]] = {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
            }
            epochs = 65
            for epoch in range(1, epochs + 1):
                train_loss, train_accuracy = train(
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
    mlflow.set_experiment("chess_eval_architecture")

    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50)


if __name__ == "__main__":
    main()
# %%
