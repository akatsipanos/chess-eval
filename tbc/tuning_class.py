# WIP
# type: ignore
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
from torch.optim import Optimizer, lr_scheduler

from chess_eval.constants import SCALING_PATH
from chess_eval.networks import Network, Network_1h, Network_2h, Network_3h
from chess_eval.schemas import CustomDataLoader, CustomDataset, TuningParams
from chess_eval.train import train, validate
from chess_eval.utils import prep_data


@dataclass
class Tuning:
    """
    Maybe put all the values from params into the init and then have
    an initialiser function that gets back their values, obviously being
    fixed or variable depending on tuning type
    """

    train_data_path: Path
    val_data_path: Path
    tuning_type: Literal["Parameters", "Architecture"]
    experiment_name: str
    tracking_uri: str
    n_trials: int = 20
    batch_size: int = 256
    epochs: int = 65
    model: torch.nn.Module = field(init=False, default_factory=torch.nn.Module)

    def __post_init__(self):
        torch.manual_seed(123)
        np.random.seed(123)
        self.device: Any = torch.device("cpu")

        self.optimizer: Optimizer = torch.optim.SGD(self.model.parameters(), lr=0.25)
        self.scheduler: lr_scheduler.LRScheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=15, gamma=0.8
        )

        self.X_train, self.y_train = prep_data(self.train_data_path, SCALING_PATH)
        self.X_val, self.y_val = prep_data(self.val_data_path, SCALING_PATH)
        self.input_size = len(self.X_train[0, :])

        if self.tuning_type == "Parameters":
            output_layer1 = 32
            output_layer2 = 16

            self.model = Network(self.input_size, output_layer1, output_layer2)

    def _set_vals(self, params: TuningParams) -> None:
        if not isinstance(params, TuningParams):
            raise ValueError("Params must be of type TuningParameters when tuning")
        if self.tuning_type == "Parameters":
            if params["optimizer"] == "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=params["lr"])
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])

            if params["scheduler"] == "exponential":
                self.scheduler = lr_scheduler.ExponentialLR(
                    optimizer, gamma=params["gamma"]
                )
            else:
                self.scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=params["step_size"], gamma=params["gamma"]
                )

            self.epoch = params["epochs"]
            self.batch_size = params["batch_size"]
        else:
            neurons = [params["h1"], params["h2"], params["h3"]]

            # model: nn.Module
            if params["n_hidden"] == 1:
                self.model = Network_1h(self.input_size, neurons[0])

            elif params["n_hidden"] == 2:
                self.model = Network_2h(self.input_size, neurons[0], neurons[1])

            else:
                self.model = Network_3h(
                    self.input_size, neurons[0], neurons[1], neurons[2]
                )

    def run_trials(self, params: TuningParams):
        def objective(trial: optuna.Trial) -> float:
            with mlflow.start_run():
                mlflow.log_params(params)

                self._set_vals(params)
                # scheduler, batch_size, epoch, optimiser = self._get_vals(params)

                criterion_train = torch.nn.CrossEntropyLoss()
                criterion_test = torch.nn.CrossEntropyLoss(reduction="sum")

                X_train_dataloader = CustomDataLoader(
                    CustomDataset(self.X_train[:, :]), self.batch_size
                )
                y_train_dataloader = CustomDataLoader(
                    CustomDataset(self.y_train[:, :]), self.batch_size
                )
                X_val_dataloader = CustomDataLoader(
                    CustomDataset(self.X_val[:, :]), self.batch_size
                )
                y_val_dataloader = CustomDataLoader(
                    CustomDataset(self.y_val[:, :]), self.batch_size
                )

                H: dict[str, list[float]] = {
                    "train_loss": [],
                    "val_loss": [],
                    "train_accuracy": [],
                    "val_accuracy": [],
                }
                # epochs = 65
                for epoch in range(1, self.epochs + 1):
                    train_loss, train_accuracy = train(
                        self.model,
                        self.device,
                        X_train_dataloader,
                        y_train_dataloader,
                        self.optimizer,
                        criterion_train,
                        criterion_test,
                        epoch,
                    )
                    val_loss, val_accuracy = validate(
                        self.model,
                        self.device,
                        X_val_dataloader,
                        y_val_dataloader,
                        criterion_test,
                    )

                    H["train_loss"].append(train_loss)
                    H["train_accuracy"].append(train_accuracy)
                    H["val_loss"].append(val_loss)
                    H["val_accuracy"].append(val_accuracy)

                    self.scheduler.step()

                mlflow.log_metric("train_loss", H["train_loss"][-1])
                mlflow.log_metric("train_accuracy", H["train_accuracy"][-1])
                mlflow.log_metric("val_loss", H["val_loss"][-1])
                mlflow.log_metric("val_accuracy", H["val_accuracy"][-1])

                plt.style.use("ggplot")
                fig, ax = plt.subplots()
                ax.plot(
                    np.arange(0, len(H["train_loss"])),
                    H["train_loss"],
                    label="train loss",
                )
                ax.plot(
                    np.arange(0, len(H["val_loss"])), H["val_loss"], label="val loss"
                )
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

                mlflow.pytorch.log_model(self.model, "model")
            return val_accuracy

        # def orchestrate(self, experiment_name: str, n_trials: int, sampler: str):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # sampler = optuna.samplers.TPESampler(seed=123)
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(objective, self.n_trials)
