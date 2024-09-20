from pathlib import Path
import yaml

import wandb
from tensorflow.keras.callbacks import Callback

from spaceship.models.base_tf import SpaceshipModel
from spaceship.data.data_processor import DataProcessor


class WandbMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        wandb.log(
            {
                "loss": logs["loss"],
                "accuracy": logs["accuracy"],
                "val_loss": logs["val_loss"],
                "val_accuracy": logs["val_accuracy"],
            }
        )

    def on_train_end(self, logs):
        wandb.log({"final_loss": logs["loss"],
                   "final_accuracy": logs["accuracy"]})


def load_yaml_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_model(data: DataProcessor, config=None):
    with wandb.init(
        project="spaceship", config=config
    ):  # nasz config yamla
        config = wandb.config
        model = SpaceshipModel()
        nn_arch = {
            "actfn1": config.actfn1,
            "actfn2": config.actfn2,
            "dropout1": config.dropout1,
            "dropout2": config.dropout2,
            "optimizer": config.optimizer,
            "loss": config.loss,
            "metrics": config.metrics,
        }
        train_cfg = {"epochs": config.epochs, "batch_size": config.batch_size}
        X_train, X_test, y_train, y_test = data.prepare_dataset()
        # breakpoint()
        model.build_model(**nn_arch)
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            callbacks=[WandbMetricsLogger()],
            **train_cfg,
        )
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy}, loss: {loss}")
        return model


if __name__ == "__main__":
    config = load_yaml_config(Path("wandb_sweep_nn.yaml"))
    sweep_id = wandb.sweep(config, project="spaceship")
    data_processor = DataProcessor(data_path=Path("spaceship/data/train.csv"))
    dataset = data_processor.prepare_dataset()
    # train_model(data_processor, config)
    wandb.agent(
        sweep_id, function=lambda: train_model(data_processor, config),
        count=10
    )
