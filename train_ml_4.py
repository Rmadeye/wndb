from pathlib import Path
import yaml
import wandb

from spaceship.models.base_ml import LogisticClassifier
from spaceship.data.data_processor import DataProcessor
from spaceship.utils import plot_confusion_matrix


def load_yaml_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_model(data: DataProcessor, config=None):
    with wandb.init(
        project="spaceship", config=config  
    ):  # nasz config yamla
        config = wandb.config  # nasz config ze sweepa
        model_cfg = {
            "solver": config.solver,
            "max_iter": config.max_iter,
            "C": config.C,
            "penalty": config.penalty,
        }

        model = LogisticClassifier(**model_cfg)
        X_train, X_test, y_train, y_test = data.prepare_dataset()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        yhat = model.predict(X_test)
        cm = plot_confusion_matrix(yhat, y_test)
        # cm.savefig('cm.png')
        wandb.log(
            {
                "accuracy": score,
                "model": "LogisticRegression",
                "train_len": len(X_train),
                "confusion_matrix": wandb.Image(cm),
            }
        )

        return model


if __name__ == "__main__":
    config = load_yaml_config(Path("wandb_sweep.yaml"))
    sweep_id = wandb.sweep(config, project="spaceship")
    data_processor = DataProcessor(data_path=Path("spaceship/data/train.csv"))
    dataset = data_processor.prepare_dataset()
    # train_model(data_processor, config)
    wandb.agent(
        sweep_id, function=lambda: train_model(data_processor, config), count=10
    )
