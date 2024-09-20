from pathlib import Path
import random
import yaml
import wandb

from spaceship.models.base_ml import LogisticClassifier
from spaceship.models.base_ml import RFClassifier

from spaceship.data.data_processor import DataProcessor


def load_yaml_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Przykład przygotowania danych
        data_processor = DataProcessor(
            data_path=Path("spaceship/data/train.csv"), shuffle=True
        )
        dataset = data_processor.prepare_dataset()
        X_train, X_test, y_train, y_test = dataset

        # Wybór modelu na podstawie config
        if config.model_type == "LogisticRegression":
            model_cfg = {
                "solver": config.solver,
                "max_iter": config.max_iter,
                "C": config.C,
                "penalty": config.penalty,
            }
            model = LogisticClassifier(**model_cfg)

        elif config.model_type == "RandomForest":
            model_cfg = {
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
            }
            model = RFClassifier(**model_cfg)

        # Trening modelu
        model.fit(X_train, y_train)

        # Ocena modelu
        score = model.score(X_test, y_test)
        wandb.log({"accuracy": score})

        print(f"Model accuracy: {score}")

        return model


# if __name__ == "__main__":
#     data_processor = DataProcessor(data_path=Path("spaceship/data/train.csv"), shuffle=True)
#     dataset = data_processor.prepare_dataset()
#     train_model(dataset)

if __name__ == "__main__":
    # Wczytanie konfiguracji sweepa z pliku YAML
    sweep_config = load_yaml_config("wandb_sweep_2.yaml")

    # Utworzenie sweepa na wandb
    sweep_id = wandb.sweep(sweep_config, project="test1")

    # Rozpoczęcie sweepa
    wandb.agent(
        sweep_id, function=train_model, count=24
    )  # count=10 oznacza, że chcemy 10 prób
