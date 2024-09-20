from pathlib import Path
import yaml

from spaceship.models.base_ml import LogisticClassifier

from spaceship.data.data_processor import DataProcessor


def load_yaml_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_model(data: DataProcessor, config=None):
    model = LogisticClassifier(**config["params"]["LogisticRegression"])
    X_train, X_test, y_train, y_test = data.prepare_dataset()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score}")
    return model


if __name__ == "__main__":
    config = load_yaml_config(Path("hparams_log.yaml"))
    data_processor = DataProcessor(
        data_path=Path("spaceship/data/train.csv"), shuffle=True
    )
    dataset = data_processor.prepare_dataset()
    train_model(data_processor, config)
