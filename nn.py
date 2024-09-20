from pathlib import Path
import yaml


from spaceship.models.base_tf import SpaceshipModel
from spaceship.data.data_processor import DataProcessor


def load_yaml_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def train_model(data: DataProcessor, config=None):
    model = SpaceshipModel()
    cfg = config["params"]
    X_train, X_test, y_train, y_test = data.prepare_dataset()
    # breakpoint()
    model.build_model()
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
    )
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy}, loss: {loss}")
    return model


if __name__ == "__main__":
    config = load_yaml_config(Path("hparams_nn.yaml"))
    data_processor = DataProcessor(
        data_path=Path("spaceship/data/train.csv"), shuffle=True
    )
    train_model(data_processor, config)
