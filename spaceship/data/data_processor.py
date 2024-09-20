from pathlib import Path
import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(
        self,
        data_path: Path,
        categorical_features: List[str] = ["HomePlanet", "Deck", "Side"],
        numerical_features: List[str] = [
            "Age",
            "TotalSpendings",
            "RoomService",
            "Spa",
            "VRDeck",
            "TotalSpendings",
        ],
        shuffle: bool = True,
    ):
        df = pd.read_csv(data_path)
        passids = [x.split("_")[0] for x in df["PassengerId"]]
        self.pass_dict = {x: passids.count(x) for x in set(passids)}
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.shuffle = shuffle
        self.ohe = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        self.df = self.process_data(df)

    def prepare_dataset(self):
        X_train, X_test, y_train, y_test = self.split_data(self.df)

        return X_train, X_test, y_train, y_test

    def process_data(self, df) -> pd.DataFrame:
        df = df.copy()
        df = self.process_features(df)
        df = df.dropna().reset_index(drop=True)
        cat_df = self.encode_categorical(df)
        num_df = self.scale_numerical(df)
        label = df["Transported"]
        df = pd.concat([cat_df, num_df, label], axis=1)
        return df

    def split_data(self, df, test_size: float = 0.2):

        X = df.drop("Transported", axis=1)
        y = df["Transported"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=self.shuffle
        )

        return X_train, X_test, y_train, y_test

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_df = df[self.categorical_features]
        cat_df = self.ohe.fit_transform(cat_df)
        cat_df = pd.DataFrame(cat_df, columns=self.ohe.get_feature_names_out())
        return cat_df

    def scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        num_df = df[self.numerical_features]
        num_df = self.scaler.fit_transform(num_df)
        num_df = pd.DataFrame(num_df, columns=self.numerical_features)
        return num_df

    def fill_home_planet(self, row: pd.Series) -> str:
        if row["Deck"] in ["A", "B", "C"]:
            return "Europa"
        elif row["Deck"] in ["D"]:
            return "Mars"
        elif row["Deck"] in ["E", "F", "G"]:
            return "Earth"
        else:
            return None

    def fill_cryo_sleep(self, row):
        if row["ServiceUsed"] == 1:
            return 0
        else:
            return 1

    def fill_deck(self, row):
        if row["HomePlanet"] == "Europa":
            return random.choice(["A", "B", "C"])
        elif row["HomePlanet"] == "Mars":
            return "D"
        elif row["HomePlanet"] == "Earth":
            return random.choice(["E", "F", "G"])
        else:
            return "A"

    def fill_vip(self, row):
        if row["TotalSpendings"] > 2000:
            return 1
        else:
            return 0

    def fill_spending_values(self, df, row, column):
        if pd.isna(row[column]):
            if row["VIP"] is True:
                return df.loc[df["VIP"], column].median()
            else:
                return df.loc[df["VIP"], column].median()
        else:
            return row[column]

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PairTravel"] = df["PassengerId"].apply(
            lambda x: self.pass_dict[x.split("_")[0]] > 1
        )
        df[["Deck", "Room", "Side"]] = df["Cabin"].str.split("/", expand=True)
        df["HomePlanet"] = df.apply(
            lambda x: (
                self.fill_home_planet(x)
                if pd.isna(x["HomePlanet"])
                else x["HomePlanet"]
            ),
            axis=1,
        )
        df["ServiceUsed"] = df.apply(
            lambda x: (
                1
                if any(
                    x[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] > 0
                )
                else 0
            ),
            axis=1,
        )
        df["CryoSleep"] = df.apply(
            lambda x: (
                self.fill_cryo_sleep(x) if pd.isna(x["CryoSleep"]) else x["CryoSleep"]
            ),
            axis=1,
        )
        df["Deck"] = df.apply(
            lambda x: self.fill_deck(x) if pd.isna(x["Deck"]) else x["Deck"], axis=1
        )
        df["TotalSpendings"] = (
            df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
            .fillna(0)
            .sum(axis=1)
        )
        df["VIP"] = df.apply(
            lambda x: self.fill_vip(x) if pd.isna(x["VIP"]) else x["VIP"], axis=1
        )
        df["RoomService"] = df.apply(
            lambda x: self.fill_spending_values(df, x, "RoomService"), axis=1
        )
        df["Spa"] = df.apply(lambda x: self.fill_spending_values(df, x, "Spa"), axis=1)
        df["VRDeck"] = df.apply(
            lambda x: self.fill_spending_values(df, x, "VRDeck"), axis=1
        )

        df["PairTravel"] = df["PairTravel"].astype(int)
        df["CryoSleep"] = df["CryoSleep"].astype(int)
        df["VIP"] = df["VIP"].astype(int)
        df["ServiceUsed"] = df["ServiceUsed"].astype(int)
        df["Transported"] = df["Transported"].apply(
            lambda x: int(x) if not pd.isna(x) else x
        )

        return df


if __name__ == "__main__":
    data_processor = DataProcessor(Path("train.csv"))
    print(data_processor.df)
