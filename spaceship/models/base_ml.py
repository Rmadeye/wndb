from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class BaseClassifier(BaseEstimator, ClassifierMixin, ABC):
    def __init__(self, **params):
        self.set_params(**params)

    def set_params(self, **params):
        """
        Ustawia parametry modelu.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        """
        Zwraca słownik z parametrami modelu zgodny z API Sklearn.
        """
        return {
            key: getattr(self,
                         key) for key in self.__dict__ if not key.startswith(
                             "_")
        }

    @abstractmethod
    def fit(self, X, y):
        """
        Dopasowuje model do danych.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Dokonuje predykcji na podstawie dopasowanego modelu.
        """
        pass

    def score(self, X, y):
        """
        Oblicza accuracy na podstawie danych testowych.
        """
        y_pred = self.predict(X)
        return (y_pred == y).mean()  # Accuracy


class LogisticClassifier(BaseClassifier):
    def __init__(self, solver: str, max_iter: str, **params):
        super().__init__(solver=solver, max_iter=max_iter, **params)
        self.model = LogisticRegression(solver=solver, max_iter=max_iter)

    def fit(self, X, y):
        """
        Dopasowuje model regresji logistycznej.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Dokonuje predykcji etykiet klas.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Zwraca prawdopodobieństwa przynależności do klasy.
        """
        return self.model.predict_proba(X)


class RFClassifier(BaseClassifier):
    def __init__(self, n_estimators: int, max_depth: int, **params):
        super().__init__(n_estimators=n_estimators,
                         max_depth=max_depth, **params)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
