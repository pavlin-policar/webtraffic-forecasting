from abc import abstractmethod, ABCMeta


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data):
        """Fit the model on some training data."""


class Model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, data):
        """Make predictions on given data."""
