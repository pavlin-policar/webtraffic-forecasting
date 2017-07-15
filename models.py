from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, train):
        """Fit the model on some training data."""

    @abstractmethod
    def predict(self, test):
        """Make predictions on given data."""


class LastNDaysMedianModel(Model):
    def __init__(self, days_to_consider=49):
        self.days_to_consider = days_to_consider
        self.data = None

    def fit(self, train):
        self.data = data = train
        data['Prediction'] = data[data.columns[-self.days_to_consider:]].median(axis=1)

    def predict(self, test):
        assert self.data is not None, 'Model must first be fitted!'
        test = test.merge(self.data[['Page', 'Prediction']], how='left')
        return test
