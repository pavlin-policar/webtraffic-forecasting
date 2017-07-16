from abc import ABCMeta, abstractmethod


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data):
        """Fit the model on some training data."""


class Model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, data):
        """Make predictions on given data."""


class LastNDaysMedianLearner(Learner):
    def __init__(self, days_to_consider=49):
        self.days_to_consider = days_to_consider

    def fit(self, data):
        data['Visits'] = data[data.columns[-self.days_to_consider:]].median(axis=1)
        return MedianModel(data[['Page', 'Visits']])


class MedianModel(Model):
    def __init__(self, data):
        assert 'Page' in data.columns and 'Visits' in data.columns, \
            'Data must contain the `Page` and `Visits` columns.'
        self.data = data

    def predict(self, data):
        data = data.merge(self.data[['Page', 'Visits']], how='left')
        return data
