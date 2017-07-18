from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

from data_provider import get_language_dataset, TRAIN_DATA
from validation import validate_last_n_days, validate_forward_chaining


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data):
        """Fit the model on some training data."""


class Model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, data):
        """Make predictions on given data."""


class Delegator(metaclass=ABCMeta):
    def __init__(self, learner_cls):
        self.learner_cls = learner_cls

    def make_predictions(self, fname, name, **kwargs):
        pass

    def validate(self, n_days=60, **kwargs):
        langs = ['de', 'en', 'es', 'fr', 'ja', 'na', 'ru', 'zh']
        scores = []
        learner = self.learner_cls(**kwargs)

        for lang in langs:
            data = pd.read_csv(get_language_dataset(TRAIN_DATA, lang))
            score = validate_last_n_days(data, learner, n_days)
            print('%s SMAPE: %.2f' % (lang, score))
            scores.append(score)
        print('SMAPE: %.2f\n' % np.mean(scores))

    def validate_fc(self, folds=5, **kwargs):
        langs = ['de', 'en', 'es', 'fr', 'ja', 'na', 'ru', 'zh']
        scores = []
        learner = self.learner_cls(**kwargs)

        for lang in langs:
            data = pd.read_csv(get_language_dataset(TRAIN_DATA, lang))
            score = validate_forward_chaining(data, learner, folds)
            print('%s SMAPE: %.2f' % (lang, score))
            scores.append(score)
        print('SMAPE: %.2f\n' % np.mean(scores))
