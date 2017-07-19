from abc import abstractmethod, ABCMeta
from datetime import date
from os import listdir, makedirs
from os.path import isdir, join, exists, split

import numpy as np
import pandas as pd

from data_provider import get_language_dataset, TRAIN_DATA, prepare_test_data, \
    PREDICTIONS_DIR, save_predictions, get_date_columns
from validation import validate_last_n_days, validate_forward_chaining, \
    validate_time_period


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data):
        """Fit the model on some training data."""


class Model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, data):
        """Make predictions on given data."""


class Delegator(metaclass=ABCMeta):
    def __init__(self, learner_cls, cv_params=None):
        self.learner_cls = learner_cls
        self.cv_params = cv_params or {}

    @staticmethod
    def _parse_prediction_files(name, train, test, learner):
        if isdir(train):
            training_data = sorted(join(train, f) for f in listdir(train) if 'train_1' in f)
            testing_data = sorted(join(train, f) for f in listdir(train) if 'key_1' in f)
            base_dir = join(PREDICTIONS_DIR, '%s_%s' % (name, learner.__class__.__name__))
        else:
            assert test is not None, 'If the training data is a file, the ' \
                                     'testing data file must be given.'
            training_data, testing_data = [train], [test]
            base_dir = PREDICTIONS_DIR

        if not exists(base_dir):
            makedirs(base_dir)

        return training_data, testing_data, base_dir

    @staticmethod
    def _perform_imputation(data):
        """Perform initial imputation on data.

        Notes
        -----
          - Pandas `bfill` imputation produces better validation results, but
            performs worse on LB

        """
        return data

    def make_predictions(self, name, train, test=None, **kwargs):
        learner = self.learner_cls(**kwargs)
        training_data, testing_data, base_dir = self._parse_prediction_files(
            name, train, test, learner)

        for ftrain, ftest in zip(training_data, testing_data):
            train = pd.read_csv(ftrain)
            train = self._perform_imputation(train)

            model = learner.fit(train)
            predictions = model.predict(prepare_test_data(pd.read_csv(ftest)))
            predictions.fillna(0, inplace=True)

            new_fname = join(base_dir, '%s.csv' % split(ftrain)[-1][:2])
            save_predictions(predictions, new_fname)

    def make_predictions_time_period(self, name, train, test=None, **kwargs):
        """Make predictions using only training data up to a cutoff date."""
        learner = self.learner_cls(**kwargs)
        training_data, testing_data, base_dir = self._parse_prediction_files(
            name, train, test, learner)

        for ftrain, ftest in zip(training_data, testing_data):
            train = pd.read_csv(ftrain)
            train = self._perform_imputation(train)

            date_columns = [date(*(int(x) for x in c.split('-')))
                            for c in get_date_columns(train)]

            cutoff_date = date(2016, 1, 1)
            assert cutoff_date in date_columns, \
                'Starting date must be present in the data.'

            training_dates = [c for c in date_columns if c < cutoff_date]
            training_cols = ['%d-%02d-%02d' % (d.year, d.month, d.day)
                             for d in training_dates]

            model = learner.fit(train[['Page'] + training_cols])

            test = prepare_test_data(pd.read_csv(ftest))
            predictions = model.predict(test)
            predictions.fillna(0, inplace=True)

            new_fname = join(base_dir, '%s.csv' % split(ftrain)[-1][:2])
            save_predictions(predictions, new_fname)

    def validate(self, n_days=60, **kwargs):
        langs = ['de', 'en', 'es', 'fr', 'ja', 'na', 'ru', 'zh']
        scores = []
        learner = self.learner_cls(**kwargs)

        for lang in langs:
            data = pd.read_csv(get_language_dataset(TRAIN_DATA, lang))
            data = self._perform_imputation(data)
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
            data = self._perform_imputation(data)
            score = validate_forward_chaining(data, learner, folds)
            print('%s SMAPE: %.2f' % (lang, score))
            scores.append(score)
        print('SMAPE: %.2f\n' % np.mean(scores))

    def validate_tp(self, **kwargs):
        langs = ['de', 'en', 'es', 'fr', 'ja', 'na', 'ru', 'zh']
        scores = []
        learner = self.learner_cls(**kwargs)

        for lang in langs:
            data = pd.read_csv(get_language_dataset(TRAIN_DATA, lang))
            data = self._perform_imputation(data)

            start, end = date(2016, 1, 1), date(2016, 3, 1)
            score = validate_time_period(data, learner, start, end)
            print('%s SMAPE: %.2f' % (lang, score))
            scores.append(score)
        print('SMAPE: %.2f\n' % np.mean(scores))

    def cross_validate(self, method):
        for key, value_set in self.cv_params.items():
            for value in value_set:
                print('%s: %s' % (key, value))
                getattr(self, method)(**{key: value})

