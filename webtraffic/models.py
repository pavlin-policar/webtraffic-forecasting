from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd
from os.path import isdir, join, dirname, exists, split

from os import listdir, makedirs

from data_provider import get_language_dataset, TRAIN_DATA, prepare_test_data, \
    PREDICTIONS_DIR, save_predictions
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

    def make_predictions(self, name, train, test=None, **kwargs):
        # Create an instance of the learner
        learner = self.learner_cls(**kwargs)

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

        for ftrain, ftest in zip(training_data, testing_data):
            model = learner.fit(pd.read_csv(ftrain))
            predictions = model.predict(prepare_test_data(pd.read_csv(ftest)))

            new_fname = join(base_dir, '%s.csv' % split(ftrain)[-1][:2])
            save_predictions(predictions, new_fname)

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
