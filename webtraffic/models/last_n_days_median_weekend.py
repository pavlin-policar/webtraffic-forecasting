import re

import numpy as np
import pandas as pd

from models import Learner, Model


class LastNDaysMedianWithWeekenedLearner(Learner):
    def __init__(self, days_to_consider=49):
        self.days_to_consider = days_to_consider

    def fit(self, data):
        date_columns = [c for c in data.columns
                        if re.match(r'\d{4}-\d{2}-\d{2}', c)]
        data = pd.melt(
            data[date_columns[-self.days_to_consider:] + ['Page']],
            id_vars='Page', var_name='date', value_name='Visits',
        )
        data['date'] = data['date'].astype('datetime64[ns]')
        data['weekend'] = (data.date.dt.dayofweek // 5 == 1).astype(np.float64)
        data_dow = data.groupby(['Page', 'weekend']).median().reset_index()

        return MedianWithWeekenedModel(data_dow[['Page', 'weekend', 'Visits']])


class MedianWithWeekenedModel(Model):
    def __init__(self, data):
        assert 'Page' in data.columns and 'Visits' in data.columns, \
            'Data must contain the `Page` and `Visits` columns.'
        self.data = data

    def predict(self, data):
        data['date'] = data.Page.apply(lambda a: a[-10:])
        data['date'] = data['date'].astype('datetime64[ns]')
        data['weekend'] = (data.date.dt.dayofweek // 5 == 1).astype(np.float64)
        data['Page'] = data.Page.apply(lambda a: a[:-11])

        data.merge(self.data[['Page', 'Visits']], how='left')

        return data