import fire
import numpy as np
import pandas as pd

from data_provider import get_date_columns
from models import Learner, Model, Delegator


class LastNDaysMedianWithWeekenedLearner(Learner):
    def __init__(self, days_to_consider=56):
        self.days_to_consider = days_to_consider

    def fit(self, data):
        date_columns = get_date_columns(data)
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
        cols = data.columns
        assert 'Page' in cols and 'weekend' in cols and 'Visits' in cols, \
            'Data must contain the `Page`, `weekend` and `Visits` columns.'
        self.data = data

    def predict(self, data):
        data['weekend'] = (data.date.dt.dayofweek // 5 == 1).astype(np.float64)

        data = data.merge(self.data, how='left')

        return data

if __name__ == '__main__':
    fire.Fire(Delegator(LastNDaysMedianWithWeekenedLearner, cv_params={
        'days_to_consider': range(14, 100, 7),
    }))
