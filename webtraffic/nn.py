import shutil
from os import makedirs
from os.path import join, exists, split, dirname

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import RandomSampler, SequentialSampler, \
    BatchSampler

from data_provider import MODELS_DIR, TEST_DATA, TRAIN_DATA, \
    get_date_columns, save_predictions, PREDICTIONS_DIR
from ml_dataset import ML_VALIDATION, ML_TRAIN, LAG_DAYS, \
    lag_test_set_fname, get_lag_columns

N_EPOCHS = 100
BATCH_SIZE = 512


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = 100
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.selu = nn.SELU()
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.selu(self.input(x))
        out = self.selu(self.fc1(out))
        # out = self.dropout(out)
        # out = self.selu(self.fc2(out))
        # out = self.dropout(out)
        # out = self.selu(self.fc3(out))
        out = self.output(out)
        return out


def get_model(input_size, output_size):
    return LinearRegression(input_size, output_size)


def compute_loss(data, y_data, criterion, model):
    losses = []
    for batch_idx in BatchSampler(SequentialSampler(data), BATCH_SIZE, False):
        x, y = data.iloc[batch_idx], y_data.iloc[batch_idx]
        x = Variable(torch.from_numpy(x.values), volatile=True).cuda()
        y = Variable(torch.from_numpy(y.values), volatile=True).cuda()

        y_hat = model(x)
        # Compute the loss and update the weights
        loss = criterion(y, y_hat)
        losses.append(loss.data[0])

    return np.mean(losses)


def save_checkpoint(model_name, state, is_best):
    model_dir = join(MODELS_DIR, model_name)
    if not exists(model_dir):
        makedirs(model_dir)

    checkpoint_fname = join(model_dir, 'checkpoint.tar')
    torch.save(state, checkpoint_fname)
    if is_best:
        shutil.copyfile(checkpoint_fname, join(model_dir, 'model_best.tar'))


def load_data(data):
    # Flags to indicate type of data
    training = 'Visits' in data

    # Add local mean and median
    lag_columns = get_lag_columns(LAG_DAYS)
    data['window_mean'] = data[lag_columns].mean(axis=1)
    data['window_std'] = data[lag_columns].std(axis=1)
    data['window_median'] = data[lag_columns].median(axis=1)
    # Rescale lag columns
    data[lag_columns] = data[lag_columns].sub(data['window_mean'], axis=0)
    data[lag_columns] = data[lag_columns].div(data['window_std'], axis=0)
    # Since sometimes division produces NaNs, we'll replace those with 0s
    data.fillna(0, inplace=True)

    # Set correct dtypes
    data['date'] = data['date'].astype('datetime64[ns]')

    # # Add some extra helpful features
    def category(result, **kwargs):
        return pd.Categorical(result, **kwargs)

    data['day_of_week'] = category(
        data.date.dt.dayofweek, categories=list(range(6))
    )
    data['weekend'] = category(
        data.date.dt.dayofweek // 5 == 1, categories=[True, False]
    )
    data['day'] = category(
        data.date.dt.day, categories=list(range(1, 32))
    )
    data['month'] = category(
        data.date.dt.month, categories=list(range(1, 13))
    )
    data['season'] = category(
        data.date.dt.month // 4, categories=list(range(4))
    )

    data[['title', 'lang', 'access', 'agent']] = data['Page'].str.extract(
        '(.+)_(\w{2})\.wikipedia\.org_([^_]+)_([^_]+)')
    data['lang'] = data['lang'].astype(
        'category', categories=['de', 'en', 'es', 'fr', 'ja', 'ru', 'zh']
    )
    data['access'] = data['access'].astype(
        'category', categories=['all-access', 'desktop', 'mobile-web']
    )
    data['agent'] = data['agent'].astype(
        'category', categories=['all-agents', 'spider']
    )

    # Remove columns that we won't use for training
    data.drop(['Page', 'title', 'date'], inplace=True, axis=1)

    # Prepare the data for training
    data = pd.get_dummies(data).astype(np.float32)

    # If training data, we will have access to the `Visits` column
    if training:
        y_data = data.pop('Visits')
        return data, y_data
    # Otherwise, assume test data and return that
    return data


class SMAPE(nn.Module):
    def forward(self, y_hat, y):
        denominator = (torch.abs(y) + torch.abs(y_hat)) / 200.
        diff = torch.abs(y_hat - y) / denominator
        diff[denominator == 0] = 0
        return torch.mean(diff)


def train_model(name):
    train, y_train = load_data(pd.read_csv(ML_TRAIN))
    val, y_val = load_data(pd.read_csv(ML_VALIDATION))

    model = get_model(train.shape[1], 1).cuda()
    # criterion = SMAPE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10)
    best_loss = np.inf

    training_losses, validation_losses = [], []

    for epoch in range(N_EPOCHS):
        # Train on minibatches
        model.train()
        for batch_idx in BatchSampler(RandomSampler(train), BATCH_SIZE, False):
            x, y = train.iloc[batch_idx], y_train.iloc[batch_idx]
            x = Variable(torch.from_numpy(x.values)).cuda()
            y = Variable(torch.from_numpy(y.values)).cuda()

            # Reset gradients
            optimizer.zero_grad()
            # Compute the predictions
            y_hat = model(x)
            # Compute the loss and updatae the weights
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()

        # Compute the training and validation loss
        model.eval()

        training_loss = compute_loss(train, y_train, criterion, model)
        training_losses.append(training_loss)

        validation_loss = compute_loss(val, y_val, criterion, model)
        scheduler.step(validation_loss)
        validation_losses.append(validation_loss)

        # Save checkpoint model and best model
        is_best = validation_loss < best_loss
        best_loss = min(validation_loss, best_loss)
        save_checkpoint(name, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        print('[%2d/%d] Training loss: %.4f - Validation loss: %.4f' %
              (epoch + 1, N_EPOCHS, training_loss, validation_loss))

    print('Best loss achieved: %.4f' % best_loss)
    plt.plot(list(range(len(training_losses))), training_losses)
    plt.plot(list(range(len(validation_losses))), validation_losses)
    plt.show()


def make_lag_test_set(lag_days=LAG_DAYS):
    data = pd.read_csv(TRAIN_DATA)
    columns = ['Page'] + get_date_columns(data)[-lag_days:]
    columns += ['ts_median', 'ts_mean', 'ts_std']
    data[columns].to_csv(lag_test_set_fname(lag_days), index=False)


def make_prediction(model_checkpoint):
    test_data = pd.read_csv(TEST_DATA)
    test_ids = dict(zip(test_data['Page'], test_data['Id']))
    # Extract date and page to own columns
    test_data['date'] = test_data['Page'].apply(lambda a: a[-10:])
    test_data['Page'] = test_data['Page'].apply(lambda a: a[:-11])
    # Add dummy variable so we can pivot
    test_data['Visits'] = 0
    # Put the test data into a the table format as in the training data
    test_data = test_data.pivot(index='Page', columns='date', values='Visits')
    test_dates = test_data.columns

    assert exists(lag_test_set_fname(LAG_DAYS)), \
        'Lag test set file does not exit. Please run `make_lag_test_set`.'
    lag_test_set = pd.read_csv(lag_test_set_fname(LAG_DAYS))
    data = lag_test_set.join(test_data, on='Page')

    # Free up some memory
    del test_data
    del lag_test_set

    model = None
    date_columns = get_date_columns(data)
    for date in test_dates:
        # Find the lag columns to be used in predicting this particular date
        date_idx = date_columns.index(date)
        lag_columns = date_columns[date_idx - LAG_DAYS:date_idx]
        lag_column_names = get_lag_columns(LAG_DAYS)

        # Prepare the dataframe for prediction
        tmp = data[['Page'] + lag_columns]
        # Impute missing values to the best of our ability
        rolling_median = tmp[lag_columns].rolling(window=5).median()
        tmp.fillna(rolling_median, inplace=True)
        tmp.fillna(0, inplace=True)

        tmp['date'] = date
        tmp = tmp.rename(columns=dict(zip(lag_columns, lag_column_names)))
        tmp = load_data(tmp)

        # Load up the model if running first time, we need to do this here
        # since we don't know the number of features in advance
        if model is None:
            model = get_model(tmp.shape[1], 1).cuda()
            checkpoint = torch.load(model_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])

        # Make predictions
        predictions = np.zeros(data.shape[0])
        for batch_idx in BatchSampler(SequentialSampler(data), BATCH_SIZE, False):
            x = Variable(torch.from_numpy(tmp.iloc[batch_idx].values), volatile=True).cuda()
            predictions[batch_idx] = model(x).cpu().data.numpy().reshape(-1)
        data[date] = predictions

    predictions = data[['Page'] + list(test_dates)]
    # Rescale the data back to regular proportions
    # data_info = get_info_file()
    # predictions[test_dates] *= data_info['std']
    # predictions[test_dates] += data_info['mean']

    flattened = pd.melt(predictions, id_vars='Page', var_name='date',
                        value_name='Visits')

    # Since SMAPE prefers under-estimates, floor the predictions and make sure
    # no prediction is below 0. Keep as floats, since those seem to do better
    flattened['Visits'] = flattened['Visits'].clip(lower=0)
    print('%d predictions were negative' % (flattened['Visits'] < 0).sum())

    # Merge the `Page` back into the original names in the key set
    flattened['Page'] = flattened['Page'].str.cat(flattened['date'], sep='_')
    flattened['Id'] = flattened['Page'].apply(test_ids.get)

    # Save the predictions
    model_name = split(dirname(model_checkpoint))[-1]
    save_predictions(flattened, join(PREDICTIONS_DIR, '%s.csv' % model_name))


if __name__ == '__main__':
    fire.Fire()
