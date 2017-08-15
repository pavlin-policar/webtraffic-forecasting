import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import MSELoss

from data_provider import TRAIN_DATA, epoch

N_EPOCHS = 1


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(39, 1)

criterion = MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


data = pd.read_csv(TRAIN_DATA)
# Normalize the data
cols = data.columns.difference(['Page'])
data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()


for i in range(N_EPOCHS):
    for minibatch in epoch(data, batch_size=256):

        targets = minibatch.pop('Visits')
        minibatch = minibatch.drop(['Page', 'date'], axis=1)

        data = pd.get_dummies(minibatch)
        data = data.values.astype(np.float32)
        targets = targets.values.astype(np.float32)

        data = Variable(torch.from_numpy(data))
        targets = Variable(torch.from_numpy(targets))

        # Reset gradients
        optimizer.zero_grad()

        # Compute the predictions
        outputs = model(data)

        # Compute the loss and updatae the weights
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Loss: %.4f' % (i + 1, N_EPOCHS, loss.data[0]))
