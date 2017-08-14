import random

import numpy as np
import pandas as pd

from data_provider import TRAIN_DATA, get_date_columns





all_data = pd.read_csv(TRAIN_DATA)


for mb in epoch(all_data, batch_size=256):
    print(mb)
    break
