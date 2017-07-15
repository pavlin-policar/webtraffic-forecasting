import numpy as np


def smape(truth, predictions):
    # type: (np.ndarray, np.ndarray) -> float
    assert truth.shape == predictions.shape, \
        'Ground truth and predictions must have the same shape'
    assert truth.ndim == 1, 'SMAPE expects 1d arrays on input'

    result = np.sum(np.abs(truth - predictions) /
                    ((np.abs(truth) + np.abs(predictions)) / 2))
    result /= truth.shape[0]

    return result
