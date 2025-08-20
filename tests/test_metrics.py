import numpy as np

from m5_forecasting.utils.metrics import mase, wmape


def test_wmape_basic():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = wmape(y_true, y_pred)
    assert round(result, 4) == round((10 + 10 + 10) / 600, 4)


def test_mase_basic():
    y_true = np.array([200, 210, 190, 205])
    y_pred = np.array([198, 215, 195, 200])
    y_insample = np.array([150, 160, 170, 180, 190, 200, 210, 220, 230])
    result = mase(y_true, y_pred, y_insample, seasonality=1)
    assert result > 0
