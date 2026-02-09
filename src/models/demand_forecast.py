from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def fit_arima(series: pd.Series, order: tuple[int, int, int] = (1, 1, 1)):
    model = sm.tsa.ARIMA(series, order=order)
    return model.fit()
