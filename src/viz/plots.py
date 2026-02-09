from __future__ import annotations

import pandas as pd
import plotly.express as px


def revenue_curve(df: pd.DataFrame, x: str = 'year', y: str = 'revenue_cop'):
    return px.line(df, x=x, y=y, title='Revenue (COP)')
