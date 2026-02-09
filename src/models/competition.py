from __future__ import annotations

import pandas as pd


def summarize_competitors(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ['name', 'beds', 'specialties'] if c in df.columns]
    return df[cols].copy()
