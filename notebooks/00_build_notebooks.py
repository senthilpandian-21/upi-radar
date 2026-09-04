"""Builds the five tutorial notebooks (run once, outputs live in notebooks/).

    python notebooks/00_build_notebooks.py

Each notebook is executable end-to-end against the project code.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def cell(code: str, kind: str = "code"):
    """Returns a proper nbformat v4 cell object."""
    import nbformat as _nbf

    if kind == "markdown":
        return _nbf.v4.new_markdown_cell(code.strip("\n"))
    return _nbf.v4.new_code_cell(code.strip("\n"))


NOTEBOOKS = {
    "01_data_exploration.ipynb": [
        cell("# 📊 01 — Data Exploration\n\nEDA on the synthetic dataset: "
             "volume curves, failure rates, calendar effects, outage "
             "distribution per bank."),
        cell("""import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/synthetic/generated_data.csv',
                 parse_dates=['datetime'])
df.info()
print(f"Rows: {len(df):,} | Outage rows: {int(df['is_outage'].sum()):,} "
      f"({df['is_outage'].mean():.2%})")"""),
        cell("""hourly = df.groupby('hour_of_day').agg(
    volume=('total_volume', 'mean'),
    failure_rate=('failure_rate', 'mean')).reset_index()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(hourly.hour_of_day, hourly.volume / 1e6)
ax[0].set_title('Avg hourly volume (millions)')
ax[1].plot(hourly.hour_of_day, hourly.failure_rate)
ax[1].set_title('Avg hourly failure rate')
plt.tight_layout(); plt.show()"""),
        cell("""# calendar effects
for flag in ['is_salary_day', 'is_fy_end', 'is_month_end', 'is_peak_hour']:
    sub = df.groupby(flag)['failure_rate'].mean()
    print(f"{flag:14s} normal={sub.get(0, 0):.5f}  flagged={sub.get(1, 0):.5f}")

print("\\nOutage rows per bank:")
print(df[df.is_outage == 1].groupby('bank').size().sort_values(ascending=False))"""),
    ],
    "02_feature_engineering.ipynb": [
        cell("# 🛠 02 — Feature Engineering\n\nTime flags, rolling averages, "
             "health scores and scaling — exactly what the LSTM consumes."),
        cell("""import pandas as pd
from data_pipeline.feature_engineering import FeatureEngineer

df = pd.read_csv('data/synthetic/generated_data.csv',
                 parse_dates=['datetime'])
eng = FeatureEngineer()
processed = eng.full_pipeline(df.copy(), fit=False, add_rolling=False)
processed[['datetime', 'is_salary_day', 'is_fy_end', 'is_peak_hour']].head()"""),
        cell("""# bank health score formula (weighted 15min / 1hr / historical)
score = eng.calculate_bank_health_score(
    td_15min=0.05, td_1hr=0.03, td_historical=0.009)
print(f"Bank health for td(15m)=5%, td(1h)=3%, hist=0.9% → {score}")"""),
        cell("""# 80/10/10 chronological split for training
paths = eng.save_split(df, output_dir='data/processed')
for name, path in paths.items():
    print(name, '→', path)"""),
    ],
    "03_lstm_training.ipynb": [
        cell("# 🧠 03 — LSTM Training\n\n`python -m models.lstm_forecaster.train` "
             "equivalents, step by step."),
        cell("""import pandas as pd
from models.lstm_forecaster.model import (FEATURE_COLUMNS, SEQUENCE_LENGTH,
                                          build_lstm_model, create_sequences)
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/synthetic/generated_data.csv', parse_dates=['datetime'])
df = df.sort_values(['bank', 'datetime'])

scaler = StandardScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS].astype('float64'))

Xs, ys = [], []
for bank in df.bank.unique():
    X, y = create_sequences(df[df.bank == bank].reset_index(drop=True),
                            FEATURE_COLUMNS, 'is_outage', SEQUENCE_LENGTH)
    Xs.append(X); ys.append(y)
import numpy as np
X, y = np.vstack(Xs), np.concatenate(ys)
print(X.shape, y.shape, f"outage share {y.mean():.2%}")"""),
        cell("""model = build_lstm_model(input_shape=(SEQUENCE_LENGTH,
                                                    len(FEATURE_COLUMNS)))
model.summary()"""),
        cell("""# Full training (uses early stopping on val_auc + class weights):
# from models.lstm_forecaster.train import train
# train(data_path='data/synthetic/generated_data.csv', epochs=100)"""),
    ],
    "04_prophet_training.ipynb": [
        cell("# 📅 04 — Prophet Training\n\nDaily seasonal risk model with "
             "salary/FY-end/festival regressors."),
        cell("""import pandas as pd
from models.prophet_forecaster.model import ProphetOutageModel

df = pd.read_csv('data/synthetic/generated_data.csv', parse_dates=['datetime'])
daily = ProphetOutageModel.prepare_daily(df)
print(daily.head())
print('banks:', daily.bank.unique(), '| days:', daily.ds.nunique())"""),
        cell("""# NOTE: requires `pip install prophet` (not installed by default)
# holder = ProphetOutageModel().train(df)     # per-bank + SYSTEM models
# holder.save()
# probe = holder.probability_for_next_day('SYSTEM')
# print('next-day P(outage) =', probe)"""),
    ],
    "05_model_evaluation.ipynb": [
        cell("# 📈 05 — Model Evaluation\n\nPrecision / Recall / AUC and the "
             "confusion story, plus false-positive policy (B1–B5 thresholds)."),
        cell("""# after training the LSTM:
# from models.lstm_forecaster.evaluate import evaluate
# report = evaluate('data/synthetic/generated_data.csv')
# print(json.dumps(report, indent=2))   # accuracy/precision/recall/AUC"""),
        cell("""from models.ensemble import EnsemblePredictor
pred = EnsemblePredictor().get_mock_prediction()
print({k: pred[k] for k in ('final_probability', 'lstm_probability',
                            'prophet_probability', 'anomaly_probability',
                            'risk_level')})"""),
        cell("""# false-positive guard: risk flags never route a healthy bank
from policy.bank_rules import BankRulesEngine
healthy = {b: {'score': 95.0} for b in
           ['SBI', 'HDFC', 'ICICI', 'AXIS', 'PNB', 'BOB', 'KOTAK', 'YES']}
d = BankRulesEngine().evaluate({'bank': 'HDFC', 'amount': 1000,
                                'method': 'upi'}, healthy, 0.1)
print('healthy tx →', d.action, '|', d.reason)"""),
    ],
}


def main() -> None:
    for filename, cells in NOTEBOOKS.items():
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}
        path = HERE / filename
        path.write_text(nbf.writes(notebook))
        print(f"✓ {path.name} ({sum(1 for c in cells if c['cell_type']=='code')} cells)")


if __name__ == "__main__":
    main()
