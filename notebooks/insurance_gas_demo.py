# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-gas: GAS Models for Dynamic Insurance Pricing
# MAGIC
# MAGIC This notebook demonstrates the full workflow for fitting GAS (Generalised
# MAGIC Autoregressive Score) models to insurance time series using the
# MAGIC `insurance-gas` library.
# MAGIC
# MAGIC **Use cases covered:**
# MAGIC 1. Motor claim frequency monitoring (Poisson GAS)
# MAGIC 2. Severity trend estimation (Gamma GAS)
# MAGIC 3. Loss ratio tracking (Beta GAS)
# MAGIC 4. Panel fitting across rating cells
# MAGIC 5. Forecasting and confidence intervals

# COMMAND ----------

# MAGIC %pip install insurance-gas

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from insurance_gas import GASModel, GASPanel
from insurance_gas.datasets import (
    load_motor_frequency,
    load_severity_trend,
    load_loss_ratio,
)

print(f"insurance-gas loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Motor Claim Frequency — Poisson GAS
# MAGIC
# MAGIC The dataset simulates 5 years of monthly motor claim counts.
# MAGIC A step change in frequency at month 30 represents a portfolio change
# MAGIC (new direct channel with younger drivers).

# COMMAND ----------

data = load_motor_frequency(T=60, seed=42, trend_break=True)
print(data.description)
print(f"\nClaims range: {data.y.min():.0f} – {data.y.max():.0f}")
print(f"Exposure range: {data.exposure.min():.0f} – {data.exposure.max():.0f}")

# COMMAND ----------

model = GASModel(
    distribution="poisson",
    p=1,
    q=1,
    scaling="fisher_inv",
)
result = model.fit(data.y, exposure=data.exposure)
print(result.summary())

# COMMAND ----------

# Filter path vs ground truth
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Claims per unit exposure (observed)
obs_rate = data.y / data.exposure
axes[0].bar(range(60), obs_rate, alpha=0.5, color="grey", label="Observed rate")
axes[0].plot(data.filter_truth["mean"], "g--", linewidth=2, label="True rate")
axes[0].plot(result.filter_path["mean"].values, "b-", linewidth=2, label="GAS filter")
axes[0].axvline(30, color="red", linestyle=":", label="Trend break")
axes[0].set_title("Poisson GAS: Claim Frequency Filter")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Claims per unit exposure")
axes[0].legend()

# Trend index
ti = result.trend_index
axes[1].plot(ti["mean"].values, "b-", linewidth=2)
axes[1].axhline(100, color="grey", linestyle="--", linewidth=0.8)
axes[1].axvline(30, color="red", linestyle=":", label="Trend break")
axes[1].set_title("Trend Index (base = first period = 100)")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Index")
axes[1].legend()

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Severity Trend — Gamma GAS
# MAGIC
# MAGIC Quarterly average claim cost with 5% per-quarter inflation.
# MAGIC The Gamma GAS adapts the mean severity smoothly while the Fisher
# MAGIC information scaling controls how fast it reacts to individual outliers.

# COMMAND ----------

sev_data = load_severity_trend(T=40, seed=1, inflation_rate=0.05)
print(sev_data.description)

sev_model = GASModel("gamma", p=1, q=1)
sev_result = sev_model.fit(sev_data.y)
print(sev_result.summary())

# COMMAND ----------

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].scatter(range(40), sev_data.y, alpha=0.6, s=30, color="grey", label="Observed")
axes[0].plot(sev_data.filter_truth["mean"], "g--", linewidth=2, label="True mean")
axes[0].plot(sev_result.filter_path["mean"].values, "b-", linewidth=2, label="GAS filter")
axes[0].set_title("Gamma GAS: Severity Trend")
axes[0].set_xlabel("Quarter")
axes[0].set_ylabel("Claim severity")
axes[0].legend()

# Score residuals ACF
from insurance_gas.diagnostics import _compute_acf
sr = sev_result.score_residuals["mean"].values
acf = _compute_acf(sr, nlags=15)
lags = np.arange(len(acf))
axes[1].bar(lags[1:], acf[1:], color="steelblue", alpha=0.8)
axes[1].axhline(1.96 / np.sqrt(40), color="red", linestyle="--", linewidth=0.8)
axes[1].axhline(-1.96 / np.sqrt(40), color="red", linestyle="--", linewidth=0.8)
axes[1].set_title("Score Residuals ACF (should be within red lines)")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("ACF")

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Loss Ratio — Beta GAS
# MAGIC
# MAGIC Monthly loss ratios (claims / premium) on the unit interval.
# MAGIC BetaGAS uses a logit link so the filtered mean always stays in (0, 1).

# COMMAND ----------

lr_data = load_loss_ratio(T=48, seed=3)
print(lr_data.description)

lr_model = GASModel("beta", p=1, q=1)
lr_result = lr_model.fit(lr_data.y)
print(lr_result.summary())

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(range(48), lr_data.y, alpha=0.5, s=25, color="grey", label="Observed LR")
ax.plot(lr_data.filter_truth["mean"], "g--", linewidth=2, label="True mean LR")
ax.plot(lr_result.filter_path["mean"].values, "b-", linewidth=2, label="Beta GAS filter")
ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="Break-even")
ax.set_title("Beta GAS: Loss Ratio Tracking")
ax.set_xlabel("Month")
ax.set_ylabel("Loss ratio")
ax.legend()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Forecasting
# MAGIC
# MAGIC Mean-path and simulated fan-chart forecasts for the frequency model.
# MAGIC Under GAS, the h-step-ahead forecast converges to the unconditional mean
# MAGIC at rate phi^h.

# COMMAND ----------

fc = result.forecast(
    h=12,
    method="simulate",
    quantiles=[0.1, 0.25, 0.75, 0.9],
    n_sim=500,
    rng=np.random.default_rng(42),
)

fig, ax = plt.subplots(figsize=(12, 5))

# Historical
ax.plot(range(60), data.y / data.exposure, color="grey", alpha=0.6, label="History")

# Forecast
from insurance_gas.plotting import plot_forecast_fan
plot_forecast_fan(fc, history=data.y / data.exposure, ax=ax)
ax.set_title("Poisson GAS: 12-month Frequency Forecast")
ax.set_xlabel("Month")
ax.set_ylabel("Claims per unit exposure")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Panel Fitting — Multiple Rating Cells
# MAGIC
# MAGIC Fit the same Poisson GAS model independently to each of four vehicle
# MAGIC classes. The panel result gives aligned filter paths and trend indices
# MAGIC for actuarial comparison.

# COMMAND ----------

import pandas as pd

rng = np.random.default_rng(2025)
rows = []
cells = ["Hatchback", "SUV", "Van", "Motorcycle"]
base_rates = [0.08, 0.06, 0.12, 0.18]

for cell, rate in zip(cells, base_rates):
    exposure = rng.uniform(800, 1500, 48)
    # Introduce a different trend per cell
    trend = np.exp(np.linspace(0, 0.3 * rng.uniform(0.5, 2.0), 48))
    claims = rng.poisson(rate * exposure * trend).astype(float)
    for t in range(48):
        rows.append({
            "period": t,
            "vehicle_class": cell,
            "claims": claims[t],
            "exposure": exposure[t],
        })

panel_df = pd.DataFrame(rows)
print(panel_df.groupby("vehicle_class")[["claims", "exposure"]].describe().round(1))

# COMMAND ----------

panel = GASPanel("poisson", p=1, q=1)
panel_result = panel.fit(
    panel_df,
    y_col="claims",
    period_col="period",
    cell_col="vehicle_class",
    exposure_col="exposure",
    verbose=True,
)

trend_wide = panel_result.trend_summary()
print("\nTrend indices (base=100):")
print(trend_wide.iloc[::12].round(1))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 5))
for cell in cells:
    if cell in trend_wide.columns:
        ax.plot(trend_wide[cell].values, label=cell, linewidth=2)

ax.axhline(100, color="grey", linestyle="--", linewidth=0.8)
ax.set_title("GAS Trend Indices by Vehicle Class")
ax.set_xlabel("Month")
ax.set_ylabel("Trend index (base = 100)")
ax.legend()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Diagnostics
# MAGIC
# MAGIC PIT uniformity test and score residual ACF for the frequency model.

# COMMAND ----------

diag = result.diagnostics()
print(diag.summary())

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

diag.pit_histogram(ax=axes[0])

from insurance_gas.plotting import plot_acf
plot_acf(diag.score_residuals_acf, ax=axes[1])

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Bootstrap Confidence Intervals

# COMMAND ----------

ci = result.bootstrap_ci(n_boot=200, confidence=0.95, rng=np.random.default_rng(42))

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(
    range(60),
    ci.filter_lower["mean"].values,
    ci.filter_upper["mean"].values,
    alpha=0.3,
    color="steelblue",
    label="95% bootstrap CI",
)
ax.plot(result.filter_path["mean"].values, "b-", linewidth=2, label="MLE filter path")
ax.plot(data.filter_truth["mean"], "g--", linewidth=1.5, label="True mean")
ax.set_title("Poisson GAS Filter Path with Bootstrap 95% CI")
ax.set_xlabel("Month")
ax.set_ylabel("Claim rate")
ax.legend()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `insurance-gas` library provides score-driven models for dynamic
# MAGIC insurance parameters. Key advantages over static GLM re-fitting:
# MAGIC
# MAGIC - Closed-form likelihood — no numerical integration
# MAGIC - Single-step filter update — computationally efficient
# MAGIC - Principled uncertainty quantification via bootstrap
# MAGIC - Actuarially interpretable output (trend index, relativities)
# MAGIC - Covers all major insurance distributions (Poisson, Gamma, NB, LN, Beta, ZIP)
