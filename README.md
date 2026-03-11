# insurance-gas

GAS (Generalised Autoregressive Score) models for dynamic insurance pricing.

## The problem

UK pricing actuaries re-fit GLMs annually or quarterly. Between model updates, the world moves — frequency rises after a cold winter, severity drifts with parts inflation, a new distribution channel shifts the portfolio mix. Static models miss these changes until the next update cycle.

GAS models solve this by updating a time-varying parameter at each observation using the score (gradient) of the log-likelihood as a forcing variable. The update is proportional to how surprising the observation is, given what we currently believe. This is economical: the likelihood is still a simple product of densities, so MLE is straightforward L-BFGS-B.

The GAS recursion for a time-varying parameter f (on the link scale) is:

```
f_{t+1} = omega + alpha * S(f_t) * nabla(y_t, f_t) + phi * f_t
```

where `nabla` is the score, `S` is the scaling matrix (inverse Fisher information), `alpha` controls how fast the filter reacts, and `phi` controls how persistent the current level is.

This is the approach of Creal, Koopman & Lucas (2013) and Harvey (2013). In R, `gasmodel` implements 36 distributions. This library provides the actuarially relevant subset in Python, with exposure offsets and pricing team output formats.

## Installation

```bash
pip install insurance-gas
```

## Quick start

```python
from insurance_gas import GASModel
from insurance_gas.datasets import load_motor_frequency

data = load_motor_frequency(T=60, trend_break=True)

model = GASModel(
    distribution="poisson",  # claim frequency
    p=1, q=1,                # GAS(1,1)
    scaling="fisher_inv",    # optimal for this distribution class
)
result = model.fit(data.y, exposure=data.exposure)

print(result.summary())

# Time-varying rate (natural scale, claims per unit exposure)
result.filter_path.plot()

# Trend index for actuarial sign-off: base period = 100
result.trend_index.plot()

# Relativities vs the time-average
result.relativities(base="mean")
```

## Distributions

| String key   | Class            | Use case                          | Link   |
|--------------|------------------|-----------------------------------|--------|
| `poisson`    | `PoissonGAS`     | Claim frequency                   | log    |
| `gamma`      | `GammaGAS`       | Claim severity                    | log    |
| `negbin`     | `NegBinGAS`      | Overdispersed frequency           | log    |
| `lognormal`  | `LogNormalGAS`   | Severity (heavy right tail)       | identity on log-mean |
| `beta`       | `BetaGAS`        | Loss ratios on (0,1)              | logit  |
| `zip`        | `ZIPGAS`         | Zero-inflated frequency           | log + logit |

## Result object

`model.fit()` returns a `GASResult` with:

```python
result.filter_path        # pd.DataFrame: time-varying params at each t
result.trend_index        # same, re-scaled to base=100
result.relativities()     # ratios vs mean or first period
result.log_likelihood     # total log-likelihood at MLE
result.aic, result.bic
result.params             # fitted parameter dict
result.std_errors         # from numerical Hessian
result.score_residuals    # standardised: should be ~iid(0,1)
result.summary()          # formatted coefficient table
result.forecast(h=6)      # h-step-ahead forecast
result.diagnostics()      # PIT test, Ljung-Box, Dawid-Sebastiani
result.bootstrap_ci()     # parametric bootstrap CI on filter path
```

## Forecasting

```python
fc = result.forecast(
    h=12,
    method="simulate",           # 'mean_path' or 'simulate'
    quantiles=[0.1, 0.5, 0.9],
    n_sim=1000,
)
fc.plot()
df = fc.to_dataframe()           # h x (mean + quantile columns)
```

## Panel data (multiple rating cells)

```python
from insurance_gas import GASPanel

panel = GASPanel("poisson")
panel_result = panel.fit(
    data,                 # DataFrame with period, cell_id, claims, exposure
    y_col="claims",
    period_col="period",
    cell_col="vehicle_class",
    exposure_col="exposure",
)

panel_result.trend_summary()     # wide DataFrame: periods x cells
panel_result.filter_paths        # dict[cell_id, DataFrame]
```

## Diagnostics

```python
diag = result.diagnostics()
print(diag.summary())

# PIT histogram (should be uniform for well-specified model)
diag.pit_histogram()

# ACF of score residuals (no remaining autocorrelation → model adequate)
diag.score_residuals_acf()
```

## Scaling options

The scaling matrix S in the GAS recursion matters for how quickly the filter adapts and how robust it is to outliers:

- `unit` — no scaling (raw score). Simplest; can over-react to outliers in heavy-tailed distributions.
- `fisher_inv` — multiply by inverse Fisher information. Optimal for distributions close to the Gaussian. Default.
- `fisher_inv_sqrt` — more robust; down-weights large scores more aggressively.

## Design decisions

**Why observation-driven rather than parameter-driven?** State-space models (Kalman filter) require integrating out the latent state. GAS models are observation-driven: the filter is a deterministic function of past data, so the likelihood is a product of closed-form densities. MLE is a standard L-BFGS-B optimisation — no MCMC, no expectation-maximisation.

**Why not use PyFlux?** PyFlux (2017) implements GAS but is unmaintained and incompatible with modern Python. This library is built against Python 3.10+ and modern NumPy/SciPy.

**Why expose relativities rather than filter parameters?** Actuaries work in relativities. Exposing `f_t = log(mu_t)` would require them to exponentiate and normalise. The `trend_index` and `relativities()` output are in the same format as development factor analyses.

**On standard errors**: they come from the numerical Hessian of the log-likelihood. This is consistent but can be unreliable in small samples (< 30 observations). Use `bootstrap_ci()` for credible uncertainty quantification in small portfolios.

## References

- Creal, D., Koopman, S.J. and Lucas, A. (2013). 'Generalized Autoregressive Score Models with Applications.' *Journal of Applied Econometrics*, 28(5):777–795.
- Harvey, A.C. (2013). *Dynamic Models for Volatility and Heavy Tails*. Cambridge University Press.
- Holy, V. and Zouhar, J. (2024). 'gasmodel: GAS models in R.' arXiv:2405.05073.

## License

MIT
