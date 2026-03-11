"""Forecasting from fitted GAS models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ForecastResult:
    """Forecast from a GAS model.

    Attributes
    ----------
    mean_path:
        Point forecast (mean of filter path) for each horizon.
    quantiles:
        Dict mapping quantile level -> array of length h.
    h:
        Forecast horizon.
    """

    mean_path: dict[str, NDArray[np.float64]]
    quantiles: dict[float, dict[str, NDArray[np.float64]]]
    h: int

    def to_dataframe(self, param: str | None = None) -> pd.DataFrame:
        """Return forecast as a DataFrame.

        Parameters
        ----------
        param:
            Which time-varying parameter to return. Uses the first if None.
        """
        if param is None:
            param = next(iter(self.mean_path))

        data: dict[str, NDArray] = {"mean": self.mean_path[param]}
        for q, paths in self.quantiles.items():
            data[f"q{int(q*100)}"] = paths[param]
        return pd.DataFrame(data)

    def plot(self, param: str | None = None, ax=None):
        """Fan chart of the forecast."""
        import matplotlib.pyplot as plt
        from .plotting import plot_forecast_fan

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        plot_forecast_fan(self, param=param, ax=ax)
        return ax


def gas_forecast(
    result,
    h: int = 6,
    method: str = "mean_path",
    quantiles: list[float] | None = None,
    n_sim: int = 1000,
    rng: np.random.Generator | None = None,
) -> ForecastResult:
    """Produce h-step-ahead forecasts from a fitted GASResult.

    Parameters
    ----------
    result:
        A fitted GASResult.
    h:
        Forecast horizon (periods ahead).
    method:
        ``'mean_path'`` propagates the filter mean;
        ``'simulate'`` draws simulation paths and computes quantiles.
    quantiles:
        Quantile levels for prediction intervals.
    n_sim:
        Number of simulation paths (``method='simulate'`` only).
    rng:
        Random number generator.

    Returns
    -------
    ForecastResult
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    if rng is None:
        rng = np.random.default_rng(42)

    model = result.model
    dist = result.distribution
    time_varying = model.time_varying
    gas_params = result.params
    static_params = {
        k: v for k, v in result.params.items() if k in model._build_static_param_names()
    }

    # Last filter state (link scale)
    last_f: dict[str, float] = {}
    for name in time_varying:
        last_val = result.filter_path[name].iloc[-1]
        last_f[name] = float(dist.link(name, last_val))

    # Mean-path forecast: propagate without new observations
    # f_{t+h} = omega + phi * f_{t+h-1} (no score update since E[score]=0)
    mean_paths: dict[str, list[float]] = {name: [] for name in time_varying}
    f_current = dict(last_f)

    for _ in range(h):
        f_next: dict[str, float] = {}
        for name in time_varying:
            omega = gas_params[f"omega_{name}"]
            phi_vals = [gas_params[f"phi_{name}_{j+1}"] for j in range(model.q)]
            val = omega + sum(p * f_current[name] for p in phi_vals)
            f_next[name] = val
            mean_paths[name].append(float(dist.unlink(name, val)))
        f_current = f_next

    mean_path_arrays = {name: np.array(vals) for name, vals in mean_paths.items()}

    # Simulation paths for quantiles
    if method == "simulate":
        sim_paths: dict[str, list[list[float]]] = {name: [] for name in time_varying}

        for _ in range(n_sim):
            f_sim = dict(last_f)
            sim_step: dict[str, list[float]] = {name: [] for name in time_varying}

            for step in range(h):
                # Sample one observation from current state
                params_nat = {
                    name: float(dist.unlink(name, f_sim[name]))
                    for name in time_varying
                }
                params_nat.update(static_params)

                # Draw y from distribution
                y_sim = _draw_sample(dist, params_nat, rng)
                y_arr = np.array([y_sim])

                # Compute scaled score
                ss = dist.scaled_score(
                    y_arr, params_nat, scaling=model.scaling
                )

                # Update filter
                f_next = {}
                for name in time_varying:
                    omega = gas_params[f"omega_{name}"]
                    alpha_vals = [gas_params[f"alpha_{name}_{i+1}"] for i in range(model.p)]
                    phi_vals = [gas_params[f"phi_{name}_{j+1}"] for j in range(model.q)]
                    score_val = float(np.squeeze(ss[name]))
                    val = omega + alpha_vals[0] * score_val + phi_vals[0] * f_sim[name]
                    f_next[name] = val
                    sim_step[name].append(float(dist.unlink(name, val)))

                f_sim = f_next

            for name in time_varying:
                sim_paths[name].append(sim_step[name])

        # Compute quantile arrays
        q_results: dict[float, dict[str, NDArray]] = {}
        for q in quantiles:
            q_results[q] = {}
            for name in time_varying:
                paths_arr = np.array(sim_paths[name])  # (n_sim, h)
                q_results[q][name] = np.quantile(paths_arr, q, axis=0)
    else:
        # Mean-path only: quantile = mean for all
        q_results = {q: {name: mean_path_arrays[name] for name in time_varying} for q in quantiles}

    return ForecastResult(mean_path=mean_path_arrays, quantiles=q_results, h=h)


def _draw_sample(
    dist: "GASDistribution",
    params: dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Draw a single sample from the distribution at current params."""
    from .distributions import PoissonGAS, GammaGAS, NegBinGAS, LogNormalGAS, BetaGAS, ZIPGAS

    if isinstance(dist, PoissonGAS):
        return float(rng.poisson(params["mean"]))
    elif isinstance(dist, GammaGAS):
        shape = params.get("shape", 1.0)
        return float(rng.gamma(shape=shape, scale=params["mean"] / shape))
    elif isinstance(dist, NegBinGAS):
        mu = params["mean"]
        r = params.get("dispersion", 1.0)
        p = r / (r + mu)
        return float(rng.negative_binomial(r, p))
    elif isinstance(dist, LogNormalGAS):
        sigma = params.get("logsigma", 0.5)
        return float(rng.lognormal(mean=params["logmean"], sigma=sigma))
    elif isinstance(dist, BetaGAS):
        mu = params["mean"]
        phi = params.get("precision", 10.0)
        return float(rng.beta(mu * phi, (1.0 - mu) * phi))
    elif isinstance(dist, ZIPGAS):
        pi = params.get("zeroprob", 0.1)
        if rng.random() < pi:
            return 0.0
        return float(rng.poisson(params["mean"]))
    else:
        raise NotImplementedError(f"Sampling not implemented for {type(dist).__name__}")
