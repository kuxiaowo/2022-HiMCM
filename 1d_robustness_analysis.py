# 1d_robustness_analysis.py
# Robustness analysis for question 1d model selection.
#
# The default configuration follows the requested setting:
#   - 100 Monte Carlo perturbations
#   - primary sigma = 0.5 ppm
#   - optional sigmas = 0.2 and 1.0 ppm are listed below but disabled by default
#   - annual quadratic/cubic/quartic/exponential models
#   - SARIMA(1,1,1)(2,1,2,12), annualized for comparison
#
# To smoke-test quickly without editing this file, set environment variables:
#   $env:ROBUSTNESS_N_SIMULATIONS='2'
#   $env:ROBUSTNESS_SIGMAS='0.5'
#   $env:ROBUSTNESS_INCLUDE_SARIMA='0'
#   $env:ROBUSTNESS_MAX_WORKERS='2'

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import warnings

# Keep each worker from starting its own full BLAS/OpenMP thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from himcm_data import get_co2_data, get_monthly_co2_data

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None


# ============================================================
# 0. Settings
# ============================================================

OUTPUT_DIR = Path("question_1d_outputs") / "robustness_analysis"

RANDOM_SEED = 20260708
N_SIMULATIONS = 100
PRIMARY_SIGMA = 0.5
EXTRA_SIGMAS = [0.2, 1.0]
INCLUDE_EXTRA_SIGMAS = False
INCLUDE_SARIMA = True
MAX_WORKERS = 24

TEST_YEARS_COUNT = 10

SARIMA_SPEC = {
    "name": "SARIMA(1,1,1)(2,1,2,12)",
    "key": "sarima_1_1_1_2_1_2_12",
    "order": (1, 1, 1),
    "seasonal_order": (2, 1, 2, 12),
    "trend": "c",
}
SARIMA_MAXITER = 80

ANNUAL_MODEL_SPECS = [
    {"name": "Quadratic", "key": "quadratic", "type": "polynomial", "degree": 2},
    {"name": "Cubic", "key": "cubic", "type": "polynomial", "degree": 3},
    {"name": "Quartic", "key": "quartic", "type": "polynomial", "degree": 4},
    {"name": "Exponential", "key": "exponential", "type": "exponential"},
]


def configured_n_simulations():
    value = os.environ.get("ROBUSTNESS_N_SIMULATIONS")
    if value is None:
        return N_SIMULATIONS
    parsed = int(value)
    if parsed < 1:
        raise ValueError("ROBUSTNESS_N_SIMULATIONS must be positive.")
    return parsed


def configured_sigmas():
    value = os.environ.get("ROBUSTNESS_SIGMAS")
    if value:
        sigmas = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        sigmas = [PRIMARY_SIGMA]
        if INCLUDE_EXTRA_SIGMAS:
            sigmas.extend(EXTRA_SIGMAS)

    if not sigmas:
        raise ValueError("At least one sigma is required.")
    if any(sigma <= 0 for sigma in sigmas):
        raise ValueError("All sigma values must be positive.")
    return sigmas


def configured_include_sarima():
    value = os.environ.get("ROBUSTNESS_INCLUDE_SARIMA")
    if value is None:
        return INCLUDE_SARIMA
    return value.strip().lower() not in {"0", "false", "no", "off"}


def configured_max_workers():
    value = os.environ.get("ROBUSTNESS_MAX_WORKERS")
    if value is None:
        return MAX_WORKERS
    parsed = int(value)
    if parsed < 1:
        raise ValueError("ROBUSTNESS_MAX_WORKERS must be positive.")
    return parsed


# ============================================================
# 1. Helpers
# ============================================================

def reset_output_dir(output_dir):
    script_dir = Path(__file__).resolve().parent
    resolved_output_dir = (script_dir / output_dir).resolve()

    if script_dir not in resolved_output_dir.parents:
        raise RuntimeError(f"Refusing to clean outside script directory: {resolved_output_dir}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    for child in resolved_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def to_builtin(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def evaluate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    sse = float(np.sum(residuals ** 2))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / y_true)) * 100)
    ss_total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - sse / ss_total) if ss_total != 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }


# ============================================================
# 2. Annual model fitting
# ============================================================

def fit_annual_model(spec, years_train, co2_train):
    years_train = np.asarray(years_train, dtype=float)
    co2_train = np.asarray(co2_train, dtype=float)
    base_year = float(years_train[0])
    t_train = years_train - base_year

    if spec["type"] == "polynomial":
        coefficients = np.polyfit(t_train, co2_train, spec["degree"])

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return np.polyval(coefficients, new_t)

        return predict

    if spec["type"] == "exponential":
        log_coefficients = np.polyfit(t_train, np.log(co2_train), 1)

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return np.exp(log_coefficients[1]) * np.exp(log_coefficients[0] * new_t)

        return predict

    raise ValueError(f"Unsupported annual model type: {spec['type']}")


def run_annual_models(years_train, co2_train_perturbed, years_test, co2_test_true):
    rows = []
    for spec in ANNUAL_MODEL_SPECS:
        predict = fit_annual_model(spec, years_train, co2_train_perturbed)
        y_pred = predict(years_test)
        metrics = evaluate_metrics(co2_test_true, y_pred)
        rows.append({
            "Model": spec["name"],
            "Model_Key": spec["key"],
            "Model_Family": "annual_fit",
            **metrics,
        })
    return rows


# ============================================================
# 3. SARIMA fitting
# ============================================================

def build_monthly_training_series(monthly_df, annual_noise_by_year, test_start_year):
    df = monthly_df.copy()
    df = df[df["year"] <= 2021].copy()
    df["Date"] = pd.to_datetime(
        {
            "year": df["year"].astype(int),
            "month": df["month"].astype(int),
            "day": 1,
        }
    )
    df["Perturbation"] = df["year"].map(annual_noise_by_year).fillna(0.0)
    df["Perturbed_CO2"] = df["average"] + df["Perturbation"]
    train_df = df[df["year"] < test_start_year].copy()

    series = pd.Series(
        train_df["Perturbed_CO2"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(train_df["Date"]),
        name="co2",
    )
    series = series.asfreq("MS")
    return series


def fit_sarima_and_forecast(monthly_df, annual_noise_by_year, years_test, co2_test_true):
    if SARIMAX is None:
        return {
            "Model": SARIMA_SPEC["name"],
            "Model_Key": SARIMA_SPEC["key"],
            "Model_Family": "monthly_sarima_annualized",
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "Status": "statsmodels_not_available",
        }

    test_start_year = int(years_test[0])
    train_series = build_monthly_training_series(monthly_df, annual_noise_by_year, test_start_year)
    steps = len(years_test) * 12

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            order=SARIMA_SPEC["order"],
            seasonal_order=SARIMA_SPEC["seasonal_order"],
            trend=SARIMA_SPEC["trend"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=SARIMA_MAXITER)
        forecast = result.get_forecast(steps=steps).predicted_mean

    forecast.index = pd.date_range(
        start=pd.Timestamp(year=test_start_year, month=1, day=1),
        periods=steps,
        freq="MS",
    )
    annual_forecast = forecast.resample("YS").mean()
    annual_forecast.index = annual_forecast.index.year
    y_pred = annual_forecast.loc[years_test].to_numpy(dtype=float)

    metrics = evaluate_metrics(co2_test_true, y_pred)
    return {
        "Model": SARIMA_SPEC["name"],
        "Model_Key": SARIMA_SPEC["key"],
        "Model_Family": "monthly_sarima_annualized",
        **metrics,
        "Status": "ok",
    }


# ============================================================
# 4. Simulation and summaries
# ============================================================

def add_rank_columns(metrics_df):
    ranked = metrics_df.copy()
    ranked["Rank_By_RMSE"] = (
        ranked.groupby(["Sigma", "Simulation"])["RMSE"]
        .rank(method="min", ascending=True)
    )
    ranked["Is_Best_By_RMSE"] = ranked["Rank_By_RMSE"] == 1
    return ranked


def build_summary(metrics_df, n_simulations):
    rows = []
    grouped = metrics_df.groupby(["Sigma", "Model", "Model_Key", "Model_Family"], dropna=False)

    for (sigma, model, model_key, model_family), group in grouped:
        valid = group.dropna(subset=["RMSE"])
        rows.append({
            "Sigma": sigma,
            "Model": model,
            "Model_Key": model_key,
            "Model_Family": model_family,
            "Successful_Simulations": int(len(valid)),
            "RMSE_Mean": float(valid["RMSE"].mean()) if len(valid) else np.nan,
            "RMSE_Std": float(valid["RMSE"].std(ddof=1)) if len(valid) > 1 else 0.0,
            "MAE_Mean": float(valid["MAE"].mean()) if len(valid) else np.nan,
            "MAE_Std": float(valid["MAE"].std(ddof=1)) if len(valid) > 1 else 0.0,
            "MAPE_Mean": float(valid["MAPE"].mean()) if len(valid) else np.nan,
            "MAPE_Std": float(valid["MAPE"].std(ddof=1)) if len(valid) > 1 else 0.0,
            "R2_Mean": float(valid["R2"].mean()) if len(valid) else np.nan,
            "R2_Std": float(valid["R2"].std(ddof=1)) if len(valid) > 1 else 0.0,
            "Best_Count": int(valid["Is_Best_By_RMSE"].sum()) if len(valid) else 0,
            "Best_Proportion": float(valid["Is_Best_By_RMSE"].sum() / n_simulations),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(["Sigma", "RMSE_Mean"]).reset_index(drop=True)


def run_single_simulation_task(task):
    sigma = task["sigma"]
    simulation = task["simulation"]
    years = np.asarray(task["years"], dtype=int)
    years_train = np.asarray(task["years_train"], dtype=int)
    years_test = np.asarray(task["years_test"], dtype=int)
    co2_train_true = np.asarray(task["co2_train_true"], dtype=float)
    co2_test_true = np.asarray(task["co2_test_true"], dtype=float)
    annual_noise = np.asarray(task["annual_noise"], dtype=float)
    include_sarima = bool(task["include_sarima"])
    monthly_df = task["monthly_df"]

    annual_noise_by_year = {
        int(year): float(noise)
        for year, noise in zip(years, annual_noise)
    }
    co2_train_perturbed = co2_train_true + annual_noise[:len(years_train)]

    model_rows = run_annual_models(
        years_train=years_train,
        co2_train_perturbed=co2_train_perturbed,
        years_test=years_test,
        co2_test_true=co2_test_true,
    )

    if include_sarima:
        try:
            model_rows.append(
                fit_sarima_and_forecast(
                    monthly_df=monthly_df,
                    annual_noise_by_year=annual_noise_by_year,
                    years_test=years_test,
                    co2_test_true=co2_test_true,
                )
            )
        except Exception as exc:
            model_rows.append({
                "Model": SARIMA_SPEC["name"],
                "Model_Key": SARIMA_SPEC["key"],
                "Model_Family": "monthly_sarima_annualized",
                "RMSE": np.nan,
                "MAE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
                "Status": f"failed: {type(exc).__name__}: {exc}",
            })

    rows = []
    for row in model_rows:
        rows.append({
            "Sigma": sigma,
            "Simulation": simulation,
            **row,
            "Perturbation_Std_ppm": sigma,
            "Test_Comparison_Target": "original_true_test_data",
        })
    return rows


def run_simulations():
    n_simulations = configured_n_simulations()
    sigmas = configured_sigmas()
    include_sarima = configured_include_sarima()
    max_workers = configured_max_workers()

    if include_sarima and SARIMAX is None:
        raise RuntimeError(
            "statsmodels is required when INCLUDE_SARIMA is True. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    years, co2 = get_co2_data()
    test_start_index = len(years) - TEST_YEARS_COUNT
    years_train = years[:test_start_index]
    years_test = years[test_start_index:]
    co2_train_true = co2[:test_start_index]
    co2_test_true = co2[test_start_index:]

    monthly_df = get_monthly_co2_data(as_dataframe=True)
    rng = np.random.default_rng(RANDOM_SEED)

    tasks = []
    for sigma in sigmas:
        for simulation in range(1, n_simulations + 1):
            tasks.append({
                "sigma": sigma,
                "simulation": simulation,
                "annual_noise": rng.normal(0.0, sigma, size=len(years)),
                "years": years,
                "years_train": years_train,
                "years_test": years_test,
                "co2_train_true": co2_train_true,
                "co2_test_true": co2_test_true,
                "monthly_df": monthly_df,
                "include_sarima": include_sarima,
            })

    rows = []
    worker_count = min(max_workers, len(tasks))
    print(
        f"Running robustness simulations: sigmas={sigmas}, "
        f"n={n_simulations}, tasks={len(tasks)}, workers={worker_count}, "
        f"include_sarima={include_sarima}"
    )

    if worker_count == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(run_single_simulation_task(task))
            if index % 10 == 0 or index == len(tasks):
                print(f"  completed {index}/{len(tasks)} tasks")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(run_single_simulation_task, task): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    rows.append({
                        "Sigma": task["sigma"],
                        "Simulation": task["simulation"],
                        "Model": "Simulation task",
                        "Model_Key": "simulation_task",
                        "Model_Family": "internal",
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "MAPE": np.nan,
                        "R2": np.nan,
                        "Status": f"failed: {type(exc).__name__}: {exc}",
                        "Perturbation_Std_ppm": task["sigma"],
                        "Test_Comparison_Target": "original_true_test_data",
                    })
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"  completed {completed}/{len(tasks)} tasks")

    metrics_df = pd.DataFrame(rows)
    metrics_df = add_rank_columns(metrics_df)
    summary_df = build_summary(metrics_df, n_simulations)
    return metrics_df, summary_df, {
        "n_simulations": n_simulations,
        "sigmas": sigmas,
        "include_sarima": include_sarima,
        "max_workers": max_workers,
        "actual_workers": worker_count,
    }


# ============================================================
# 5. Plots
# ============================================================

def main_sigma_dataframe(metrics_df):
    sigmas = sorted(metrics_df["Sigma"].unique())
    sigma = PRIMARY_SIGMA if PRIMARY_SIGMA in sigmas else sigmas[0]
    return sigma, metrics_df[metrics_df["Sigma"] == sigma].copy()


def model_order_from_summary(summary_df, sigma):
    subset = summary_df[summary_df["Sigma"] == sigma].sort_values("RMSE_Mean")
    return subset["Model"].tolist()


def plot_average_rmse(summary_df, sigma):
    plot_df = summary_df[summary_df["Sigma"] == sigma].sort_values("RMSE_Mean")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["Model"], plot_df["RMSE_Mean"], color="#4c78a8")
    plt.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean test RMSE / ppm")
    plt.title(f"Robustness: Mean Test RMSE by Model (sigma={sigma} ppm)")
    plt.grid(axis="y", alpha=0.35)
    save_plot(OUTPUT_DIR / "plots" / f"robustness_mean_rmse_sigma_{sigma}.svg")


def plot_rmse_std(summary_df, sigma):
    plot_df = summary_df[summary_df["Sigma"] == sigma].sort_values("RMSE_Std")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["Model"], plot_df["RMSE_Std"], color="#f58518")
    plt.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("RMSE standard deviation / ppm")
    plt.title(f"Robustness: RMSE Variability by Model (sigma={sigma} ppm)")
    plt.grid(axis="y", alpha=0.35)
    save_plot(OUTPUT_DIR / "plots" / f"robustness_rmse_std_sigma_{sigma}.svg")


def plot_best_frequency(summary_df, sigma):
    plot_df = summary_df[summary_df["Sigma"] == sigma].sort_values("Best_Proportion", ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["Model"], plot_df["Best_Proportion"], color="#54a24b")
    plt.bar_label(bars, labels=[f"{value:.0%}" for value in plot_df["Best_Proportion"]], fontsize=8, padding=3)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, max(1.0, float(plot_df["Best_Proportion"].max()) * 1.15))
    plt.ylabel("Best-model frequency")
    plt.title(f"Robustness: Frequency of Lowest Test RMSE (sigma={sigma} ppm)")
    plt.grid(axis="y", alpha=0.35)
    save_plot(OUTPUT_DIR / "plots" / f"robustness_best_frequency_sigma_{sigma}.svg")


def plot_metric_boxplot(metrics_df, sigma, metric):
    plot_df = metrics_df[metrics_df["Sigma"] == sigma].dropna(subset=[metric]).copy()
    order = (
        plot_df.groupby("Model")[metric]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    data = [plot_df.loc[plot_df["Model"] == model, metric].to_numpy(dtype=float) for model in order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=order, showmeans=True)
    plt.xticks(rotation=25, ha="right")
    ylabel = f"{metric} / ppm" if metric in {"RMSE", "MAE"} else f"{metric} / %"
    plt.ylabel(ylabel)
    plt.title(f"Robustness: {metric} Distribution by Model (sigma={sigma} ppm)")
    plt.grid(axis="y", alpha=0.35)
    save_plot(OUTPUT_DIR / "plots" / f"robustness_{metric.lower()}_boxplot_sigma_{sigma}.svg")


def plot_all_sigma_mean_rmse(summary_df):
    pivot = summary_df.pivot(index="Model", columns="Sigma", values="RMSE_Mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel("Mean test RMSE / ppm")
    ax.set_title("Robustness: Mean Test RMSE Across Perturbation Strengths")
    ax.grid(axis="y", alpha=0.35)
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Sigma / ppm")
    save_plot(OUTPUT_DIR / "plots" / "robustness_mean_rmse_all_sigmas.svg")


def create_plots(metrics_df, summary_df):
    sigma, main_df = main_sigma_dataframe(metrics_df)
    plot_average_rmse(summary_df, sigma)
    plot_rmse_std(summary_df, sigma)
    plot_best_frequency(summary_df, sigma)
    plot_metric_boxplot(main_df, sigma, "RMSE")
    plot_metric_boxplot(main_df, sigma, "MAE")
    plot_metric_boxplot(main_df, sigma, "MAPE")

    if len(summary_df["Sigma"].unique()) > 1:
        plot_all_sigma_mean_rmse(summary_df)


# ============================================================
# 6. Main
# ============================================================

def main():
    reset_output_dir(OUTPUT_DIR)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    metrics_df, summary_df, run_config = run_simulations()

    metrics_csv = OUTPUT_DIR / "robustness_simulation_metrics.csv"
    summary_csv = OUTPUT_DIR / "robustness_summary_by_sigma.csv"
    main_summary_csv = OUTPUT_DIR / "robustness_summary_primary_sigma.csv"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    primary_sigma = PRIMARY_SIGMA if PRIMARY_SIGMA in summary_df["Sigma"].unique() else summary_df["Sigma"].iloc[0]
    primary_summary = summary_df[summary_df["Sigma"] == primary_sigma].copy()
    primary_summary.to_csv(main_summary_csv, index=False, encoding="utf-8-sig")

    create_plots(metrics_df, summary_df)

    most_stable = primary_summary.sort_values(
        ["Best_Proportion", "RMSE_Mean", "RMSE_Std"],
        ascending=[False, True, True],
    ).iloc[0].to_dict()

    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "random_seed": RANDOM_SEED,
        "run_config": run_config,
        "annual_models": ANNUAL_MODEL_SPECS,
        "sarima_spec": SARIMA_SPEC if run_config["include_sarima"] else None,
        "test_years_count": TEST_YEARS_COUNT,
        "test_error_note": "All test metrics compare predictions to original true test data, not perturbed test values.",
        "sarima_perturbation_note": (
            "Annual perturbations are mapped to monthly observations by applying the same "
            "annual noise to every monthly value in that year, then SARIMA forecasts are "
            "aggregated to annual means for comparison."
        ),
        "primary_sigma": primary_sigma,
        "most_stable_model_primary_sigma": most_stable,
        "output_files": [
            str(metrics_csv),
            str(summary_csv),
            str(main_summary_csv),
            *[str(path) for path in sorted((OUTPUT_DIR / "plots").glob("*.svg"))],
        ],
    }
    (OUTPUT_DIR / "robustness_manifest.json").write_text(
        json.dumps(to_builtin(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("QUESTION 1D ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Primary sigma: {primary_sigma} ppm")
    print("\nPrimary-sigma robustness summary:")
    print(
        primary_summary[
            [
                "Model",
                "RMSE_Mean",
                "RMSE_Std",
                "MAE_Mean",
                "MAE_Std",
                "MAPE_Mean",
                "MAPE_Std",
                "Best_Count",
                "Best_Proportion",
            ]
        ].to_string(index=False)
    )
    print("\nMost stable by primary-sigma best frequency, mean RMSE, and RMSE std:")
    print(f"  {most_stable['Model']}")


if __name__ == "__main__":
    main()
