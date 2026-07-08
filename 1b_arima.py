# question_1b_sarima_monthly.py
# 1b: SARIMA models for monthly CO2 data
#
# This script uses monthly CO2 data and includes a 12-month seasonal cycle.
#
# Required file in the same directory:
#   himcm_data.py
#
# Required packages:
#   numpy
#   pandas
#   matplotlib
#   statsmodels
#
# Install missing package:
#   pip install statsmodels

# ============================================================
# User settings: edit this block first
# ============================================================

OUTPUT_DIR_NAME = "SARIMA图"
SHOW_FIGURES = False

# Keep the last 10 years as test data.
# Monthly data: 10 years = 120 months.
TEST_MONTHS_COUNT = 120

# To keep the modeling consistent with the original HiMCM dataset,
# this script only uses data up to 2021-12 by default.
#
# If you want to use all uploaded monthly data, set:
# DATA_END_DATE = None
DATA_END_DATE = "2021-12-01"

# Use original monthly average CO2, not deseasonalized data.
# SARIMA itself is used to capture seasonality.
USE_DESEASONALIZED = False

# Custom model count:
#   None: run all models in CANDIDATE_MODELS.
#   1, 2, 3, ...: run only the first N models in CANDIDATE_MODELS.
MODEL_COUNT = None

# Custom SARIMA models.
# Add, remove, or edit dictionaries here to control the model candidates.
# To disable a model, remove its dictionary or comment every line with #.
# Do not wrap dictionaries with triple quotes inside this list.
# order = (p, d, q)
# seasonal_order = (P, D, Q, 12)
# 12 means the seasonal cycle is 12 months.
CANDIDATE_MODELS = [
    {
        "name": "SARIMA(1,1,1)(1,1,1,12)",
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12),
        "trend": "c"
    },
    {
        "name": "SARIMA(1,2,1)(1,1,1,12)",
        "order": (1, 2, 1),
        "seasonal_order": (1, 1, 1, 12),
        "trend": "c"
    },
    {
        "name": "SARIMA(2,1,1)(1,1,1,12)",
        "order": (2, 1, 1),
        "seasonal_order": (1, 1, 1, 12),
        "trend": "c"
    },
    {
        "name": "SARIMA(1,1,1)(2,1,2,12)",
        "order": (1, 1, 1),
        "seasonal_order": (2, 1, 2, 12),
        "trend": "c"
    },
    {
        "name": "SARIMA(1,1,1)(5,1,5,12)",
        "order": (1, 1, 1),
        "seasonal_order": (5, 1, 5, 12),
        "trend": "c"
    },
]

from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from himcm_data import get_monthly_co2_data

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as error:
    raise ImportError(
        "statsmodels is required. Install it with: pip install statsmodels"
    ) from error


def reset_output_dir(output_dir):
    """
    Remove old generated output inside the script's own output directory.
    """

    script_dir = Path(__file__).resolve().parent
    resolved_output_dir = (script_dir / output_dir).resolve()

    if script_dir not in resolved_output_dir.parents:
        raise RuntimeError(f"Refusing to clean outside script directory: {resolved_output_dir}")

    resolved_output_dir.mkdir(exist_ok=True)

    for child in resolved_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def select_active_models(candidate_models, model_count):
    """
    Select and validate the SARIMA models to run.
    """

    if len(candidate_models) == 0:
        raise ValueError("CANDIDATE_MODELS must contain at least one model.")

    names = [spec["name"] for spec in candidate_models]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})

    if duplicate_names:
        raise ValueError(f"Model names must be unique: {duplicate_names}")

    for spec in candidate_models:
        for required_key in ["name", "order", "seasonal_order", "trend"]:
            if required_key not in spec:
                raise ValueError(f"Missing {required_key!r} in model spec: {spec}")

    if model_count is None:
        return list(candidate_models)

    if not isinstance(model_count, int):
        raise ValueError("MODEL_COUNT must be None or an integer.")

    if model_count < 1 or model_count > len(candidate_models):
        raise ValueError(
            f"MODEL_COUNT must be between 1 and {len(candidate_models)}, got {model_count}."
        )

    return list(candidate_models[:model_count])


# ============================================================
# 0. Settings
# ============================================================

ACTIVE_MODELS = select_active_models(CANDIDATE_MODELS, MODEL_COUNT)

OUTPUT_DIR = Path(OUTPUT_DIR_NAME)
reset_output_dir(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)


class ConsoleTee:
    """
    Write console output to the terminal and keep a copy for a text log.
    """

    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def write(self, text):
        self.stream.write(text)
        self.parts.append(text)

    def flush(self):
        self.stream.flush()

    def getvalue(self):
        return "".join(self.parts)


_original_stdout = sys.stdout
console_capture = ConsoleTee(_original_stdout)
sys.stdout = console_capture

# ============================================================
# 1. Read monthly CO2 data
# ============================================================

monthly_df = get_monthly_co2_data(as_dataframe=True)

monthly_df["date"] = pd.to_datetime(
    {
        "year": monthly_df["year"],
        "month": monthly_df["month"],
        "day": 1
    }
)

monthly_df = monthly_df.sort_values("date").reset_index(drop=True)

if USE_DESEASONALIZED:
    value_column = "deseasonalized"
else:
    value_column = "average"

# Remove invalid values if any exist.
monthly_df = monthly_df[monthly_df[value_column] > 0].copy()

# Optional cutoff date.
if DATA_END_DATE is not None:
    monthly_df = monthly_df[monthly_df["date"] <= pd.to_datetime(DATA_END_DATE)].copy()

co2_series = pd.Series(
    monthly_df[value_column].values,
    index=monthly_df["date"]
)

# Set monthly frequency.
co2_series = co2_series.asfreq("MS")

# Fill possible missing monthly values by interpolation.
if co2_series.isna().any():
    co2_series = co2_series.interpolate(method="time")

train_series = co2_series.iloc[:-TEST_MONTHS_COUNT]
test_series = co2_series.iloc[-TEST_MONTHS_COUNT:]

print("=" * 80)
print("SARIMA MODELING FOR MONTHLY CO2 DATA")
print("=" * 80)
print(f"Data used: {value_column}")
print(f"Full data period: {co2_series.index[0].strftime('%Y-%m')} to {co2_series.index[-1].strftime('%Y-%m')}")
print(f"Full data observations: {len(co2_series)}")
print(f"Training period: {train_series.index[0].strftime('%Y-%m')} to {train_series.index[-1].strftime('%Y-%m')}")
print(f"Test period: {test_series.index[0].strftime('%Y-%m')} to {test_series.index[-1].strftime('%Y-%m')}")
print(f"Test observations: {len(test_series)} months")
print(f"Configured candidate models: {len(CANDIDATE_MODELS)}")
print(f"Active models to run: {len(ACTIVE_MODELS)}")
print(f"MODEL_COUNT setting: {MODEL_COUNT}")
print("Active model parameters:")
for index, spec in enumerate(ACTIVE_MODELS, start=1):
    print(
        f"  {index}. {spec['name']} | "
        f"order={spec['order']} | "
        f"seasonal_order={spec['seasonal_order']} | "
        f"trend={spec['trend']}"
    )


# ============================================================
# 2. Evaluation functions
# ============================================================

def evaluate_model(y_true, y_pred):
    """
    Calculate error metrics.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred

    sse = np.sum(residuals ** 2)
    mse = sse / len(y_true)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y_true)) * 100

    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    if sst == 0:
        r2 = np.nan
    else:
        r2 = 1 - sse / sst

    return {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }


def to_builtin(value):
    """
    Convert numpy, pandas, and non-finite values into JSON-friendly values.
    """

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if math.isnan(converted) or math.isinf(converted):
            return None
        return converted
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m")
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }
    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_builtin(item) for item in value]

    return value


def records_from_dataframe(dataframe):
    """
    Convert a DataFrame to plain records for JSON.
    """

    return to_builtin(dataframe.to_dict(orient="records"))


def parameter_row(
    model_name,
    order,
    seasonal_order,
    trend,
    fit_stage,
    result,
    data_start,
    data_end
):
    """
    Build one row for the SARIMA parameter table.
    """

    parameters = to_builtin(result.params)

    return {
        "Model": model_name,
        "Order": str(order),
        "Seasonal_Order": str(seasonal_order),
        "Trend": trend,
        "Fit_Stage": fit_stage,
        "Data_Start": data_start.strftime("%Y-%m"),
        "Data_End": data_end.strftime("%Y-%m"),
        "AIC": result.aic,
        "BIC": result.bic,
        "Parameters_JSON": json.dumps(parameters, ensure_ascii=False),
    }


def fitted_value_rows(
    model_name,
    order,
    seasonal_order,
    trend,
    fit_stage,
    observed_series,
    prediction_series
):
    """
    Build fitted-value and residual rows.
    """

    rows = []

    for date, observed, fitted in zip(
        observed_series.index,
        observed_series.values,
        prediction_series.values
    ):
        residual = observed - fitted

        rows.append({
            "Model": model_name,
            "Order": str(order),
            "Seasonal_Order": str(seasonal_order),
            "Trend": trend,
            "Fit_Stage": fit_stage,
            "Date": date.strftime("%Y-%m"),
            "Observed_CO2": observed,
            "Fitted_CO2": fitted,
            "Residual": residual,
            "Absolute_Residual": abs(residual),
        })

    return rows


def print_metrics(metrics):
    """
    Print error metrics.
    """

    print(f"SSE  = {metrics['SSE']:.4f}")
    print(f"MSE  = {metrics['MSE']:.4f}")
    print(f"RMSE = {metrics['RMSE']:.4f} ppm")
    print(f"MAE  = {metrics['MAE']:.4f} ppm")
    print(f"MAPE = {metrics['MAPE']:.4f}%")
    print(f"R^2  = {metrics['R2']:.4f}")


def valid_start_position(order, seasonal_order):
    """
    Decide how many early fitted values should be skipped.

    SARIMA uses differencing and lag terms, so the earliest fitted values
    are usually unstable and should not be used for error evaluation.
    """

    p, d, q = order
    P, D, Q, s = seasonal_order

    return max(
        p + d + q,
        D * s + d,
        P * s,
        Q * s,
        1
    )


def fit_sarima(series, order, seasonal_order, trend):
    """
    Fit SARIMA model.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)

    return result


def get_in_sample_prediction(result, series, order, seasonal_order):
    """
    Get in-sample prediction and remove early unstable values.
    """

    start_position = valid_start_position(order, seasonal_order)
    start_date = series.index[start_position]
    end_date = series.index[-1]

    prediction = result.get_prediction(
        start=start_date,
        end=end_date
    ).predicted_mean

    y_true = series.loc[prediction.index]

    return y_true, prediction


def safe_filename(model_name):
    """
    Convert model name into a safe file name.
    """

    return (
        model_name
        .replace("SARIMA", "SARIMA")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace(" ", "")
    )


def model_output_dir(model_name):
    """
    Return the per-model output directory.
    """

    return OUTPUT_DIR / "models" / safe_filename(model_name)


# ============================================================
# 3. Plotting functions
# ============================================================

def save_or_show(filename):
    """
    Save current figure and optionally show it.
    """

    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def plot_fit_and_forecast(
    model_name,
    order,
    seasonal_order,
    full_series,
    train_series,
    test_series,
    full_result,
    train_result,
    test_forecast
):
    """
    Plot original monthly data, full-data fitted values,
    and training-only fitted values plus forecast.
    """

    _, full_pred = get_in_sample_prediction(
        full_result,
        full_series,
        order,
        seasonal_order
    )

    _, train_pred = get_in_sample_prediction(
        train_result,
        train_series,
        order,
        seasonal_order
    )

    plt.figure(figsize=(12, 6))

    plt.scatter(
        full_series.index,
        full_series.values,
        color="blue",
        s=8,
        label="Original monthly data"
    )

    plt.plot(
        full_pred.index,
        full_pred.values,
        color="green",
        linewidth=2,
        label="Full-data SARIMA fitted values"
    )

    plt.plot(
        train_pred.index,
        train_pred.values,
        color="orange",
        linewidth=2,
        label="Training-only SARIMA fitted values"
    )

    plt.plot(
        test_series.index,
        test_forecast.values,
        color="orange",
        linewidth=2,
        linestyle="--",
        label="Training-only SARIMA forecast"
    )

    plt.axvline(
        x=test_series.index[0],
        color="gray",
        linestyle="--",
        label="Start of test period"
    )

    plt.xlabel("Date")
    plt.ylabel("CO2 concentration / ppm")
    plt.title(f"{model_name}: Monthly CO2 Fit and Test Forecast")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(model_name) / f"{safe_filename(model_name)}_fit_forecast.svg"
    save_or_show(filename)


def plot_test_forecast(
    model_name,
    test_series,
    test_forecast,
    conf_int=None
):
    """
    Plot observed test data and forecast data.
    """

    plt.figure(figsize=(12, 5))

    plt.scatter(
        test_series.index,
        test_series.values,
        color="blue",
        s=12,
        label="Observed test data"
    )

    plt.plot(
        test_forecast.index,
        test_forecast.values,
        color="orange",
        linewidth=2,
        label="SARIMA forecast"
    )

    if conf_int is not None:
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]

        plt.fill_between(
            conf_int.index,
            lower,
            upper,
            alpha=0.2,
            label="95% confidence interval"
        )

    plt.xlabel("Date")
    plt.ylabel("CO2 concentration / ppm")
    plt.title(f"{model_name}: Test Period Forecast")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(model_name) / f"{safe_filename(model_name)}_test_forecast.svg"
    save_or_show(filename)


def plot_residuals_over_time(
    model_name,
    series,
    prediction,
    data_label
):
    """
    Plot residuals over time.

    Residual = observed - fitted.
    """

    y_true = series.loc[prediction.index]
    residuals = y_true - prediction

    plt.figure(figsize=(12, 5))

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        label="Zero residual"
    )

    plt.plot(
        residuals.index,
        residuals.values,
        linewidth=1.5,
        label="Residuals"
    )

    plt.xlabel("Date")
    plt.ylabel("Residual / ppm")
    plt.title(f"{model_name}: {data_label} Residuals Over Time")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(model_name) / f"{safe_filename(model_name)}_{data_label}_residuals_over_time.svg"
    save_or_show(filename)


def plot_test_residuals_over_time(
    model_name,
    test_series,
    test_forecast
):
    """
    Plot test forecast residuals over time.

    Residual = observed - forecast.
    If residual > 0, the model underestimates CO2.
    If residual < 0, the model overestimates CO2.
    """

    residuals = test_series - test_forecast

    plt.figure(figsize=(12, 5))

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        label="Zero residual"
    )

    plt.plot(
        residuals.index,
        residuals.values,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Test residuals"
    )

    plt.xlabel("Date")
    plt.ylabel("Residual / ppm")
    plt.title(f"{model_name}: Test Residuals Over Time")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(model_name) / f"{safe_filename(model_name)}_test_residuals_over_time.svg"
    save_or_show(filename)


def plot_test_error_comparison(summary_df):
    """
    Plot RMSE, MAE, and MAPE comparison on the test set.
    """

    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width,
        summary_df["Test_RMSE"],
        width,
        label="Test RMSE"
    )

    plt.bar(
        x,
        summary_df["Test_MAE"],
        width,
        label="Test MAE"
    )

    plt.bar(
        x + width,
        summary_df["Test_MAPE"],
        width,
        label="Test MAPE (%)"
    )

    plt.xticks(
        x,
        summary_df["Model"],
        rotation=25,
        ha="right"
    )

    plt.ylabel("Error value")
    plt.title("SARIMA Model Test Error Comparison")
    plt.legend()
    plt.grid(axis="y")

    filename = OUTPUT_DIR / "SARIMA_test_error_comparison.svg"
    save_or_show(filename)


def plot_train_error_comparison(summary_df):
    """
    Plot RMSE, MAE, and MAPE comparison on the training set.
    """

    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width,
        summary_df["Train_RMSE"],
        width,
        label="Train RMSE"
    )

    plt.bar(
        x,
        summary_df["Train_MAE"],
        width,
        label="Train MAE"
    )

    plt.bar(
        x + width,
        summary_df["Train_MAPE"],
        width,
        label="Train MAPE (%)"
    )

    plt.xticks(
        x,
        summary_df["Model"],
        rotation=25,
        ha="right"
    )

    plt.ylabel("Error value")
    plt.title("SARIMA Model Training Error Comparison")
    plt.legend()
    plt.grid(axis="y")

    filename = OUTPUT_DIR / "SARIMA_train_error_comparison.svg"
    save_or_show(filename)


def plot_test_residual_comparison(forecast_df):
    """
    Plot test residual lines for all SARIMA models.
    """

    plt.figure(figsize=(13, 6))

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        label="Zero residual"
    )

    for model_name, model_df in forecast_df.groupby("Model", sort=False):
        dates = pd.to_datetime(model_df["Date"] + "-01")

        plt.plot(
            dates,
            model_df["Error"],
            linewidth=1.5,
            label=model_name
        )

    plt.xlabel("Date")
    plt.ylabel("Residual / ppm")
    plt.title("SARIMA Model Test Residual Comparison")
    plt.legend()
    plt.grid(True)

    filename = OUTPUT_DIR / "SARIMA_test_residual_comparison.svg"
    save_or_show(filename)


def plot_aic_bic_comparison(summary_df):
    """
    Plot full-data AIC and BIC comparison.
    """

    x = np.arange(len(summary_df))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        summary_df["Full_AIC"],
        width,
        label="Full-data AIC"
    )

    plt.bar(
        x + width / 2,
        summary_df["Full_BIC"],
        width,
        label="Full-data BIC"
    )

    plt.xticks(
        x,
        summary_df["Model"],
        rotation=25,
        ha="right"
    )

    plt.ylabel("Information criterion")
    plt.title("SARIMA Full-data AIC and BIC Comparison")
    plt.legend()
    plt.grid(axis="y")

    filename = OUTPUT_DIR / "SARIMA_AIC_BIC_comparison.svg"
    save_or_show(filename)


# ============================================================
# 4. Fit candidate SARIMA models
# ============================================================

summary_rows = []
forecast_rows = []
parameter_rows = []
fitted_rows = []
json_model_rows = []

for spec in ACTIVE_MODELS:
    model_name = spec["name"]
    order = spec["order"]
    seasonal_order = spec["seasonal_order"]
    trend = spec["trend"]

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print(f"Order: {order}")
    print(f"Seasonal order: {seasonal_order}")
    print(f"Trend: {trend}")

    try:
        # ----------------------------------------------------
        # 4.1 Full-data SARIMA model
        # ----------------------------------------------------

        full_result = fit_sarima(
            series=co2_series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend
        )

        full_true, full_pred = get_in_sample_prediction(
            full_result,
            co2_series,
            order,
            seasonal_order
        )

        full_metrics = evaluate_model(
            full_true.values,
            full_pred.values
        )

        print("\nFull-data SARIMA model")
        print("-" * 35)
        print("Parameters:")
        print(full_result.params)
        print(f"AIC = {full_result.aic:.4f}")
        print(f"BIC = {full_result.bic:.4f}")
        print("Full-data in-sample error:")
        print_metrics(full_metrics)

        # ----------------------------------------------------
        # 4.2 Training-only SARIMA model
        # ----------------------------------------------------

        train_result = fit_sarima(
            series=train_series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend
        )

        train_true, train_pred = get_in_sample_prediction(
            train_result,
            train_series,
            order,
            seasonal_order
        )

        train_metrics = evaluate_model(
            train_true.values,
            train_pred.values
        )

        forecast_result = train_result.get_forecast(
            steps=len(test_series)
        )

        test_forecast = forecast_result.predicted_mean
        test_forecast.index = test_series.index

        conf_int = forecast_result.conf_int()
        conf_int.index = test_series.index

        test_metrics = evaluate_model(
            test_series.values,
            test_forecast.values
        )

        print("\nTraining-only SARIMA model")
        print("-" * 35)
        print("Parameters:")
        print(train_result.params)
        print(f"AIC = {train_result.aic:.4f}")
        print(f"BIC = {train_result.bic:.4f}")

        print("\nTraining-only in-sample error:")
        print_metrics(train_metrics)

        print("\nTraining-only model on test data:")
        print_metrics(test_metrics)

        # ----------------------------------------------------
        # 4.3 Store results
        # ----------------------------------------------------

        summary_rows.append({
            "Model": model_name,
            "Order": str(order),
            "Seasonal_Order": str(seasonal_order),
            "Trend": trend,

            "Full_AIC": full_result.aic,
            "Full_BIC": full_result.bic,
            "Train_AIC": train_result.aic,
            "Train_BIC": train_result.bic,

            "Full_SSE": full_metrics["SSE"],
            "Full_MSE": full_metrics["MSE"],
            "Full_RMSE": full_metrics["RMSE"],
            "Full_MAE": full_metrics["MAE"],
            "Full_MAPE": full_metrics["MAPE"],
            "Full_R2": full_metrics["R2"],

            "Train_SSE": train_metrics["SSE"],
            "Train_MSE": train_metrics["MSE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_MAE": train_metrics["MAE"],
            "Train_MAPE": train_metrics["MAPE"],
            "Train_R2": train_metrics["R2"],

            "Test_SSE": test_metrics["SSE"],
            "Test_MSE": test_metrics["MSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_MAE": test_metrics["MAE"],
            "Test_MAPE": test_metrics["MAPE"],
            "Test_R2": test_metrics["R2"],
        })

        parameter_rows.append(
            parameter_row(
                model_name=model_name,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                fit_stage="full_data",
                result=full_result,
                data_start=co2_series.index[0],
                data_end=co2_series.index[-1]
            )
        )
        parameter_rows.append(
            parameter_row(
                model_name=model_name,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                fit_stage="training_data",
                result=train_result,
                data_start=train_series.index[0],
                data_end=train_series.index[-1]
            )
        )

        fitted_rows.extend(
            fitted_value_rows(
                model_name=model_name,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                fit_stage="full_data",
                observed_series=full_true,
                prediction_series=full_pred
            )
        )
        fitted_rows.extend(
            fitted_value_rows(
                model_name=model_name,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                fit_stage="training_data",
                observed_series=train_true,
                prediction_series=train_pred
            )
        )

        for date, observed, predicted in zip(
            test_series.index,
            test_series.values,
            test_forecast.values
        ):
            forecast_rows.append({
                "Model": model_name,
                "Date": date.strftime("%Y-%m"),
                "Observed_CO2": observed,
                "Forecast_CO2": predicted,
                "Error": observed - predicted,
                "Absolute_Error": abs(observed - predicted)
            })

        json_model_rows.append({
            "model": model_name,
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
            "full_data": {
                "aic": full_result.aic,
                "bic": full_result.bic,
                "parameters": to_builtin(full_result.params),
                "metrics": full_metrics,
            },
            "training_data": {
                "aic": train_result.aic,
                "bic": train_result.bic,
                "parameters": to_builtin(train_result.params),
                "metrics": train_metrics,
            },
            "test_data": {
                "metrics": test_metrics,
                "predictions": [
                    {
                        "date": date.strftime("%Y-%m"),
                        "observed_co2": observed,
                        "forecast_co2": predicted,
                        "error": observed - predicted,
                        "absolute_error": abs(observed - predicted),
                    }
                    for date, observed, predicted in zip(
                        test_series.index,
                        test_series.values,
                        test_forecast.values
                    )
                ],
            },
        })

        # ----------------------------------------------------
        # 4.4 Save plots
        # ----------------------------------------------------

        plot_fit_and_forecast(
            model_name=model_name,
            order=order,
            seasonal_order=seasonal_order,
            full_series=co2_series,
            train_series=train_series,
            test_series=test_series,
            full_result=full_result,
            train_result=train_result,
            test_forecast=test_forecast
        )

        plot_test_forecast(
            model_name=model_name,
            test_series=test_series,
            test_forecast=test_forecast,
            conf_int=conf_int
        )

        plot_residuals_over_time(
            model_name=model_name,
            series=co2_series,
            prediction=full_pred,
            data_label="full_data"
        )

        plot_residuals_over_time(
            model_name=model_name,
            series=train_series,
            prediction=train_pred,
            data_label="training_data"
        )

        plot_test_residuals_over_time(
            model_name=model_name,
            test_series=test_series,
            test_forecast=test_forecast
        )

    except Exception as error:
        print(f"\n{model_name} failed.")
        print(f"Error message: {error}")


# ============================================================
# 5. Save summary tables and comparison plots
# ============================================================

summary_df = pd.DataFrame(summary_rows)
forecast_df = pd.DataFrame(forecast_rows)
parameter_df = pd.DataFrame(parameter_rows)
fitted_df = pd.DataFrame(fitted_rows)

if len(summary_df) > 0:
    summary_df = summary_df.sort_values(
        by="Test_RMSE"
    ).reset_index(drop=True)

    summary_csv = OUTPUT_DIR / "SARIMA_model_summary.csv"
    forecast_csv = OUTPUT_DIR / "SARIMA_test_forecasts.csv"
    parameter_csv = OUTPUT_DIR / "SARIMA_model_parameters.csv"
    fitted_csv = OUTPUT_DIR / "SARIMA_fitted_values.csv"
    results_json_path = OUTPUT_DIR / "SARIMA_results.json"
    manifest_json_path = OUTPUT_DIR / "SARIMA_run_manifest.json"
    console_output_path = OUTPUT_DIR / "SARIMA_console_output.txt"

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    forecast_df.to_csv(
        forecast_csv,
        index=False,
        encoding="utf-8-sig"
    )

    parameter_df.to_csv(
        parameter_csv,
        index=False,
        encoding="utf-8-sig"
    )

    fitted_df.to_csv(
        fitted_csv,
        index=False,
        encoding="utf-8-sig"
    )

    per_model_data_files = []
    json_models_by_name = {
        model_payload["model"]: model_payload
        for model_payload in json_model_rows
    }
    successful_model_names = set(json_models_by_name)

    for spec in ACTIVE_MODELS:
        model_name = spec["name"]
        if model_name not in successful_model_names:
            continue

        model_key = safe_filename(model_name)
        model_dir = model_output_dir(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_parameter_csv = model_dir / f"{model_key}_model_parameters.csv"
        model_fitted_csv = model_dir / f"{model_key}_fitted_values.csv"
        model_forecast_csv = model_dir / f"{model_key}_test_forecasts.csv"
        model_results_json = model_dir / f"{model_key}_results.json"

        parameter_df[parameter_df["Model"] == model_name].to_csv(
            model_parameter_csv,
            index=False,
            encoding="utf-8-sig"
        )
        fitted_df[fitted_df["Model"] == model_name].to_csv(
            model_fitted_csv,
            index=False,
            encoding="utf-8-sig"
        )
        forecast_df[forecast_df["Model"] == model_name].to_csv(
            model_forecast_csv,
            index=False,
            encoding="utf-8-sig"
        )
        model_results_json.write_text(
            json.dumps(
                to_builtin(json_models_by_name[model_name]),
                indent=2,
                allow_nan=False
            ),
            encoding="utf-8"
        )

        per_model_data_files.extend([
            str(model_parameter_csv.relative_to(OUTPUT_DIR)),
            str(model_fitted_csv.relative_to(OUTPUT_DIR)),
            str(model_forecast_csv.relative_to(OUTPUT_DIR)),
            str(model_results_json.relative_to(OUTPUT_DIR)),
        ])

    print("\n" + "=" * 80)
    print("SUMMARY SORTED BY TEST RMSE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    best_test_model = summary_df.loc[summary_df["Test_RMSE"].idxmin()]
    best_aic_model = summary_df.loc[summary_df["Full_AIC"].idxmin()]
    best_bic_model = summary_df.loc[summary_df["Full_BIC"].idxmin()]

    print("\n" + "=" * 80)
    print("BEST MODEL INDICATORS")
    print("=" * 80)
    print(f"Best by test RMSE: {best_test_model['Model']}")
    print(f"Best by full-data AIC: {best_aic_model['Model']}")
    print(f"Best by full-data BIC: {best_bic_model['Model']}")

    plot_train_error_comparison(summary_df)
    plot_test_error_comparison(summary_df)
    plot_test_residual_comparison(forecast_df)
    plot_aic_bic_comparison(summary_df)

    model_plot_files = []
    for spec in ACTIVE_MODELS:
        if spec["name"] not in successful_model_names:
            continue

        model_key = safe_filename(spec["name"])
        model_relative_dir = Path("models") / model_key
        model_plot_files.extend([
            str(model_relative_dir / f"{model_key}_fit_forecast.svg"),
            str(model_relative_dir / f"{model_key}_test_forecast.svg"),
            str(model_relative_dir / f"{model_key}_full_data_residuals_over_time.svg"),
            str(model_relative_dir / f"{model_key}_training_data_residuals_over_time.svg"),
            str(model_relative_dir / f"{model_key}_test_residuals_over_time.svg"),
        ])

    output_files = sorted([
        summary_csv.name,
        forecast_csv.name,
        parameter_csv.name,
        fitted_csv.name,
        results_json_path.name,
        manifest_json_path.name,
        console_output_path.name,
        "SARIMA_train_error_comparison.svg",
        "SARIMA_test_error_comparison.svg",
        "SARIMA_test_residual_comparison.svg",
        "SARIMA_AIC_BIC_comparison.svg",
        *model_plot_files,
        *per_model_data_files,
    ])

    results_payload = {
        "metadata": {
            "script": Path(__file__).name,
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(OUTPUT_DIR),
            "data_used": value_column,
            "use_deseasonalized": USE_DESEASONALIZED,
            "data_end_date": DATA_END_DATE,
            "full_data_period": {
                "start": co2_series.index[0].strftime("%Y-%m"),
                "end": co2_series.index[-1].strftime("%Y-%m"),
                "observations": len(co2_series),
            },
            "training_period": {
                "start": train_series.index[0].strftime("%Y-%m"),
                "end": train_series.index[-1].strftime("%Y-%m"),
                "observations": len(train_series),
            },
            "test_period": {
                "start": test_series.index[0].strftime("%Y-%m"),
                "end": test_series.index[-1].strftime("%Y-%m"),
                "observations": len(test_series),
            },
        },
        "models": json_model_rows,
        "summary": records_from_dataframe(summary_df),
        "parameters": records_from_dataframe(parameter_df),
        "fitted_values": records_from_dataframe(fitted_df),
        "test_forecasts": records_from_dataframe(forecast_df),
    }

    manifest_payload = {
        "script": Path(__file__).name,
        "run_timestamp": results_payload["metadata"]["run_timestamp"],
        "output_dir": str(OUTPUT_DIR),
        "config": {
            "test_months_count": TEST_MONTHS_COUNT,
            "data_end_date": DATA_END_DATE,
            "use_deseasonalized": USE_DESEASONALIZED,
            "show_figures": SHOW_FIGURES,
            "model_count": MODEL_COUNT,
            "configured_model_count": len(CANDIDATE_MODELS),
            "active_model_count": len(ACTIVE_MODELS),
            "configured_candidate_models": CANDIDATE_MODELS,
            "active_models": ACTIVE_MODELS,
        },
        "data": results_payload["metadata"],
        "best_models": {
            "best_by_test_rmse": {
                "model": best_test_model["Model"],
                "test_rmse": best_test_model["Test_RMSE"],
            },
            "best_by_full_data_aic": {
                "model": best_aic_model["Model"],
                "full_aic": best_aic_model["Full_AIC"],
            },
            "best_by_full_data_bic": {
                "model": best_bic_model["Model"],
                "full_bic": best_bic_model["Full_BIC"],
            },
        },
        "output_files": output_files,
    }

    results_json_path.write_text(
        json.dumps(to_builtin(results_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest_json_path.write_text(
        json.dumps(to_builtin(manifest_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nSaved result files:")
    for filename in output_files:
        print(f"  {OUTPUT_DIR / filename}")

    console_output_path.write_text(
        console_capture.getvalue(),
        encoding="utf-8",
    )
else:
    print("\nNo SARIMA model was fitted successfully.")
    (OUTPUT_DIR / "SARIMA_console_output.txt").write_text(
        console_capture.getvalue(),
        encoding="utf-8",
    )
