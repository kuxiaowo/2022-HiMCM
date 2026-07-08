# question_raw_scatter_plots.py
# Generate raw-data scatter plots:
# 1. Annual CO2 concentration
# 2. Monthly CO2 concentration
# 3. Monthly CO2 first difference
# 4. Monthly CO2 second difference
# 5. Annual global temperature anomaly
#
# Required file in the same directory:
#   himcm_data.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from himcm_data import (
    get_co2_data,
    get_monthly_co2_series,
    get_temperature_data
)


# ============================================================
# 0. Settings
# ============================================================

SAVE_FIGURES = True

OUTPUT_DIR = Path("scatter_plots_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

annual_co2_output = OUTPUT_DIR / "annual_co2_scatter.svg"
monthly_co2_output = OUTPUT_DIR / "monthly_co2_scatter.svg"
monthly_co2_first_diff_output = OUTPUT_DIR / "monthly_co2_first_difference_scatter.svg"
monthly_co2_second_diff_output = OUTPUT_DIR / "monthly_co2_second_difference_scatter.svg"
temperature_output = OUTPUT_DIR / "temperature_anomaly_scatter.svg"
# ============================================================
# 1. Annual CO2 concentration scatter plot
# ============================================================

annual_years, annual_co2 = get_co2_data()

plt.figure(figsize=(10, 6))

plt.scatter(
    annual_years,
    annual_co2,
    s=25,
    label="Annual CO2 data"
)

plt.xlabel("Year")
plt.ylabel("CO2 concentration / ppm")
plt.title("Annual CO2 Concentration")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(annual_co2_output, dpi=300)

plt.show()


# ============================================================
# 2. Monthly CO2 concentration scatter plot
# ============================================================

monthly_decimal_date, monthly_co2 = get_monthly_co2_series(
    use_deseasonalized=False
)

plt.figure(figsize=(10, 6))

plt.scatter(
    monthly_decimal_date,
    monthly_co2,
    s=8,
    label="Monthly CO2 data"
)

plt.xlabel("Year")
plt.ylabel("CO2 concentration / ppm")
plt.title("Monthly CO2 Concentration")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(monthly_co2_output, dpi=300)

plt.show()


# ============================================================
# 3. Monthly CO2 first difference scatter plot
# ============================================================

monthly_co2_first_diff = np.diff(monthly_co2, n=1)
monthly_first_diff_decimal_date = monthly_decimal_date[1:]

plt.figure(figsize=(10, 6))

plt.axhline(
    y=0,
    color="gray",
    linestyle="--",
    label="Zero difference"
)

plt.scatter(
    monthly_first_diff_decimal_date,
    monthly_co2_first_diff,
    s=8,
    label="Monthly CO2 first difference"
)

plt.xlabel("Year")
plt.ylabel("First difference / ppm per month")
plt.title("Monthly CO2 Concentration First Difference")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(monthly_co2_first_diff_output, dpi=300)

plt.show()


# ============================================================
# 4. Monthly CO2 second difference scatter plot
# ============================================================

monthly_co2_second_diff = np.diff(monthly_co2, n=2)
monthly_second_diff_decimal_date = monthly_decimal_date[2:]

plt.figure(figsize=(10, 6))

plt.axhline(
    y=0,
    color="gray",
    linestyle="--",
    label="Zero difference"
)

plt.scatter(
    monthly_second_diff_decimal_date,
    monthly_co2_second_diff,
    s=8,
    label="Monthly CO2 second difference"
)

plt.xlabel("Year")
plt.ylabel("Second difference / ppm")
plt.title("Monthly CO2 Concentration Second Difference")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(monthly_co2_second_diff_output, dpi=300)

plt.show()


# ============================================================
# 5. Annual global temperature anomaly scatter plot
# ============================================================

temperature_years, temperature_anomaly = get_temperature_data()

plt.figure(figsize=(10, 6))

plt.scatter(
    temperature_years,
    temperature_anomaly,
    s=25,
    label="Annual temperature anomaly"
)

plt.xlabel("Year")
plt.ylabel("Temperature anomaly / °C")
plt.title("Annual Global Land-Ocean Temperature Anomaly")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(temperature_output, dpi=300)

plt.show()


# ============================================================
# 4. Output information
# ============================================================

print("Scatter plots generated.")

if SAVE_FIGURES:
    print(f"Saved: {annual_co2_output}")
    print(f"Saved: {monthly_co2_output}")
    print(f"Saved: {monthly_co2_first_diff_output}")
    print(f"Saved: {monthly_co2_second_diff_output}")
    print(f"Saved: {temperature_output}")
