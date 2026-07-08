from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from himcm_data import get_co2_data


OUTPUT_DIR = Path("question_1a_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

years, co2 = get_co2_data()

# Convert arrays into a dictionary for easier year-based calculation
co2_data = dict(zip(years, co2))
# Calculate the 10-year increase ending in 2004
increase_2004 = co2_data[2004] - co2_data[1994]

print("10-year increase ending in 2004:")
print(f"2004 - 1994 = {co2_data[2004]} - {co2_data[1994]} = {increase_2004:.2f} ppm")
print()

# Calculate all previous 10-year increases
# "Previous" means ending before 2004
previous_increases = {}

for year in sorted(co2_data.keys()):
    if year < 2004 and year - 10 in co2_data:
        increase = co2_data[year] - co2_data[year - 10]
        previous_increases[year] = increase

# Find the largest previous 10-year increase
max_previous_year = max(previous_increases, key=previous_increases.get)
max_previous_increase = previous_increases[max_previous_year]

print("Largest previous 10-year increase before 2004:")
print(
    f"{max_previous_year} - {max_previous_year - 10} = "
    f"{co2_data[max_previous_year]} - {co2_data[max_previous_year - 10]} "
    f"= {max_previous_increase:.2f} ppm"
)
print()

# Compare 2004 increase with previous maximum
if increase_2004 > max_previous_increase:
    print("Conclusion:")
    print("The claim is supported: the 10-year increase ending in 2004 is larger than any previous 10-year increase.")
else:
    print("Conclusion:")
    print("The claim is not strictly supported.")
    print(
        f"The 10-year increase ending in 2004 is {increase_2004:.2f} ppm, "
        f"but the largest previous 10-year increase is {max_previous_increase:.2f} ppm, "
        f"ending in {max_previous_year}."
    )

print()
print("All previous 10-year increases:")
for year, increase in previous_increases.items():
    print(f"{year - 10}-{year}: {increase:.2f} ppm")


# ============================================================
# Plot and save 10-year increase comparison
# ============================================================

increase_rows = []

for year in sorted(co2_data.keys()):
    if year - 10 in co2_data and year <= 2004:
        increase = co2_data[year] - co2_data[year - 10]
        increase_rows.append({
            "Start_Year": year - 10,
            "End_Year": year,
            "Ten_Year_Increase_ppm": increase,
            "Is_2004_Window": year == 2004,
            "Is_Previous_Max": year == max_previous_year,
        })

increase_df = pd.DataFrame(increase_rows)

increase_csv = OUTPUT_DIR / "1a_10_year_co2_increases.csv"
increase_plot = OUTPUT_DIR / "1a_10_year_co2_increase_comparison.svg"

increase_df.to_csv(
    increase_csv,
    index=False,
    encoding="utf-8-sig",
)

bar_colors = []

for _, row in increase_df.iterrows():
    if row["Is_Previous_Max"]:
        bar_colors.append("red")
    elif row["Is_2004_Window"]:
        bar_colors.append("green")
    else:
        bar_colors.append("steelblue")

plt.figure(figsize=(12, 6))

plt.bar(
    increase_df["End_Year"],
    increase_df["Ten_Year_Increase_ppm"],
    color=bar_colors,
    label="10-year CO2 increase",
)

plt.axhline(
    y=max_previous_increase,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Largest 10-year increase ({max_previous_increase:.2f} ppm)",
)

plt.xlabel("End year of 10-year period")
plt.ylabel("10-year CO2 increase / ppm")
plt.title("Comparison of 10-Year CO2 Increases Ending in or Before 2004")
plt.ylim(7.5, max_previous_increase + 0.8)
plt.grid(axis="y")
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Rectangle((0, 0), 1, 1, color="red"))
labels.append(f"1993-2003 maximum ({max_previous_increase:.2f} ppm)")
handles.append(plt.Rectangle((0, 0), 1, 1, color="green"))
labels.append(f"1994-2004 increase ({increase_2004:.2f} ppm)")
plt.legend(handles, labels)
plt.tight_layout()
plt.savefig(increase_plot, dpi=300)
plt.close()

print()
print("Saved 1a output files:")
print(f"  {increase_csv}")
print(f"  {increase_plot}")
