"""
Workout CSV fatigue analysis (Notion export style)

What it does:
- Loads your CSV from a file path you set
- Parses Notion-style date strings like: "July 14, 2024 5:35 AM (EDT)"
- Extracts exercise names from the "Exercises" column
- Produces:
  1) Time series plots: Rating and Body Weight
  2) Scatter plots: Rating vs #exercises, Rating vs days since last session
  3) "Fatigue suspects" table: exercises overrepresented in low-rating sessions (<=5)
  4) OPTIONAL: a controlled model (ridge regression) to estimate per-exercise association with rating
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# USER: set your CSV path here
# ----------------------------
CSV_PATH = r"C:\Users\abhin\Desktop\lift-o-lytics\bod_data.csv" # <-- CHANGE THIS





# ----------------------------
# Helpers
# ----------------------------
def parse_notion_datetime(x: object) -> pd.Timestamp:
    """Parse strings like 'July 14, 2024 5:35 AM (EDT)' or fallback."""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT

    # Remove trailing timezone in parentheses, e.g. (EDT)
    s = re.sub(r"\s*\(.*\)\s*$", "", s)

    # Try common formats first
    for fmt in ("%B %d, %Y %I:%M %p", "%B %d, %Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass

    # Fallback
    return pd.to_datetime(s, errors="coerce")


def extract_exercise_names(ex_str: object) -> List[str]:
    """
    Exercises column often looks like:
    'RG Lat Pulldown (https://...), CG Seated Cable Row (https://...), ...'
    This returns just the names:
    ['RG Lat Pulldown', 'CG Seated Cable Row', ...]
    """
    if pd.isna(ex_str):
        return []
    s = str(ex_str).strip()
    if not s:
        return []

    # Split by '), ' to keep URL-parentheses grouped
    parts = s.split("), ")
    cleaned = []
    for i, p in enumerate(parts):
        p = p.strip()
        if i < len(parts) - 1 and not p.endswith(")"):
            p = p + ")"
        cleaned.append(p)

    names: List[str] = []
    for p in cleaned:
        # Name before "(http...)"
        m = re.match(r"^(.*?)\s*\(https?://", p)
        name = (m.group(1) if m else p.split("(")[0]).strip()
        if name:
            names.append(name)

    return names


def workout_group(workout: object) -> str:
    """Coarse workout grouping from Workout column."""
    if pd.isna(workout):
        return "Unknown"
    w = str(workout).strip()

    m = re.match(r"^(Pull|Push|Leg|Arm|Full|Chest|Back|Shoulder)", w, flags=re.I)
    if m:
        return m.group(1).title()

    return w.split()[0].title() if w else "Unknown"


# ----------------------------
# Load + clean
# ----------------------------
csv_file = Path(CSV_PATH)
if not csv_file.exists():
    raise FileNotFoundError(f"CSV not found: {csv_file}")

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

required_cols = {"Workout", "Body Weight (lbs)", "Date", "Exercises", "Rating"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["dt"] = df["Date"].map(parse_notion_datetime)
df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

# Ensure numeric columns
df["Body Weight (lbs)"] = pd.to_numeric(df["Body Weight (lbs)"], errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df["exercise_list"] = df["Exercises"].map(extract_exercise_names)
df["n_exercises"] = df["exercise_list"].map(len)
df["group"] = df["Workout"].map(workout_group)

# Gap since last session (days)
df["day_gap"] = df["dt"].diff().dt.total_seconds() / 86400.0
median_gap = np.nanmedian(df["day_gap"].values)
df["day_gap"] = df["day_gap"].fillna(median_gap if np.isfinite(median_gap) else 0.0)

print("\n=== Basic info ===")
print("Rows:", len(df))
print("Date range:", df["dt"].min(), "->", df["dt"].max())
print("Mean rating:", df["Rating"].mean())
print("Mean bodyweight:", df["Body Weight (lbs)"].mean())


# ----------------------------
# Plots: time series
# ----------------------------
plt.figure()
plt.plot(df["dt"], df["Body Weight (lbs)"])
plt.title("Body Weight Over Time")
plt.xlabel("Date")
plt.ylabel("Body Weight (lbs)")
plt.show()

plt.figure()
plt.plot(df["dt"], df["Rating"])
plt.title("Workout Rating Over Time")
plt.xlabel("Date")
plt.ylabel("Rating")
plt.show()

# Scatter: rating vs session size and rest gap
plt.figure()
plt.scatter(df["n_exercises"], df["Rating"])
plt.title("Rating vs Number of Exercises")
plt.xlabel("# Exercises in session")
plt.ylabel("Rating")
plt.show()

plt.figure()
plt.scatter(df["day_gap"], df["Rating"])
plt.title("Rating vs Days Since Last Workout")
plt.xlabel("Days since last workout")
plt.ylabel("Rating")
plt.show()


# ----------------------------
# Exercise "fatigue suspects" table
# (overrepresented in low-rated sessions)
# ----------------------------
LOW_THRESHOLD = 5  # You can change this
MIN_EXERCISE_COUNT = 6  # only consider exercises appearing >= this many sessions

all_ex = [e for lst in df["exercise_list"] for e in lst]
ex_counts = pd.Series(all_ex).value_counts()

overall_low_rate = float((df["Rating"] <= LOW_THRESHOLD).mean())

rows = []
for ex, cnt in ex_counts.items():
    if cnt < MIN_EXERCISE_COUNT:
        continue
    present = df["exercise_list"].apply(lambda lst: ex in lst)
    low_rate_present = float((df.loc[present, "Rating"] <= LOW_THRESHOLD).mean())
    rows.append(
        {
            "Exercise": ex,
            "Count": int(cnt),
            "MeanRating_whenPresent": float(df.loc[present, "Rating"].mean()),
            "LowRate_whenPresent": low_rate_present,
            "LowRate_minus_overall": low_rate_present - overall_low_rate,
        }
    )

suspects = (
    pd.DataFrame(rows)
    .sort_values(["LowRate_whenPresent", "Count"], ascending=[False, False])
    .reset_index(drop=True)
)

print("\n=== Fatigue suspects (exercise enrichment in low-rated sessions) ===")
print(f"Overall low-rate (Rating <= {LOW_THRESHOLD}): {overall_low_rate:.3f}\n")
print(suspects.head(30).to_string(index=False))


# ----------------------------
# OPTIONAL: Controlled model
# Ridge regression on rating with controls:
# - day_gap, n_exercises, workout group
# - exercise presence indicators
# Gives you "association" with rating, not causation.
# ----------------------------
DO_CONTROLLED_MODEL = True
TOP_EXERCISES_FOR_MODEL = 60  # limit feature count to avoid huge matrices

if DO_CONTROLLED_MODEL:
    # Keep the most frequent exercises for the model
    model_ex = ex_counts.head(TOP_EXERCISES_FOR_MODEL).index.tolist()

    # Build design matrix
    X = pd.DataFrame(index=df.index)
    X["day_gap"] = df["day_gap"].astype(float)
    X["n_exercises"] = df["n_exercises"].astype(float)

    # Workout group dummies
    grp = pd.get_dummies(df["group"], prefix="grp", drop_first=True)
    X = pd.concat([X, grp], axis=1)

    # Exercise indicators
    for ex in model_ex:
        X[f"ex_{ex}"] = df["exercise_list"].apply(lambda lst: 1 if ex in lst else 0)

    # Drop rows with missing rating
    mask = df["Rating"].notna()
    X = X.loc[mask].copy()
    y = df.loc[mask, "Rating"].astype(float)

    # Standardize
    X_mean = X.mean()
    X_std = X.std().replace(0, 1)
    Xs = (X - X_mean) / X_std

    # Fit ridge
    try:
        from sklearn.linear_model import Ridge
    except ImportError as e:
        raise ImportError("Install scikit-learn: pip install scikit-learn") from e

    model = Ridge(alpha=5.0)
    model.fit(Xs, y)

    coef = pd.Series(model.coef_, index=X.columns).sort_values()
    ex_coef = coef[coef.index.str.startswith("ex_")]

    print("\n=== Controlled model: exercises most associated with LOWER ratings ===")
    print(ex_coef.head(25).to_string())

    print("\n=== Controlled model: exercises most associated with HIGHER ratings ===")
    print(ex_coef.tail(25).to_string())
