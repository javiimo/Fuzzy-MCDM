# ================= LOOWA decision analysis script ==================
#
# Implements:
#   • LOOWA inside four lexicographic concept blocks
#   • (W)OWA across concepts
#   • optional weighted-average or OWA for Seasonality
#   • sensitivity analysis over outer-level orness
#   • monotonicity sanity check

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# -------------------------------------------------------------------
# 1. Helper utilities
# -------------------------------------------------------------------

def owa_weights(n: int, orness: float) -> np.ndarray:
    """
    Build an OWA weight vector of length *n* that achieves the requested orness
    using Yager’s power-quantifier method (Q(x) = x**a).

    orness = 1 → behaves like Max,  0.5 → arithmetic mean,  0 → Min.
    """
    if orness <= 0.0:                     # pure Min
        w = np.zeros(n)
        w[-1] = 1.0
        return w
    if orness >= 1.0:                     # pure Max
        w = np.zeros(n)
        w[0] = 1.0
        return w

    # Solve for exponent a:  orness = 2 / (a + 2)
    a = max(1e-6, (2.0 / orness) - 2.0)
    idx = np.arange(1, n + 1)
    w = (idx / n) ** a - ((idx - 1) / n) ** a
    return w / w.sum()


def owa(values: np.ndarray, orness: float) -> float:
    """Ordered Weighted Averaging aggregation of a 1-D array."""
    w = owa_weights(len(values), orness)
    ordered = np.sort(values)[::-1]       # descending
    return float(np.dot(ordered, w))


def loowa(block_vals: np.ndarray) -> float:
    """
    Lexicographic-Ordinal OWA for a single block already sorted by priority
    (index 0 = highest priority).

    Implements (Yager 2010):
        T_j = min_{k<j} v_k
        LOOWA(v) = max_j  min(v_j, T_j)
    """
    current_min = 1.0
    best = 0.0
    for v in block_vals:
        candidate = min(v, current_min)
        best = max(best, candidate)
        current_min = min(current_min, v)
    return best


# -------------------------------------------------------------------
# 2. Configuration (edit to taste)
# -------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # Column groups in priority order
    "size_cols": [
        "Size_small", "Size_mid_small", "Size_medium",
        "Size_mid_large", "Size_large",
    ],
    "risk_cols": [
        "Risk_low", "Risk_mid_low", "Risk_mid",
        "Risk_mid_high", "Risk_high",
    ],
    "env_cols": [
        "EnvImpact_close", "EnvImpact_mid_close", "EnvImpact_mid",
        "EnvImpact_mid_far", "EnvImpact_far",
    ],
    "closeness_cols": [
        "Closeness_close", "Closeness_mid_close", "Closeness_mid",
        "Closeness_mid_far", "Closeness_far",
    ],  # priority: close ≫ far

    # Seasonality
    "seasonality_cols": ["winter-like", "summer-like", "is-like"],
    "seasonality_method": "weighted_avg",        # or "owa"
    "seasonality_weights": [0.4, 0.4, 0.2],      # used if weighted_avg
    "seasonality_orness": 0.5,                   # used if OWA

    # Outer aggregation
    "outer_orness": 0.5,        # 0.5 = neutral; >0.5 optimistic; <0.5 pessimistic
}


# -------------------------------------------------------------------
# 3. Core pipeline functions
# -------------------------------------------------------------------

def aggregate_matrix(
    csv_path: str | Path,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with intermediate concept scores and the final Global
    score for every alternative in the decision matrix.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    df = pd.read_csv(csv_path)

    # Quick header check
    all_needed = (
        ["Alternative", "Highest Concurrency"]
        + cfg["size_cols"]
        + cfg["risk_cols"]
        + cfg["env_cols"]
        + cfg["closeness_cols"]
        + cfg["seasonality_cols"]
    )
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    records: list[dict] = []

    for _, row in df.iterrows():
        # --- LOOWA within blocks ---
        size_score = loowa(row[cfg["size_cols"]].to_numpy(float))
        risk_score = loowa(row[cfg["risk_cols"]].to_numpy(float))
        env_score  = loowa(row[cfg["env_cols"]].to_numpy(float))
        close_score = loowa(row[cfg["closeness_cols"]].to_numpy(float))

        # --- Seasonality block ---
        season_vals = row[cfg["seasonality_cols"]].to_numpy(float)
        if cfg["seasonality_method"] == "weighted_avg":
            w = np.asarray(cfg["seasonality_weights"], float)
            season_score = float(np.dot(season_vals, w / w.sum()))
        else:  # OWA
            season_score = owa(season_vals, cfg["seasonality_orness"])

        # --- Vector of concept scores ---
        concept_vector = np.array([
            float(row["Highest Concurrency"]),
            size_score,
            risk_score,
            env_score,
            close_score,
            season_score,
        ])

        global_score = owa(concept_vector, cfg["outer_orness"])

        records.append({
            "Alternative": row["Alternative"],
            "Concurrency": row["Highest Concurrency"],
            "Size": size_score,
            "Risk": risk_score,
            "EnvImpact": env_score,
            "Closeness": close_score,
            "Seasonality": season_score,
            "Global": global_score,
        })

    out = (
        pd.DataFrame(records)
        .set_index("Alternative")
        .sort_values("Global", ascending=False)
    )
    return out


# -------------------------------------------------------------------
# 4. Diagnostics
# -------------------------------------------------------------------

def sensitivity_analysis(
    csv_path: str | Path,
    outer_orness_values=(0.25, 0.5, 0.75),
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Re-compute scores for several outer-level orness values and return the ranks
    of each alternative for each scenario.
    """
    base = {**(cfg or {})}
    rank_table = {}
    for α in outer_orness_values:
        base["outer_orness"] = α
        scores = aggregate_matrix(csv_path, base)
        rank_table[f"rank@α={α}"] = (
            scores["Global"].rank(ascending=False, method="min").astype(int)
        )
    return pd.DataFrame(rank_table)


def monotonicity_check(
    csv_path: str | Path,
    cfg: dict | None = None,
    epsilon: float = 1e-3,
) -> list[str]:
    """
    Sanity-check: raising the highest-priority value in any block should never
    lower the final score (monotonicity).
    Returns list of violation messages (empty ⇒ pass).
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    df = pd.read_csv(csv_path)

    if df.empty:
        return ["CSV has no rows"]

    alt0 = df.iloc[0].copy()
    baseline = aggregate_matrix(csv_path, cfg).loc[alt0["Alternative"], "Global"]

    blocks = {
        "size": cfg["size_cols"][0],
        "risk": cfg["risk_cols"][0],
        "env": cfg["env_cols"][0],
        "closeness": cfg["closeness_cols"][0],
    }

    msgs: list[str] = []
    for block, col in blocks.items():
        df_mod = df.copy()
        df_mod.loc[0, col] = min(1.0, df_mod.loc[0, col] + epsilon)

        tmp = "_tmp_loowa.csv"
        df_mod.to_csv(tmp, index=False)
        new_global = aggregate_matrix(tmp, cfg).loc[alt0["Alternative"], "Global"]
        Path(tmp).unlink(missing_ok=True)

        if new_global + 1e-9 < baseline:
            msgs.append(
                f"[{block}] Increasing highest-priority column '{col}' "
                f"decreased the global score!"
            )
    return msgs


# -------------------------------------------------------------------
# 5. Command-line entry
# -------------------------------------------------------------------

def main() -> None:
    """
    Example pipeline:
      1) aggregate once (outer_orness = 0.5)
      2) show top-5 alternatives
      3) run sensitivity analysis for α = 0.3, 0.5, 0.7
      4) run monotonicity sanity check
    """
    csv_path = "decision_matrix_expanded.csv"      # ← adjust if needed
    cfg = {
        # Example: change the outer orness here
        "outer_orness": 0.5,
    }

    # 1. Aggregate
    scores = aggregate_matrix(csv_path, cfg)
    print("\n*** Global scores (top 5) ***")
    print(scores.head(5)[["Global"]])

    # 2. Sensitivity
    sens = sensitivity_analysis(
        csv_path,
        outer_orness_values=[0.3, 0.5, 0.7],
        cfg=cfg,
    )
    print("\n*** Sensitivity analysis (rank positions) ***")
    print(sens)

    # 3. Sanity checks
    issues = monotonicity_check(csv_path, cfg)
    if issues:
        print("\n*** Sanity issues detected ***")
        for m in issues:
            print(" !", m)
    else:
        print("\nSanity checks passed ✅")


if __name__ == "__main__":
    main()
