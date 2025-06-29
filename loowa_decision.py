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
    df = pd.read_csv(csv_path).round(3)   

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
    for alpha in outer_orness_values:
        base["outer_orness"] = alpha
        scores = aggregate_matrix(csv_path, base)
        rank_table[f"rank@alpha={alpha}"] = (
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
    df = pd.read_csv(csv_path).round(3)   

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
# 4-bis.  Orness sweep – find every winner over [0, 1]
# -------------------------------------------------------------------
from collections import defaultdict
from typing import Iterable, Tuple, Set, Dict

def sweep_orness_winners(
    csv_path: str | Path,
    n_points: int = 100,
    cfg: dict | None = None,
    return_mapping: bool = False,
) -> Tuple[Set[str], Dict[str, list[float]] | None]:
    """
    Sweep the outer-level orness alpha across `n_points` equally-spaced values
    in [0, 1].  For each alpha, aggregate the matrix and find the winner(s).
    
    Parameters
    ----------
    csv_path : str or Path
        Location of the normalised decision matrix.
    n_points : int, optional
        Number of alpha values (default 21 → increments of 0.05).
    cfg : dict, optional
        Base configuration overriding DEFAULT_CONFIG.
    return_mapping : bool, optional
        If True, also return a dict {alternative: [alpha₁, alpha₂, …]} recording
        the exact alpha values at which the alternative is a winner.
    
    Returns
    -------
    winners : set[str]
        Pool of all alternatives that are winners for at least one alpha.
    mapping : dict[str, list[float]] or None
        Only if `return_mapping` is True; otherwise None.
    """
    base_cfg = {**(cfg or {})}
    alphas = np.linspace(0.0, 1.0, n_points)

    winners: set[str] = set()
    mapping: dict[str, list[float]] = defaultdict(list)

    for alpha in alphas:
        base_cfg["outer_orness"] = float(alpha)
        scores = aggregate_matrix(csv_path, base_cfg)

        best_val = scores["Global"].max()
        alts = scores[scores["Global"] >= best_val - 1e-12].index.tolist()

        winners.update(alts)
        for alt in alts:
            mapping[alt].append(float(alpha))

    return (winners, mapping if return_mapping else None)

# ================================================================
# PLOT  A — 4-pane “winner-region” diagram under four seasonality
#           preference scenarios (winter, summer, inter-season,
#           equal weights)
# ================================================================
# -------------------------------------------------------------------
# Adapter so both old & new helper code compile together
# -------------------------------------------------------------------
def agg_with_alpha(csv_path, outer_orness, cfg=None):
    """
    Wrapper around the original aggregate_matrix(csv_path, cfg) that
    EXPECTS outer_orness to sit inside cfg.
    """
    cfg = {**(cfg or {}), "outer_orness": outer_orness}
    return aggregate_matrix(csv_path, cfg)

from typing import Sequence, Dict, Iterable, Set
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_winner_regions_four(
    csv_path: str | Path,
    winner_pool: Set[str],
    outer_orness_grid: int | Iterable[float] = 101,
    base_cfg: Dict | None = None,
    season_scenarios: Dict[str, Sequence[float]] | None = None,
    colours: Dict[str, str] | None = None,
) -> None:
    """
    Produce a 2×2 figure (Matplotlib) – one panel per seasonality weight
    scenario – where each panel shows *coloured regions* indicating which
    alternative(s) in `winner_pool` dominate as the outer OWA orness alpha sweeps
    from 0 to 1.

    Parameters
    ----------
    csv_path : str | Path
        Path to the fully-normalised decision matrix.
    winner_pool : set[str]
        The alternatives you want to track (usually the overall winners you
        found in the coarse sweep).
    outer_orness_grid : int | iterable, optional
        • If **int** → number of equally-spaced alpha values in [0,1].  
        • If **iterable** → direct list/array of alpha’s.
    base_cfg : dict, optional
        Configuration baseline; copied and slightly modified inside each panel.
    season_scenarios : dict[label → 3-tuple], optional
        Custom seasonality weightings.  Default =  
        {
            "Winter focus": (1, 0, 0),
            "Summer focus": (0, 1, 0),
            "Inter-season": (0, 0, 1),
            "Equal":        (1/3, 1/3, 1/3),
        }
    colours : dict[alt → colour], optional
        Colour to use for each alternative when shading its winning span.
    """
    # ---------- grid of alpha values ----------
    if isinstance(outer_orness_grid, int):
        alphas = np.linspace(0.0, 1.0, outer_orness_grid)
    else:
        alphas = np.asarray(list(outer_orness_grid), float)

    # ---------- seasonality scenarios ----------
    season_scenarios = season_scenarios or {
        "Winter focus": (1.0, 0.0, 0.0),
        "Summer focus": (0.0, 1.0, 0.0),
        "Inter-season": (0.0, 0.0, 1.0),
        "Equal":        (1/3, 1/3, 1/3),
    }

    # colours for each alt
    import itertools
    base_palette = plt.cm.get_cmap("tab20")
    col_cycle = itertools.cycle(base_palette.colors)
    colours = dict(colours) if colours else {}
    for alt in winner_pool:
        colours.setdefault(alt, next(col_cycle))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (label, weights) in zip(axes, season_scenarios.items()):
        cfg = {**(base_cfg or {})}
        cfg["seasonality_method"] = "weighted_avg"
        cfg["seasonality_weights"] = weights

        # score table for tracked alts
        score_tbl = {alt: [] for alt in winner_pool}
        winner_per_alpha = []

        for alpha in alphas:
            cfg["outer_orness"] = float(alpha)
            scores = agg_with_alpha(csv_path, alpha, cfg)

            # store tracked scores
            for alt in winner_pool:
                score_tbl[alt].append(scores.loc[alt, "Global"])

            best_val = scores["Global"].max()
            winners = scores.index[
                scores["Global"] >= best_val - 1e-12
            ].tolist()
            winner_per_alpha.append(winners)

        # background coloured spans
        current = winner_per_alpha[0]
        span_start = alphas[0]
        for idx in range(1, len(alphas)):
            if set(winner_per_alpha[idx]) != set(current):
                alt_key = list(current)[0]
                if alt_key not in colours:
                    colours[alt_key] = next(col_cycle)
                ax.axvspan(
                    span_start, alphas[idx],
                    color=colours[alt_key], alpha=0.25
                )
                span_start = alphas[idx]
                current = winner_per_alpha[idx]
        # last span
        ax.axvspan(
            span_start, alphas[-1],
            color=colours[list(current)[0]], alpha=0.25
        )

        # overlay score curves (no markers)
        for alt, y in score_tbl.items():
            ax.plot(alphas, y, linewidth=1.5, color=colours[alt], label=alt)

        ax.set_title(label, fontsize=11)
        ax.grid(True, linestyle=":")
        ax.set_xlabel("Orness")
        ax.set_ylabel("Aggregation score")

    # One global legend
    custom_lines = [plt.Line2D([0], [0], color=colours[a], lw=3) for a in winner_pool]
    fig.legend(custom_lines, winner_pool, ncol=len(winner_pool), loc="upper center", bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Winning solutions as orness varies (four seasonality preferences)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


# ================================================================
# PLOT  B (enhanced) — prettier grouped-bar chart with tie stars
# ================================================================
def plot_concept_profiles_pretty(
    csv_path: str | Path,
    alternatives: Sequence[str],
    cfg: Dict | None = None,
    outer_orness: float = 0.5,
    tie_tol: float = 0.0005,
    normalise: bool = True,
) -> None:
    """
    Horizontal grouped-bar chart of the six concept scores for *alternatives*.

    Improvements vs. the basic version
    ----------------------------------
    • horizontal orientation → labels never overlap  
    • best-in-concept bars outlined darker for instant “winner” cue  
    • exact values printed at bar tips (two decimals)  
    • faint grid makes relative heights legible  
    • ★ still marks that LOOWA had to look past the 1st column (tie)  

    Parameters
    ----------
    csv_path       : path to the normalised decision matrix
    alternatives   : list / tuple of alternative IDs to plot
    cfg            : optional config overriding DEFAULT_CONFIG
    outer_orness   : α used when computing concept scores
    tie_tol        : tolerance for “values equal” inside a block
    normalise      : if True (default) each concept is scaled 0-1 across
                     *all* plotted alts so contrasts pop out visually
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # ---- 1.  Assemble concept-level table via aggregate_matrix ----
    cfg_local = {**DEFAULT_CONFIG, **(cfg or {})}
    cfg_local["outer_orness"] = outer_orness
    concept_cols = ["Concurrency", "Size", "Risk",
                    "EnvImpact", "Closeness", "Seasonality"]

    scores = aggregate_matrix(csv_path, cfg_local)
    tbl = scores.loc[list(alternatives), concept_cols]

    # ---- 2.  Detect per-block ties (first two columns equal?) ----
    raw = pd.read_csv(csv_path).set_index("Alternative").round(3)   

    def tie_needed(row, cols):
        return len(cols) >= 2 and abs(row[cols[0]] - row[cols[1]]) <= tie_tol

    tie_flag = defaultdict(lambda: {c: False for c in concept_cols})
    for alt in alternatives:
        r = raw.loc[alt]
        tie_flag[alt]["Size"]       = tie_needed(r, cfg_local["size_cols"])
        tie_flag[alt]["Risk"]       = tie_needed(r, cfg_local["risk_cols"])
        tie_flag[alt]["EnvImpact"]  = tie_needed(r, cfg_local["env_cols"])
        tie_flag[alt]["Closeness"]  = tie_needed(r, cfg_local["closeness_cols"])

    # ---- 3.  Optional 0-1 normalisation per concept ----
    if normalise:
        tbl = (tbl - tbl.min()) / (tbl.max() - tbl.min() + 1e-12)

    # ---- 4.  Build the plot ----
    n_alt   = len(alternatives)
    n_conc  = len(concept_cols)
    bar_h   = 0.8 / n_alt                 # bar height inside each group
    y_ticks = np.arange(n_conc)

    palette = plt.cm.get_cmap("Set2", n_alt)
    fig, ax = plt.subplots(figsize=(10, 5 + 0.3 * n_alt))

    for k, alt in enumerate(alternatives):
        offsets = y_ticks + k * bar_h - 0.4 + bar_h / 2
        vals    = tbl.loc[alt]

        # draw bars
        bars = ax.barh(
            offsets, vals,
            height=bar_h,
            color=palette(k),
            edgecolor="black",
            label=alt,
            linewidth=0.7,
            alpha=0.9,
        )

        # annotate value + tie star
        for j, bar in enumerate(bars):
            concept = concept_cols[j]
            # numeric label
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar_h / 2,
                f"{vals[j]:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )
            # tie star
            if tie_flag[alt][concept]:
                ax.text(
                    bar.get_width() + 0.07,
                    bar.get_y() + bar_h / 2,
                    "★",
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="firebrick",
                )

    # ---- 5.  Highlight the best bar in each concept ----
    best_per_concept = tbl.idxmax()
    for j, conc in enumerate(concept_cols):
        best_alt = best_per_concept[conc]
        k        = alternatives.index(best_alt)
        offset   = y_ticks[j] + k * bar_h - 0.4 + bar_h / 2
        ax.barh(
            offset, tbl.loc[best_alt, conc],
            height=bar_h,
            edgecolor="black",
            linewidth=2.0,
            fill=False,
        )

    # ---- 6.  Cosmetics ----
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(concept_cols)
    ax.set_xlim(0, 1.05 if normalise else tbl.max().max() * 1.05)
    ax.invert_yaxis()                             # top concept on top
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlabel("concept score" +
                  (" (normalised 0-1)" if normalise else ""))
    ax.set_title("Concept strengths")
    ax.legend()
    plt.tight_layout()



# ================================================================
# PLOT  C (v2) — heat-map of LOOWA tie depth for *all* alternatives
# ================================================================
def plot_lexico_ties(
    csv_path: str | Path,
    alternatives: Sequence[str] | None = None,
    cfg: Dict | None = None,
    tie_tol: float = 1e-6,
    sort_by: str = "alpha",          # "alpha" | "depth" | "global"
    outer_orness_for_sort: float = 0.5,
) -> None:
    """
    Heat-map where each cell shows how many priority levels LOOWA inspected
    (tie depth) before a difference appeared inside the block.

    Parameters
    ----------
    csv_path : path-like
        Normalised decision matrix.
    alternatives : None | list of str
        • None  → include *all* alternatives in the file.  
        • list  → restrict to those rows (old behaviour).
    cfg : dict, optional
        Custom configuration overriding DEFAULT_CONFIG.
    tie_tol : float
        Two scores are considered tied if |v1 − v2| ≤ tie_tol.
    sort_by : str
        "alpha"  → alphabetical alt ID (default);  
        "depth"  → descending total tie depth;  
        "global" → descending global score at `outer_orness_for_sort`.
    outer_orness_for_sort : float
        Only used when sort_by == "global".
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    cfg_local = {**DEFAULT_CONFIG, **(cfg or {})}
    df_raw    = pd.read_csv(csv_path).set_index("Alternative").round(3)   

    # ---------- figure out which alternatives to display ----------
    if alternatives is None:
        alts = df_raw.index.tolist()
    else:
        alts = list(alternatives)

    # ---------- helper: how deep did we look? ----------
    def depth(row, cols):
        vals = row[cols].to_numpy(float)
        for i in range(1, len(vals)):
            if abs(vals[i] - vals[0]) > tie_tol:
                return i
        return len(vals)                 # fully tied

    concept_cols = ["Concurrency", "Size", "Risk",
                    "EnvImpact", "Closeness", "Seasonality"]

    depth_tbl = pd.DataFrame(
        0, index=alts, columns=concept_cols, dtype=int
    )

    for alt in alts:
        r = df_raw.loc[alt]
        depth_tbl.loc[alt, "Size"]       = depth(r, cfg_local["size_cols"])
        depth_tbl.loc[alt, "Risk"]       = depth(r, cfg_local["risk_cols"])
        depth_tbl.loc[alt, "EnvImpact"]  = depth(r, cfg_local["env_cols"])
        depth_tbl.loc[alt, "Closeness"]  = depth(r, cfg_local["closeness_cols"])
        # single-criterion blocks stay 0

    # ---------- optional sorting ----------
    if sort_by == "depth":
        depth_tbl["__sum"] = depth_tbl.sum(axis=1)
        depth_tbl = depth_tbl.sort_values("__sum", ascending=False).drop("__sum", axis=1)
    elif sort_by == "global":
        cfg_tmp = {**cfg_local, "outer_orness": outer_orness_for_sort}
        global_scores = aggregate_matrix(csv_path, cfg_tmp)["Global"]
        depth_tbl["__g"] = global_scores
        depth_tbl = depth_tbl.sort_values("__g", ascending=False).drop("__g", axis=1)
    else:  # alphabetical
        depth_tbl = depth_tbl.sort_index()

    # ---------- heat-map ----------
    cmap = plt.cm.get_cmap("Reds", depth_tbl.values.max() + 1)
    h   = 0.3 * len(depth_tbl) + 3
    fig, ax = plt.subplots(figsize=(9, h))

    im = ax.imshow(depth_tbl, cmap=cmap, vmin=0, vmax=depth_tbl.values.max())

    # annotate each cell
    for i in range(len(depth_tbl)):
        for j in range(len(concept_cols)):
            v = depth_tbl.iat[i, j]
            txt = "—" if v == 0 else str(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    ax.set_xticks(np.arange(len(concept_cols)))
    ax.set_xticklabels(concept_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(depth_tbl)))
    ax.set_yticklabels(depth_tbl.index)

    cbar = fig.colorbar(
        im, ax=ax, shrink=0.8, pad=0.02,
        ticks=range(0, depth_tbl.values.max() + 1)
    )
    cbar.set_label("tie depth (number of columns consulted)")

    ax.set_title("LOOWA lexicographic tie depth across all alternatives")
    plt.tight_layout()
    plt.show()




# -------------------------------------------------------------------
# 5. Command-line entry
# -------------------------------------------------------------------

def main() -> None:
    """
    Example pipeline:
      1) aggregate once (outer_orness = 0.5)
      2) show top-5 alternatives
      3) run sensitivity analysis for alpha = 0.3, 0.5, 0.7
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
    #main()


    # #4 Bis:----------------------------------------
    # csv_path = "decision_matrix_expanded.csv"

    # # --- get just the set of winners across all alpha ---
    # winners, _ = sweep_orness_winners(csv_path)
    # print("Overall winner pool:", winners)

    # # --- or, get the alpha-by-winner mapping as well ---
    # winners, alpha_map = sweep_orness_winners(csv_path, return_mapping=True)
    # print("\nWinner pool:", winners)
    # for alt, alphas in alpha_map.items():
    #     print(f"{alt} wins for alpha in {sorted(alphas)}")


    #4 Ter:-----------------------------------------

    csv = "decision_matrix_expanded.csv"

    # pool you reported:
    winners = {"T1_D700_S33", "T3_D700_S73", "T3_D700_S33"}

    # # 1.  Four-panel winner-region diagram
    # plot_winner_regions_four(csv, winners, outer_orness_grid=101)

    # plt.show()
    # # 2.  Concept bar chart with tie indicators
    # plot_concept_profiles_pretty(csv, sorted(winners), outer_orness=0.5)
    
    # plt.show()
    plot_lexico_ties(csv)
    plt.show()
