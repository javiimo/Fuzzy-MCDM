# ================= decision_analysis.py  (epsilon-lexico edition) ==============
#
# Implements:
#   • Epsilon-lexicographic aggregation inside four concept blocks
#     (strict hierarchy, ties broken with ε-radix weights)
#   • Seasonality block: weighted avg or OWA (unchanged)
#   • Classic OWA across the six concept scores (unchanged)
#   • Diagnostics + three graphics (A, B, C) – Plot C updated for ε-method
#
# ------------------------------------------------------------------------------

from __future__ import annotations
import numpy  as np
import pandas as pd
from pathlib import Path
from typing import Dict, Sequence, Iterable, Set, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Generic helpers (OWA) – unchanged
# -------------------------------------------------------------------

def owa_weights(n: int, orness: float) -> np.ndarray:
    if orness <= 0.0:                     # Min
        w = np.zeros(n); w[-1] = 1.0; return w
    if orness >= 1.0:                     # Max
        w = np.zeros(n); w[0]  = 1.0; return w
    a   = max(1e-6, (2.0 / orness) - 2.0)          # Yager quantifier
    idx = np.arange(1, n + 1)
    w   = (idx / n) ** a - ((idx - 1) / n) ** a
    return w / w.sum()

def owa(values: np.ndarray, orness: float) -> float:
    w = owa_weights(len(values), orness)
    return float(np.dot(np.sort(values)[::-1], w))

# -------------------------------------------------------------------
# 2.  ε-lexicographic aggregation inside a block
# -------------------------------------------------------------------

# ==== Δ 2025-06-18 epsilon-lexicographic update ====
def epsilon_lexico_block(
    block_df : pd.DataFrame,
    eps_base : float  = 1e-3,
    tie_tol  : float  = 1e-6,
) -> pd.Series:
    """
    Return a Series with the ε-lexicographic score of *every* alternative in
    `block_df` (columns are already in priority order).

    • If an alternative’s first component is unique → score = C1
    • If it is tied              → score = C1 + Σ Cj · (eps_base**(j-1))/2
    """
    first = block_df.iloc[:, 0]
    alts  = block_df.index
    scores = pd.Series(index=alts, dtype=float)

    # group rows that tie on the first priority value
    processed: set[str] = set()
    for alt in alts:
        if alt in processed:
            continue
        val0 = first.at[alt]
        mask = np.isclose(first.values, val0, atol=tie_tol)
        group = first.index[mask]

        group_size = len(group)
        for g in group:
            vals  = block_df.loc[g].to_numpy(float)
            score = vals[0]
            if group_size > 1:                           # resolve the tie
                for j in range(1, len(vals)):
                    score += vals[j] * ((eps_base ** j) / 2.0)
            scores.at[g] = score
        processed.update(group)

    return scores
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 3. Configuration
# -------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    # priority lists -------------------------------------------------
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
    ],

    # Seasonality ----------------------------------------------------
    "seasonality_cols": ["winter-like", "summer-like", "is-like"],
    "seasonality_method": "weighted_avg",        # or "owa"
    "seasonality_weights": [0.4, 0.4, 0.2],
    "seasonality_orness": 0.5,

    # Outer aggregation ---------------------------------------------
    "outer_orness": 0.5,

    # ε-lexicographic parameters ------------------------------------
    "eps_base": 1e-3,
    "tie_tol":  1e-6,
}

# -------------------------------------------------------------------
# 4.  Core aggregation pipeline  (index bug fixed)
# -------------------------------------------------------------------
def aggregate_matrix(csv_path: str | Path, cfg: Dict | None = None) -> pd.DataFrame:
    """
    Read the expanded decision‐matrix CSV and return a DataFrame whose rows are
    the alternatives and whose columns are the six concept scores **plus** the
    global OWA score.  Now uses the ‘Alternative’ column as the index
    throughout, so look-ups never fail.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}

    # -------- read & freeze Alternative as index --------
    df      = pd.read_csv(csv_path).round(6)
    df_idx  = df.set_index("Alternative")          # <- crucial fix

    # -------- ε-lexicographic scores per block ---------
    size_scores  = epsilon_lexico_block(df_idx[cfg["size_cols"]],
                                        cfg["eps_base"], cfg["tie_tol"])
    risk_scores  = epsilon_lexico_block(df_idx[cfg["risk_cols"]],
                                        cfg["eps_base"], cfg["tie_tol"])
    env_scores   = epsilon_lexico_block(df_idx[cfg["env_cols"]],
                                        cfg["eps_base"], cfg["tie_tol"])
    close_scores = epsilon_lexico_block(df_idx[cfg["closeness_cols"]],
                                        cfg["eps_base"], cfg["tie_tol"])

    # -------- Seasonality block (unchanged logic) -------
    if cfg["seasonality_method"] == "weighted_avg":
        w = np.asarray(cfg["seasonality_weights"], float)
        season_scores = pd.Series(
            df_idx[cfg["seasonality_cols"]].to_numpy(float) @ (w / w.sum()),
            index=df_idx.index,
        )
    else:
        season_scores = pd.Series(
            np.apply_along_axis(
                owa, 1,
                df_idx[cfg["seasonality_cols"]].to_numpy(float),
                cfg["seasonality_orness"],
            ),
            index=df_idx.index,
        )

    # -------- fuse the six concept scores ---------------
    records = []
    for alt, row in df_idx.iterrows():
        concept_vec = np.array([
            row["Highest Concurrency"],
            size_scores.at[alt],
            risk_scores.at[alt],
            env_scores.at[alt],
            close_scores.at[alt],
            season_scores.at[alt],
        ])
        records.append({
            "Alternative": alt,
            "Concurrency": row["Highest Concurrency"],
            "Size":        size_scores.at[alt],
            "Risk":        risk_scores.at[alt],
            "EnvImpact":   env_scores.at[alt],
            "Closeness":   close_scores.at[alt],
            "Seasonality": season_scores.at[alt],
            "Global":      owa(concept_vec, cfg["outer_orness"]),
        })

    return (
        pd.DataFrame(records)
          .set_index("Alternative")
          .sort_values("Global", ascending=False)
    )



# -------------------------------------------------------------------
# 5. Diagnostics
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
# 5-bis.  Orness sweep – find every winner over [0, 1]
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

# =======================================================================
#  PLOT A — winner-region (4 panels)  with *arbitrary-zoom* insets
#           • every alternative that ever wins in a panel is drawn
#           • inset shows an interpolated curve segment centred at the
#             winner-switch α, so the crossing point is visible even
#             between sampled α-grid points
#           • inset has scale bars:  Δα  (horizontal) and  Δscore (vertical)
# =======================================================================
from typing import Sequence, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# small helper ------------------------------------------------------------
def _agg_with_alpha(csv_path: str | Path, alpha: float, cfg: dict | None):
    return aggregate_matrix(csv_path, {**(cfg or {}), "outer_orness": alpha})


def plot_winner_regions_four(
    csv_path: str | Path,
    outer_orness_grid: int | Iterable[float] = 101,
    *,
    base_cfg: Dict | None = None,
    season_scenarios: Dict[str, Sequence[float]] | None = None,
    colours: Dict[str, str] | None = None,
    # zoom controls -------------------------------------------------------
    zoom_width: float = 0.02,     # width of the inset window in α-units
    zoom_fine: int = 300,         # number of interpolated points inside inset
    inset_size: float = 1.2,      # inches
) -> None:
    """
    Four panels (winter / summer / inter-season / equal).  For each panel the
    function:
        1.  finds *all* alternatives that are winners for any α in [0,1],
        2.  colours the background by the current winner,
        3.  plots every winner's score curve,
        4.  adds a magnifier inset at each α where the winner set changes.
    Parameters
    ----------
    zoom_width : float
        Horizontal width of the magnifier window (Δα).  Make it smaller for
        stronger magnification.
    zoom_fine : int
        How many interpolated points to draw inside each inset.
    inset_size : float
        Side length of the square inset window in inches.
    """
    # ------------- α-grid -------------------------------------------------
    if isinstance(outer_orness_grid, int):
        alphas = np.linspace(0.0, 1.0, outer_orness_grid)
    else:
        alphas = np.asarray(list(outer_orness_grid), float)

    # ------------- seasonality scenarios ---------------------------------
    season_scenarios = season_scenarios or {
        "Winter focus": (1.0, 0.0, 0.0),
        "Summer focus": (0.0, 1.0, 0.0),
        "Inter-season": (0.0, 0.0, 1.0),
        "Equal":        (1/3, 1/3, 1/3),
    }

    # ---------- colour palette (infinite cycle) ----------------------
    import itertools
    base_cycle = ["tab:green", "tab:orange", "tab:blue", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:olive",
                "tab:cyan", "tab:gray"]
    colour_cycle = itertools.cycle(base_cycle)      #  ← will never exhaust
    colours = dict(colours) if colours else {}

    # ------------- figure -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (label, weights) in zip(axes, season_scenarios.items()):

        # --- configuration for this panel --------------------------------
        cfg_p = {**(base_cfg or {}),
                 "seasonality_method": "weighted_avg",
                 "seasonality_weights": weights}

        # --- discover winner set & background spans ----------------------
        winner_per_alpha, alt_set = [], set()
        for alpha in alphas:
            scores = _agg_with_alpha(csv_path, alpha, cfg_p)
            best_val = scores["Global"].max()
            winners  = scores.index[scores["Global"] >= best_val - 1e-12].tolist()
            winner_per_alpha.append(winners)
            alt_set.update(winners)

        # assign colours
        for alt in sorted(alt_set):
            colours.setdefault(alt, next(colour_cycle))


        # background colouring
        cur = winner_per_alpha[0]
        span_start = alphas[0]
        for i in range(1, len(alphas)):
            if set(winner_per_alpha[i]) != set(cur):
                ax.axvspan(span_start, alphas[i],
                           color=colours[cur[0]], alpha=0.25)
                span_start, cur = alphas[i], winner_per_alpha[i]
        ax.axvspan(span_start, alphas[-1],
                   color=colours[cur[0]], alpha=0.25)

        # --- gather full score vectors for winners -----------------------
        score_tbl = {alt: [] for alt in alt_set}
        for alpha in alphas:
            scores = _agg_with_alpha(csv_path, alpha, cfg_p)
            for alt in alt_set:
                score_tbl[alt].append(scores.loc[alt, "Global"])

        # plot main curves
        for alt in sorted(alt_set):
            ax.plot(alphas, score_tbl[alt],
                    lw=1.6, color=colours[alt], label=alt)

        # --- magnifier insets at change points ---------------------------
        change_idx = [i for i in range(1, len(alphas))
                      if set(winner_per_alpha[i]) != set(winner_per_alpha[i-1])]

        for idx in change_idx:
            x_c = alphas[idx]
            a_min = max(0.0, x_c - zoom_width/2)
            a_max = min(1.0, x_c + zoom_width/2)
            fine_alpha = np.linspace(a_min, a_max, zoom_fine)

            axins = inset_axes(ax, width=inset_size, height=inset_size,
                               bbox_to_anchor=(x_c, np.mean([score_tbl[w][idx] for w in alt_set])),
                               bbox_transform=ax.transData, loc='center')
            axins.set_aspect('auto')

            # interpolated curves
            for alt in sorted(alt_set):
                interp_y = np.interp(fine_alpha, alphas, score_tbl[alt])
                axins.plot(fine_alpha, interp_y,
                           color=colours[alt], lw=1.2)

            axins.set_xticks([]); axins.set_yticks([])
            y_slice = [np.interp(fa, alphas, score_tbl[alt]).item()
                       for alt in alt_set for fa in fine_alpha]
            y_min, y_max = min(y_slice), max(y_slice)
            pad = (y_max - y_min)*0.1 or 0.02
            axins.set_xlim(a_min, a_max)
            axins.set_ylim(y_min - pad, y_max + pad)

            # --- scale bars -------------------------------------------
            # horizontal Δα
            scale_a = zoom_width * 0.4           # 40 % of window width
            axins.plot([a_min + 0.05*zoom_width,
                        a_min + 0.05*zoom_width + scale_a],
                       [y_min + pad*0.6]*2, 'k-', lw=1)
            axins.text(a_min + 0.05*zoom_width + scale_a/2,
                       y_min + pad*0.8,
                       f"Δα = {scale_a:.3f}",
                       ha='center', va='bottom', fontsize=6)

            # vertical Δscore
            scale_s = (y_max - y_min) * 0.4
            axins.plot([a_max - 0.1*zoom_width]*2,
                       [y_min + pad*0.6,
                        y_min + pad*0.6 + scale_s], 'k-', lw=1)
            axins.text(a_max - 0.12*zoom_width,
                       y_min + pad*0.6 + scale_s/2,
                       f"Δs = {scale_s:.2f}", rotation=90,
                       ha='right', va='center', fontsize=6)

            for s in axins.spines.values():
                s.set_edgecolor('k'); s.set_linewidth(0.8)

        # cosmetics
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Orness")
        ax.set_ylabel("Aggregation score")
        ax.grid(True, linestyle=":")

    # legend
    handles = [plt.Line2D([0], [0], color=colours[a], lw=3) for a in colours]
    fig.legend(handles, list(colours.keys()),
               ncol=min(5, len(colours)), loc="upper center",
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Winning solutions as orness varies (four seasonality preferences)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])






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



# -------------------------------------------------------------------
# Plot C – lexicographic tie depth **across alternatives**
# -------------------------------------------------------------------
# ==== Δ 2025-06-18 epsilon-lexicographic update ====
def _tie_depth_across(df_block: pd.DataFrame, alt: str, tie_tol: float) -> int:
    """
    Number of priority levels needed to uniquely identify `alt`
    against *all other* rows in `df_block`.
    """
    mask = np.ones(len(df_block), dtype=bool)
    depth = 0
    for j, col in enumerate(df_block.columns, start=1):
        mask &= np.isclose(df_block[col], df_block.at[alt, col], atol=tie_tol)
        if mask.sum() == 1:          # uniqueness reached
            depth = j
            break
    return depth if depth else len(df_block.columns)

def plot_lexico_ties(
    csv_path: str | Path,
    alternatives: Sequence[str] | None = None,
    cfg: Dict | None = None,
    sort_by: str = "depth",
) -> None:
    """
    Heat-map: each cell shows how many criteria were *actually needed* to break
    the tie in that block for each alternative (1 = unique on C1, … 5 = full tie).
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    df  = pd.read_csv(csv_path).set_index("Alternative").round(6)

    if alternatives is None:
        alternatives = list(df.index)

    # table [alt × concept]
    depth_tbl = pd.DataFrame(0, index=alternatives,
                             columns=["Concurrency","Size","Risk","Env","Close","Season"],
                             dtype=int)

    # Concurrency & Seasonality are single-criterion → depth = 1 always
    depth_tbl["Concurrency"] = 1
    depth_tbl["Season"]      = 1

    depth_tbl.rename(columns={"Env":"EnvImpact","Close":"Closeness"}, inplace=True)

    # multi-attribute blocks
    depth_funcs = {
        "Size":       cfg["size_cols"],
        "Risk":       cfg["risk_cols"],
        "EnvImpact":  cfg["env_cols"],
        "Closeness":  cfg["closeness_cols"],
    }
    for concept, cols in depth_funcs.items():
        sub = df[cols]
        for alt in alternatives:
            depth_tbl.at[alt, concept] = _tie_depth_across(sub, alt, cfg["tie_tol"])

    # optional ordering
    if sort_by == "depth":
        depth_tbl = depth_tbl.loc[depth_tbl.sum(axis=1).sort_values(ascending=False).index]
    else:
        depth_tbl = depth_tbl.sort_index()

    # ---- heat-map -------------------------------------------------
    vmax = depth_tbl.values.max()
    cmap = plt.cm.get_cmap("Blues", vmax+1)
    fig, ax = plt.subplots(figsize=(9, 0.35*len(depth_tbl)+3))
    im = ax.imshow(depth_tbl, cmap=cmap, vmin=1, vmax=vmax)

    for i, alt in enumerate(depth_tbl.index):
        for j, conc in enumerate(depth_tbl.columns):
            ax.text(j, i, depth_tbl.iat[i,j], ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(depth_tbl.columns)))
    ax.set_xticklabels(depth_tbl.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(depth_tbl)))
    ax.set_yticklabels(depth_tbl.index)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("criteria consulted to resolve tie")
    ax.set_title("ε-Lexicographic tie depth per concept")
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
    import matplotlib.pyplot as plt

    # Path to the (already expanded) decision matrix
    csv_path = "decision_matrix_expanded.csv"

    # ------------------------------------------------------------
    # 1)  Sweep orness α ∈ [0,1] and find the winner set
    # ------------------------------------------------------------
    winners, alpha_map = sweep_orness_winners(csv_path, return_mapping=True)
    print("Overall winner pool:", winners)
    for alt, alphas in alpha_map.items():
        print(f"{alt} wins for α in {sorted(alphas)}")

    # ------------------------------------------------------------
    # 2)  Four-panel diagram of winner regions over (α, alternative)
    # ------------------------------------------------------------
    plot_winner_regions_four(csv_path, zoom_width=0.001)
    plt.show()

    # ------------------------------------------------------------
    # 3)  Concept-profile bar chart with ★ tie indicators (α = 0.5)
    # ------------------------------------------------------------
    plot_concept_profiles_pretty(csv_path, sorted(winners), outer_orness=0.5)
    plt.show()

    # ------------------------------------------------------------
    # 4)  Heat-map of ε-lexicographic tie depth inside each block
    # ------------------------------------------------------------
    plot_lexico_ties(csv_path)
    plt.show()

