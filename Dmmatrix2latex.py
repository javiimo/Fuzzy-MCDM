"""
generate_matrix_tex.py  –  FINAL (v2)

Adds:
  • proper single-width vertical rules between every big category
  • struts so header words no longer touch the top \hline (and no gap
    appears between the \hline and vertical rules)
"""

from pathlib import Path
import pandas as pd

# ─────────────────────── 1.  Load & tidy the CSV ─────────────────────
csv_file = Path("decision_matrix_expanded.csv")
if not csv_file.exists():
    raise SystemExit("CSV file decision_matrix_expanded.csv not found.")

df = pd.read_csv(csv_file)

df = df.rename(columns={
    "Alternative":  "Solutions",
    "winter-like":  "Winter",
    "summer-like":  "Summer",
    "is-like":      "Inter-season",
})

order = [
    "Solutions", "Highest Concurrency",
    # Size
    "Size_small", "Size_mid_small", "Size_medium",
    "Size_mid_large", "Size_large",
    # Risk
    "Risk_low", "Risk_mid_low", "Risk_mid",
    "Risk_mid_high", "Risk_high",
    # Closeness
    "Closeness_close", "Closeness_mid_close", "Closeness_mid",
    "Closeness_mid_far", "Closeness_far",
    # Env impact
    "EnvImpact_close", "EnvImpact_mid_close", "EnvImpact_mid",
    "EnvImpact_mid_far", "EnvImpact_far",
    # Seasons
    "Winter", "Summer", "Inter-season",
]
df = df[order]

# escape underscores so LaTeX prints them literally
df["Solutions"] = df["Solutions"].str.replace("_", r"\_", regex=False)

# 3-decimal formatting for every numeric field
df = df.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))

# ─────────────────────── 2.  Column → colour prefix ──────────────────
def tint(base: str, pct: int) -> str:
    """Return \\cellcolor{<base>!<pct>}"""
    return rf"\cellcolor{{{base}!{pct}}}"

colour = {
    # Size – purple
    **{c: tint("SizeBase", p) for c, p in zip(
        ["Size_small","Size_mid_small","Size_medium",
         "Size_mid_large","Size_large"], (10, 20, 30, 40, 50))},
    # Risk – soft red
    **{c: tint("RiskBase", p) for c, p in zip(
        ["Risk_low","Risk_mid_low","Risk_mid",
         "Risk_mid_high","Risk_high"], (10, 20, 30, 40, 50))},
    # Closeness – cyan
    **{c: tint("CloseBase", p) for c, p in zip(
        ["Closeness_close","Closeness_mid_close","Closeness_mid",
         "Closeness_mid_far","Closeness_far"], (10, 20, 30, 40, 50))},
    # Env-impact – green
    **{c: tint("EnvBase", p) for c, p in zip(
        ["EnvImpact_close","EnvImpact_mid_close","EnvImpact_mid",
         "EnvImpact_mid_far","EnvImpact_far"], (10, 20, 30, 40, 50))},
    # Seasons – fixed pastels
    "Winter":        r"\cellcolor{Winter}",
    "Summer":        r"\cellcolor{Summer}",
    "Inter-season":  r"\cellcolor{Inter}",
}

# ─────────────────────── 3.  LaTeX document pieces ───────────────────
preamble = r"""
\documentclass{article}
\usepackage[a4paper]{geometry}
\usepackage[table]{xcolor}
\usepackage{multirow,rotating,array}

% pastel bases
\definecolor{SizeBase}{HTML}{9C27B0}
\definecolor{RiskBase}{HTML}{EF5350}
\definecolor{CloseBase}{HTML}{03A9F4}
\definecolor{EnvBase}{HTML}{4CAF50}
\definecolor{Winter}{HTML}{BBDEFB}
\definecolor{Summer}{HTML}{FFE0B2}
\definecolor{Inter}{HTML}{C8E6C9}

\begin{document}
"""

# ––– header rows –––
#   • each \multicolumn ends with ‘c|’  → puts back the missing right-hand rule
#   • \rule{0pt}{2.3ex} inside the multi-row cells lifts the text a little
group_header = r"""
\multirow{2}{*}{\rule{0pt}{2.3ex}\textbf{Solutions}} &
\multirow{2}{*}{\rule{0pt}{2.3ex}\shortstack{\textbf{Highest}\\\textbf{Concurrency}}} &
\multicolumn{5}{c|}{\textbf{Size Concurrency}} &
\multicolumn{5}{c|}{\textbf{Risk Concurrency}} &
\multicolumn{5}{c|}{\textbf{Closeness Concurrency}} &
\multicolumn{5}{c|}{\textbf{Environmental-Impact Concurrency}} &
\multicolumn{3}{c|}{\textbf{Seasonality}} \\[2pt]
"""

sub_header = (
    r"& & " +
    " & ".join(
        # Size labels
        [f"{tint('SizeBase',p)}{l}"   for p,l in zip(
            (10, 20, 30, 40, 50),
            ["small","mid-small","mid","mid-large","large"])] +
        # Risk labels
        [f"{tint('RiskBase',p)}{l}"   for p,l in zip(
            (10, 20, 30, 40, 50),
            ["low","mid-low","mid","mid-high","high"])] +
        # Closeness labels
        [f"{tint('CloseBase',p)}{l}"  for p,l in zip(
            (10, 20, 30, 40, 50),
            ["close","mid-close","mid","mid-far","far"])] +
        # Env-impact labels
        [f"{tint('EnvBase',p)}{l}"    for p,l in zip(
            (10, 20, 30, 40, 50),
            ["low","mid-low","mid","mid-high","high"])] +
        # Season labels
        [r"\cellcolor{Winter}Winter",
         r"\cellcolor{Summer}Summer",
         r"\cellcolor{Inter}Inter-season"]
    ) + r" \\"
)

table_start = r"""
\clearpage
\newgeometry{bottom=0.6cm}
\begin{sidewaystable}[!ht]
\thispagestyle{empty}
\centering
\scriptsize
\setlength\tabcolsep{3pt}
\renewcommand{\arraystretch}{1.5}

\begin{tabular}{|l|c|ccccc|ccccc|ccccc|ccccc|ccc|}
\hline
""" + group_header + sub_header + r"""
\hline
"""

# ––– body –––
body_rows = []
for _, row in df.iterrows():
    body_rows.append(
        " & ".join(colour.get(c, "") + row[c] for c in order) + r" \\"
    )
body = "\n".join(body_rows)

table_end = r"""
\hline
\end{tabular}

\caption{Decision-making matrix containing }
\end{sidewaystable}
\restoregeometry

\end{document}
"""

# ─────────────────────── 4.  Emit LaTeX ──────────────────────────────
print(preamble + table_start + body + table_end)
