from __future__ import annotations

"""clustering_regions.py

Clean‑room rewrite focused on **distance‑based fuzzy clustering** for French
interventions.  Plotting and UI concerns have been removed so the module now
contains **only data‑processing utilities** that can be imported or executed
from the CLI.

Key public helpers
------------------
- ``build_fuzzy_intervention_distance`` — pairwise intervention × intervention
  fuzzy membership matrices.
- ``build_fuzzy_intervention_park_distance`` — intervention × park fuzzy
  membership matrices.

Both helpers return a ``dict[str, pandas.DataFrame]`` where each key is a term
name from the supplied :class:`fuzzy_var.FuzzyVariable` (e.g. ``"close"``,
``"mid"``…) and each value is a DataFrame of membership degrees ∈ [0, 1].

The only external dependency beyond the scientific Python stack is
``france.json`` (French regions polygons) and, for park‑related utilities,
``geojsons_nat_parks/pnr_polygon.csv`` which must contain a WKT column
``the_geom``.
"""

from pathlib import Path
from typing import Dict, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt

from fuzzy_var import FuzzyVariable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

FRANCE_JSON_PATH = Path("france.json")
PARKS_CSV_PATH = Path("geojsons_nat_parks/pnr_polygon.csv")


def _load_french_regions(path: str | Path = FRANCE_JSON_PATH) -> gpd.GeoDataFrame:
    """Load metropolitan French regions in WGS‑84 (EPSG:4326)."""
    gdf = gpd.read_file(Path(path))
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return gdf


def _scale_points_to_france(points: np.ndarray, france_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Affine‑scale 2‑D *MDS* points to fit France's bounding box.

    The raw MDS configuration ("points") is assumed to be *relative* — typically
    in an arbitrary unit square.  We linearly map it onto France's bounding box
    so that later projections (to meters) give distances of the right order of
    magnitude.
    """
    x_raw, y_raw = points[:, 0], points[:, 1]

    # --- source & target bounding boxes -----------------------------------------------------
    src_minx, src_miny, src_maxx, src_maxy = x_raw.min(), y_raw.min(), x_raw.max(), y_raw.max()
    tgt_minx, tgt_miny, tgt_maxx, tgt_maxy = france_gdf.total_bounds

    # --- independent linear scales ----------------------------------------------------------
    scale_x = (tgt_maxx - tgt_minx) / (src_maxx - src_minx)
    scale_y = (tgt_maxy - tgt_miny) / (src_maxy - src_miny)

    x_scaled = tgt_minx + (x_raw - src_minx) * scale_x / 2 + 4  # empirical offsets as before
    y_scaled = tgt_miny + (y_raw - src_miny) * scale_y / 2 + 2.5

    return np.column_stack([x_scaled, y_scaled])


def _to_geodf(points_xy: np.ndarray, crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame given *lon/lat* pairs and their CRS."""
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(points_xy[:, 0], points_xy[:, 1]),
        crs=crs,
    )


_PROJECT_TO_M = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _project_to_meters(xy_deg: np.ndarray) -> np.ndarray:
    """Vectorised transformation *lon/lat°* -> *x/y m* (Web Mercator)."""
    x_m, y_m = _PROJECT_TO_M.transform(xy_deg[:, 0], xy_deg[:, 1])
    return np.column_stack([x_m, y_m])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MembershipMatrices = Dict[str, pd.DataFrame]


def build_fuzzy_intervention_distance(
    points: np.ndarray,
    point_keys: Sequence[str],
    fuzzy_dist: FuzzyVariable,
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
) -> MembershipMatrices:
    """Return fuzzy membership matrices **intervention × intervention**.

    Parameters
    ----------
    points
        Raw MDS coordinates, shape *(N, 2).*  These will be affinely scaled to
        France before metric projection.
    point_keys
        Sequence of human‑readable labels (length *N*).
    fuzzy_dist
        Fuzzy variable encoding the linguistic distance terms (``close``,
        ``mid‑close`` …).
    france_regions_path
        Path to *france.json* (needed only for bounding‑box scaling).

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``{"close": df_close, "mid": df_mid, …}``  Each DataFrame is
        square *(N × N)* with identical index/columns = *point_keys*.
    """
    if len(points) != len(point_keys):
        raise ValueError("'points' and 'point_keys' must be the same length")

    # ---------------------------------------------------------------------
    # 1. Affine‑scale then project to meters
    # ---------------------------------------------------------------------
    france_gdf = _load_french_regions(france_regions_path)
    pts_scaled_deg = _scale_points_to_france(points, france_gdf)
    pts_scaled_m = _project_to_meters(pts_scaled_deg)

    # ---------------------------------------------------------------------
    # 2. Pairwise Euclidean distances (in metres)
    # ---------------------------------------------------------------------
    diff_x = pts_scaled_m[:, None, 0] - pts_scaled_m[None, :, 0]
    diff_y = pts_scaled_m[:, None, 1] - pts_scaled_m[None, :, 1]
    dist_matrix = np.hypot(diff_x, diff_y)  # shape (N, N)

    # ---------------------------------------------------------------------
    # 3. Apply fuzzy variable term‑wise
    # ---------------------------------------------------------------------
    mem_dict_raw = fuzzy_dist(dist_matrix)  # {term: ndarray(N,N)}
    membership_matrices: MembershipMatrices = {
        term: pd.DataFrame(vals, index=point_keys, columns=point_keys)
        for term, vals in mem_dict_raw.items()
    }
    return membership_matrices


def load_national_parks(
    csv_path: str | Path = PARKS_CSV_PATH,
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
) -> gpd.GeoDataFrame:
    """Load French *Parcs naturels régionaux* polygons as GeoDataFrame WGS‑84."""
    df = pd.read_csv(Path(csv_path))
    if "the_geom" not in df.columns:
        raise ValueError("CSV must contain a 'the_geom' WKT column")

    gdf = gpd.GeoDataFrame(df, geometry=df["the_geom"].apply(wkt.loads), crs="EPSG:3857")
    france_gdf = _load_french_regions(france_regions_path)
    return gdf.to_crs(france_gdf.crs)


_DEF_PARK_NAME_COLS = ("name", "NAME", "nom", "NOM", "park_name")


def _park_names(gdf: gpd.GeoDataFrame) -> Sequence[str]:
    for col in _DEF_PARK_NAME_COLS:
        if col in gdf.columns:
            return gdf[col].astype(str).tolist()
    # fallback to numeric indices
    return gdf.index.astype(str).tolist()


def build_fuzzy_intervention_park_distance(
    points: np.ndarray,
    point_keys: Sequence[str],
    parks_gdf: gpd.GeoDataFrame,
    fuzzy_dist: FuzzyVariable,
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
) -> MembershipMatrices:
    """Return fuzzy membership matrices **intervention × park**.

    Rows = interventions, columns = parks.
    """
    if len(points) != len(point_keys):
        raise ValueError("'points' and 'point_keys' must be the same length")

    # ---------------------------------------------------------------------
    # 1. Prepare geometries (project to metres)
    # ---------------------------------------------------------------------
    france_gdf = _load_french_regions(france_regions_path)
    pts_scaled_deg = _scale_points_to_france(points, france_gdf)
    pts_gdf = _to_geodf(pts_scaled_deg, france_gdf.crs)

    # common metric CRS
    metric_crs = "EPSG:3857"
    pts_proj = pts_gdf.to_crs(metric_crs)
    parks_proj = parks_gdf.to_crs(metric_crs)

    # ---------------------------------------------------------------------
    # 2. Distance matrix (N_interventions × N_parks)
    # ---------------------------------------------------------------------
    n_i, n_p = len(point_keys), len(parks_proj)
    dist_ip = np.zeros((n_i, n_p))
    for j, park_geom in enumerate(parks_proj.geometry):
        # vectorised distance: returns Series (meters)
        dist_ip[:, j] = pts_proj.distance(park_geom).values

    # ---------------------------------------------------------------------
    # 3. Fuzzy membership
    # ---------------------------------------------------------------------
    mem_dict_raw = fuzzy_dist(dist_ip)  # {term: ndarray}

    park_names = _park_names(parks_gdf)
    membership_matrices: MembershipMatrices = {
        term: pd.DataFrame(vals, index=point_keys, columns=park_names)
        for term, vals in mem_dict_raw.items()
    }
    return membership_matrices


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def _load_numpy(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


if __name__ == "__main__":
    # Paths to your NumPy files and parks CSV
    points_path = "points_20250329_203043.npy"
    keys_path = "points_keys_20250329_203043.npy"
    parks_csv_path = "geojsons_nat_parks/pnr_polygon.csv"

    # Load data
    pts = _load_numpy(points_path)
    raw_keys = _load_numpy(keys_path)
    point_keys = [str(k) for k in raw_keys]

    # Define fuzzy variables (tweak thresholds as needed)
    from fuzzy_var import fuzz_dist
    fuzzy_interv = fuzz_dist(5_000, 15_000, 50_000, 100_000, 200_000)
    fuzzy_parks = fuzz_dist(1_000, 10_000, 20_000, 60_000, 150_000)

    # Load park geometries
    parks_gdf = load_national_parks(parks_csv_path)

    # Compute fuzzy membership matrices
    interv_mems = build_fuzzy_intervention_distance(pts, point_keys, fuzzy_interv)
    park_mems = build_fuzzy_intervention_park_distance(pts, point_keys, parks_gdf, fuzzy_parks)

    # Summary of all terms
    print("--- Fuzzy membership summary ---")
    for label, mems in [("Intervention×Intervention", interv_mems), ("Intervention×Park", park_mems)]:
        print(f"{label} terms:")
        for term, df in mems.items():
            print(f"  • {term:10s}: shape={df.shape}, values ∈ [{df.values.min():.3f}, {df.values.max():.3f}]")

    # === Plot heatmaps for all fuzzy terms ===
    import matplotlib.pyplot as plt

    # Intervention × Intervention heatmaps
    for term, df in interv_mems.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(df.values, aspect='auto')
        plt.title(f"Heatmap of '{term}' membership (intervention × intervention)")
        plt.xticks(range(len(point_keys)), point_keys, rotation=90, fontsize=6)
        plt.yticks(range(len(point_keys)), point_keys, fontsize=6)
        plt.colorbar(label='Membership degree')
        plt.tight_layout()
        plt.show()

    # Intervention × Park heatmaps
    park_keys = list(park_mems[next(iter(park_mems))].columns)
    for term, df in park_mems.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(df.values, aspect='auto')
        plt.title(f"Heatmap of '{term}' membership (intervention × park)")
        plt.xticks(range(len(park_keys)), park_keys, rotation=90, fontsize=6)
        plt.yticks(range(len(point_keys)), point_keys, fontsize=6)
        plt.colorbar(label='Membership degree')
        plt.tight_layout()
        plt.show()
