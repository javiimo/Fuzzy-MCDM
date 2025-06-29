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

from fuzzy_var import FuzzyVariable, tconorm_aggregate

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
    y_scaled = tgt_miny + (y_raw - src_miny) * scale_y / 2 + 2.6

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
# #### Plots ################################################################
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def _to_points_gdf(points_xy: np.ndarray,
                   france_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame (WGS-84) from already *scaled* lon/lat pairs."""
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(points_xy[:, 0], points_xy[:, 1]),
        crs=france_gdf.crs
    )


# ---------------------------------------------------------------------------
# ❶ 2-D France  |  regions + parks + interventions
# ---------------------------------------------------------------------------
def plot_france_regions_parks_interventions(
    raw_points: np.ndarray,
    point_keys: list[str],
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
    parks_csv_path: str | Path = PARKS_CSV_PATH,
    marker_kw=None,
) -> None:
    """Same as before, but uses *markersize* instead of *s*."""
    marker_kw = marker_kw or dict(marker="o",
                                  markersize=50,
                                  edgecolor="k",
                                  zorder=4)

    france_gdf = _load_french_regions(france_regions_path)
    parks_gdf  = load_national_parks(parks_csv_path)

    pts_scaled_deg = _scale_points_to_france(raw_points, france_gdf)
    pts_gdf        = _to_points_gdf(pts_scaled_deg, france_gdf)

    fig, ax = plt.subplots(figsize=(10, 10))
    france_gdf.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=.5)
    parks_gdf.plot(ax=ax, facecolor="forestgreen", alpha=.4, edgecolor="none")

    pts_gdf.plot(ax=ax, **marker_kw)

    for key, (x, y) in zip(point_keys, pts_scaled_deg):
        ax.text(x, y, key, fontsize=7, ha="left", va="bottom", color="blue")

    ax.set_title("French regions + natural parks + interventions")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ❷ 2-D France  |  gradient = fuzzy proximity (intervention → parks)
# ---------------------------------------------------------------------------
def plot_intervention_park_gradient(
    raw_points: np.ndarray,
    point_keys: list[str],
    park_memberships: MembershipMatrices,
    s_norm,
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
    parks_csv_path: str | Path = PARKS_CSV_PATH,
    cmap = plt.get_cmap("coolwarm"),          # avoid Matplotlib 3.7 deprecation
) -> None:
    """Colour each intervention from blue→red by fuzzy proximity to parks."""
    france_gdf = _load_french_regions(france_regions_path)
    parks_gdf  = load_national_parks(parks_csv_path)

    pts_scaled_deg = _scale_points_to_france(raw_points, france_gdf)
    pts_gdf        = _to_points_gdf(pts_scaled_deg, france_gdf)

    df_close = park_memberships["close"]
    μ_close  = tconorm_aggregate(df_close, s_norm)

    colours = cmap(μ_close)

    fig, ax = plt.subplots(figsize=(10, 10))
    france_gdf.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=.5)
    parks_gdf.plot(ax=ax, facecolor="forestgreen", alpha=.4, edgecolor="none")

    pts_gdf.plot(ax=ax, marker="o", markersize=80,
                 color=colours, edgecolor="k", zorder=4)

    for key, (x, y) in zip(point_keys, pts_scaled_deg):
        ax.text(x, y, key, fontsize=7, ha="left", va="bottom")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, shrink=.7, pad=.02,
                 label=r"aggregated $\,\mu_{\mathrm{close}}$ (0 far → 1 close)")

    ax.set_title("Fuzzy proximity of interventions to parks")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ❸ & ❹  3-D fuzzy-distance “surfaces” (radial symmetry)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------- 3-D helpers (replace the previous ones) -------------------------
# ---------------------------------------------------------------------------
from shapely.geometry import Polygon, MultiPolygon
from matplotlib import colormaps as _cm


def _plot_poly_edges_3d(ax, geoms, *, z=0, color="black", lw=0.4):
    """Draw every outer + inner ring of (Multi)Polygons on the plane z = const."""
    def _draw_single(poly: Polygon):
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, zs=z, zdir="z", color=color, linewidth=lw)
        for hole in poly.interiors:
            xs, ys = hole.xy
            ax.plot(xs, ys, zs=z, zdir="z", color=color, linewidth=lw)

    for g in geoms:
        if isinstance(g, Polygon):
            _draw_single(g)
        elif isinstance(g, MultiPolygon):
            for poly in g.geoms:
                _draw_single(poly)


def _make_radial_surfaces(france_gdf: gpd.GeoDataFrame,
                          fuzzy_var: FuzzyVariable,
                          centre: Point | None = None,
                          n: int = 300) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return {term: (X, Y, Z)}.  Each *Z* is the membership matrix for that term,
    so every term becomes its own surface (alpha blending will show intersections).
    """
    # ---- generate a lon-lat grid ------------------------------------------------
    minx, miny, maxx, maxy = france_gdf.total_bounds
    lon = np.linspace(minx, maxx, n)
    lat = np.linspace(miny, maxy, n)
    X, Y = np.meshgrid(lon, lat)

    # ---- choose centre ----------------------------------------------------------
    if centre is None:
        try:                       # GeoPandas ≥0.14
            union_geom = france_gdf.union_all()
        except AttributeError:     # fall-back for older versions
            union_geom = france_gdf.unary_union
        centre = union_geom.centroid

    # ---- Euclidean distance (projected) ----------------------------------------
    centre_m = _project_to_meters(np.array([[centre.x, centre.y]]))[0]
    grid_m   = _project_to_meters(np.column_stack([X.ravel(), Y.ravel()]))

    dists = np.hypot(grid_m[:, 0] - centre_m[0],
                     grid_m[:, 1] - centre_m[1]).reshape(X.shape)

    # ---- evaluate fuzzy-variable & pack -----------------------------------------
    μ = fuzzy_var(dists)                # dict(term → ndarray)
    return {term: (X, Y, Z) for term, Z in μ.items()}


# ---------------------------------------------------------------------------
# ---------- 3-D plot: fuzzy distance between interventions ------------------
# ---------------------------------------------------------------------------
def plot_3d_fuzzy_distance_interventions(
    fuzzy_var: FuzzyVariable,
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
    elev: int = 55, azim: int = 45,
    alpha: float = 0.55,
) -> None:
    """Multiple semi-transparent surfaces (one per term) + French regions."""
    france_gdf = _load_french_regions(france_regions_path)
    surfaces   = _make_radial_surfaces(france_gdf, fuzzy_var)

    # colour maps to cycle through
    cmaps = ["Reds", "Greens", "Blues", "Purples", "Oranges", "Greys"]

    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")

    # plot each membership surface
    for i, (term, (X, Y, Z)) in enumerate(surfaces.items()):
        cmap = _cm.get_cmap(cmaps[i % len(cmaps)])
        surf = ax.plot_surface(X, Y, Z,
                               cmap=cmap,
                               linewidth=0,
                               antialiased=False,
                               alpha=alpha)
        # add colour-bar per surface (small, stacked)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cbar = plt.colorbar(m, ax=ax, fraction=.02, pad=.02)
        cbar.set_label(term, rotation=90)

    # political outline on ground plane
    _plot_poly_edges_3d(ax, france_gdf.geometry, z=0, color="black")

    ax.set_title("Fuzzy radial distance between interventions (one surface per term)")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ---------- 3-D plot: fuzzy distance to parks -------------------------------
# ---------------------------------------------------------------------------
def plot_3d_fuzzy_distance_parks(
    fuzzy_var: FuzzyVariable,
    *,
    france_regions_path: str | Path = FRANCE_JSON_PATH,
    parks_csv_path: str | Path = PARKS_CSV_PATH,
    elev: int = 55, azim: int = 45,
    alpha: float = 0.55,
) -> None:
    """Multiple semi-transparent surfaces (one per term) + French national parks."""
    france_gdf = _load_french_regions(france_regions_path)
    parks_gdf  = load_national_parks(parks_csv_path)
    surfaces   = _make_radial_surfaces(france_gdf, fuzzy_var)

    cmaps = ["Reds", "Greens", "Blues", "Purples", "Oranges", "Greys"]

    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")

    for i, (term, (X, Y, Z)) in enumerate(surfaces.items()):
        cmap = _cm.get_cmap(cmaps[i % len(cmaps)])
        surf = ax.plot_surface(X, Y, Z,
                               cmap=cmap,
                               linewidth=0,
                               antialiased=False,
                               alpha=alpha)
        m = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cbar = plt.colorbar(m, ax=ax, fraction=.02, pad=.02)
        cbar.set_label(term, rotation=90)

    # park borders (no call to .plot(), we draw manually)
    _plot_poly_edges_3d(ax, parks_gdf.geometry, z=0, color="forestgreen")

    ax.set_title("Fuzzy radial distance to national parks (one surface per term)")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()



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

    import pickle
    with open('interv_mems.pkl', 'wb') as f:
        pickle.dump(interv_mems, f)
    with open('park_mems.pkl', 'wb') as f:
        pickle.dump(park_mems, f)

    

    # # Summary of all terms
    # print("--- Fuzzy membership summary ---")
    # for label, mems in [("Intervention×Intervention", interv_mems), ("Intervention×Park", park_mems)]:
    #     print(f"{label} terms:")
    #     for term, df in mems.items():
    #         print(f"  • {term:10s}: shape={df.shape}, values ∈ [{df.values.min():.3f}, {df.values.max():.3f}]")

    # # === Plot heatmaps for all fuzzy terms ===
    # import matplotlib.pyplot as plt

    # # # Intervention × Intervention heatmaps
    # # for term, df in interv_mems.items():
    # #     plt.figure(figsize=(8, 6))
    # #     plt.imshow(df.values, aspect='auto')
    # #     plt.title(f"Heatmap of '{term}' membership (intervention × intervention)")
    # #     plt.xticks(range(len(point_keys)), point_keys, rotation=90, fontsize=6)
    # #     plt.yticks(range(len(point_keys)), point_keys, fontsize=6)
    # #     plt.colorbar(label='Membership degree')
    # #     plt.tight_layout()
    # #     plt.show()

    # # # Intervention × Park heatmaps
    # # park_keys = list(park_mems[next(iter(park_mems))].columns)
    # # for term, df in park_mems.items():
    # #     plt.figure(figsize=(8, 6))
    # #     plt.imshow(df.values, aspect='auto')
    # #     plt.title(f"Heatmap of '{term}' membership (intervention × park)")
    # #     plt.xticks(range(len(park_keys)), park_keys, rotation=90, fontsize=6)
    # #     plt.yticks(range(len(point_keys)), point_keys, fontsize=6)
    # #     plt.colorbar(label='Membership degree')
    # #     plt.tight_layout()
    # #     plt.show()

    # 1️⃣ basic map with everything
    plot_france_regions_parks_interventions(pts, point_keys)

    # 2️⃣ gradient map  (use any t-conorm you like)
    from fuzzy_var import s_norm_max
    plot_intervention_park_gradient(pts, point_keys, park_mems, s_norm_max)

    # # 3️⃣ 3-D radial fuzzy distance between interventions
    # plot_3d_fuzzy_distance_interventions(fuzzy_interv)

    # # 4️⃣ 3-D radial fuzzy distance to parks
    # plot_3d_fuzzy_distance_parks(fuzzy_parks)
