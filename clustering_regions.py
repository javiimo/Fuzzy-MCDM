import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def load_french_regions(file_path: str) -> gpd.GeoDataFrame:
    """Load French regions from a GeoJSON file and print basic info."""
    france_gdf = gpd.read_file(file_path)
    print("Columns:", france_gdf.columns)
    print("CRS:", france_gdf.crs)
    return france_gdf


def load_points_and_keys(points_file: str, keys_file: str):
    """Load intervention points and keys from numpy files."""
    points = np.load(points_file)           # shape (n_points, 2)
    keys = np.load(keys_file)
    return points, keys


def scale_points(points: np.ndarray, france_gdf: gpd.GeoDataFrame):
    """
    Scale raw points to fit within the bounding box of the French map.
    
    Returns scaled x and y coordinate arrays.
    """
    x_points = points[:, 0]
    y_points = points[:, 1]
    
    # Bounding box for France
    minx, miny, maxx, maxy = france_gdf.total_bounds
    # Bounding box for the raw MDS coordinates
    mds_minx, mds_miny, mds_maxx, mds_maxy = x_points.min(), y_points.min(), x_points.max(), y_points.max()
    
    # Compute scaling factors
    scale_x = (maxx - minx) / (mds_maxx - mds_minx)
    scale_y = (maxy - miny) / (mds_maxy - mds_miny)
    
    # Scale points with fixed translation constants
    x_points_scaled = minx + (x_points - mds_minx) * scale_x / 2 + 4
    y_points_scaled = miny + (y_points - mds_miny) * scale_y / 2 + 2.5
    
    return x_points_scaled, y_points_scaled


def create_points_geodf(x_points_scaled, y_points_scaled, crs) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from scaled x and y coordinates."""
    df = pd.DataFrame({'x': x_points_scaled, 'y': y_points_scaled})
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x, df.y),
        crs=crs  # ensure CRS consistency
    )
    return gdf


def assign_regions(points_gdf: gpd.GeoDataFrame, france_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Perform a spatial join to assign each point a French region.
    Points outside any region will have NaN.
    """
    points_in_regions = gpd.sjoin(
        points_gdf,
        france_gdf,
        how="left",
        predicate="within"
    )
    points_gdf["region"] = points_in_regions["NAME_1"].values
    print(points_gdf.head())
    return points_gdf


def format_keys(keys: np.ndarray) -> list:
    """
    Format intervention keys so that they appear as I1, I2, I3, etc.
    """
    new_keys = []
    for key in keys:
        if '_' in key:
            num = key.split('_')[1]
            new_keys.append(f"I{num}")
        else:
            new_keys.append(key)
    return new_keys


def plot_french_map(france_gdf: gpd.GeoDataFrame, points_gdf: gpd.GeoDataFrame, new_keys: list):
    """
    Plot the French map with intervention points colored by region and label each point.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    france_gdf.plot(ax=ax, color='white', edgecolor='black')  # French map

    # Create a color mapping for each unique region (excluding NaN)
    unique_regions = points_gdf["region"].dropna().unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_regions))
    region_color_map = {region: cmap(i) for i, region in enumerate(unique_regions)}

    # Plot points by region
    for region in unique_regions:
        subset = points_gdf[points_gdf["region"] == region]
        subset.plot(ax=ax, color=region_color_map[region], markersize=50, label=region)

    # Plot points with no region in gray
    no_region = points_gdf[points_gdf["region"].isna()]
    if not no_region.empty:
        no_region.plot(ax=ax, color='gray', markersize=50, label='No Region')

    # Label each point with its corresponding new key
    for i, (x, y) in enumerate(zip(points_gdf['x'], points_gdf['y'])):
        ax.text(x, y, new_keys[i], fontsize=9, color='blue', ha='right', va='bottom')

    ax.legend(title="French Regions")
    plt.show()


def compute_region_adjacency(france_gdf: gpd.GeoDataFrame) -> dict:
    """
    Create an adjacency dictionary keyed by region name.
    Each region maps to a set of neighboring regions (i.e., regions whose polygons touch).
    """
    region_adjacency = {r["NAME_1"]: set() for idx, r in france_gdf.iterrows()}
    for i, region1 in france_gdf.iterrows():
        for j, region2 in france_gdf.iterrows():
            if i != j:
                if region1.geometry.touches(region2.geometry):
                    region_adjacency[region1["NAME_1"]].add(region2["NAME_1"])
    return region_adjacency


def compute_fuzzy_distance_matrix(new_keys: list, points_gdf: gpd.GeoDataFrame, region_adjacency: dict) -> pd.DataFrame:
    """
    Build a fuzzy distance matrix comparing interventions.
    Matrix labels are:
        - "close"  if interventions are in the same region,
        - "medium" if interventions are in adjacent regions,
        - "far"    otherwise (including if region info is missing).
    """
    n = len(new_keys)
    dist_matrix = np.empty((n, n), dtype=object)
    intervention_regions = points_gdf["region"].tolist()

    for i in range(n):
        region_i = intervention_regions[i]
        for j in range(n):
            region_j = intervention_regions[j]
            if pd.isna(region_i) or pd.isna(region_j):
                dist_matrix[i, j] = "far"
            else:
                if region_i == region_j:
                    dist_matrix[i, j] = "close"
                elif region_j in region_adjacency.get(region_i, set()):
                    dist_matrix[i, j] = "medium"
                else:
                    dist_matrix[i, j] = "far"

    dist_df = pd.DataFrame(dist_matrix, index=new_keys, columns=new_keys)
    print(dist_df)
    return dist_df


def plot_heatmap(dist_df: pd.DataFrame):
    """
    Plot a heatmap of the fuzzy distance matrix.
    Colors: green for "close", yellow for "medium", red for "far".
    """
    label_to_num = {"close": 0, "medium": 1, "far": 2}
    num_matrix = dist_df.replace(label_to_num).values

    # Create a custom colormap
    cmap = mcolors.ListedColormap(["green", "yellow", "red"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 8))
    sns.heatmap(num_matrix,
                cmap=cmap,
                norm=norm,
                xticklabels=False,
                yticklabels=False,
                cbar=False)
    plt.xlabel("Interventions")
    plt.ylabel("Interventions")
    
    # Create a legend mapping colors to labels
    patches = [
        mpatches.Patch(color="green", label="close"),
        mpatches.Patch(color="yellow", label="medium"),
        mpatches.Patch(color="red", label="far")
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.title("Distance Matrix Heatmap")
    plt.tight_layout()
    plt.show()


def plot_franceWparks():
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely import wkt

    # Load French regions (already EPSG:4326).
    france_gdf = load_french_regions("france.json")  
    print("France CRS:", france_gdf.crs)

    # Read the CSV containing park polygons. Convert WKT -> geometry.
    df = pd.read_csv("geojsons_nat_parks/pnr_polygon.csv")
    df["geometry"] = df["the_geom"].apply(wkt.loads)
    parks_gdf = gpd.GeoDataFrame(df, geometry="geometry")

    # The parks bounding box is large, indicating Web Mercator (EPSG:3857).
    # Assign that CRS explicitly (instead of EPSG:4326).
    parks_gdf.crs = "EPSG:3857"

    # Now reproject the parks to match France's EPSG:4326.
    parks_gdf = parks_gdf.to_crs(france_gdf.crs)

    # Plot both so they coincide.
    fig, ax = plt.subplots(figsize=(12, 8))
    france_gdf.plot(ax=ax, color="white", edgecolor="black")
    parks_gdf.plot(ax=ax, alpha=0.5, color="green")

    plt.title("National Parks of France")
    plt.axis("equal")
    plt.show()


def main():
    # Load French regions and intervention points
    france_gdf = load_french_regions("france.json")
    points, keys = load_points_and_keys("points.npy", "points_keys.npy")
    
    # Scale points to fit the French map and create a GeoDataFrame
    x_scaled, y_scaled = scale_points(points, france_gdf)
    points_gdf = create_points_geodf(x_scaled, y_scaled, france_gdf.crs)
    
    # Perform spatial join to assign regions
    points_gdf = assign_regions(points_gdf, france_gdf)
    
    # Format intervention keys (e.g., "Intervention_1" -> "I1")
    new_keys = format_keys(keys)
    
    # Plot the French map with intervention points and labels
    plot_french_map(france_gdf, points_gdf, new_keys)
    
    # Compute fuzzy distance matrix between interventions
    region_adjacency = compute_region_adjacency(france_gdf)
    dist_df = compute_fuzzy_distance_matrix(new_keys, points_gdf, region_adjacency)
    
    # Plot a heatmap of the distance matrix
    plot_heatmap(dist_df)

    plot_franceWparks()

if __name__ == "__main__":
    main()
