import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Load the French regions GeoJSON file downloaded from GADM
france_gdf = gpd.read_file("france.json")

# Inspect columns and CRS
print(france_gdf.columns)
print(france_gdf.crs)

# Load the saved points and the intervention names (keys)
points = np.load("points.npy")           # shape (n_points, 2)
keys = np.load("points_keys.npy")         

# Separate the x and y coordinates from the raw points
x_points = points[:, 0]
y_points = points[:, 1]

# Calculate the bounding box of the French map
minx, miny, maxx, maxy = france_gdf.total_bounds

# Calculate the bounding box of your raw MDS coordinates
mds_minx, mds_miny, mds_maxx, mds_maxy = x_points.min(), y_points.min(), x_points.max(), y_points.max()

# Compute scaling factors so that the relative distances are preserved
scale_x = (maxx - minx) / (mds_maxx - mds_minx)
scale_y = (maxy - miny) / (mds_maxy - mds_miny)

# Scale the points to fit within the French map bounding box
# (The constants in the translation adjust the location within the bounding box)
x_points_scaled = minx + (x_points - mds_minx) * scale_x / 2 + 4
y_points_scaled = miny + (y_points - mds_miny) * scale_y / 2 + 2.5

# Create a DataFrame and then a GeoDataFrame for the scaled points
mds_points_df = pd.DataFrame({'x': x_points_scaled, 'y': y_points_scaled})
mds_points_gdf = gpd.GeoDataFrame(
    mds_points_df,
    geometry=gpd.points_from_xy(mds_points_df.x, mds_points_df.y),
    crs=france_gdf.crs  # ensure CRS consistency
)

# Perform a spatial join to see which region each point falls in.
# Assumes that the region name is in the column 'NAME_1' of france_gdf.
points_in_regions = gpd.sjoin(
    mds_points_gdf,
    france_gdf,
    how="left",
    predicate="within"
)
print(points_in_regions.head())

# Add the region information to the mds_points_gdf.
# Points that do not fall within any region will have NaN.
mds_points_gdf["region"] = points_in_regions["NAME_1"].values

# Create new keys formatted as I1, I2, I3, ...
# Create new keys by extracting numbers and formatting as I1, I2, etc.
new_keys = []
for key in keys:
    # Extract the number from strings like "Intervention_1"
    num = key.split('_')[1]
    new_keys.append(f"I{num}")


# Prepare the plot
fig, ax = plt.subplots(figsize=(8, 8))
france_gdf.plot(ax=ax, color='white', edgecolor='black')  # Plot map of France

# Create a color mapping for each unique region (excluding NaN)
unique_regions = mds_points_gdf["region"].dropna().unique()
cmap = plt.cm.get_cmap('tab20', len(unique_regions))
region_color_map = {region: cmap(i) for i, region in enumerate(unique_regions)}

# Plot points by region
for region in unique_regions:
    subset = mds_points_gdf[mds_points_gdf["region"] == region]
    subset.plot(ax=ax, color=region_color_map[region], markersize=50, label=region)

# Plot any points that did not fall within a region in gray
no_region = mds_points_gdf[mds_points_gdf["region"].isna()]
if not no_region.empty:
    no_region.plot(ax=ax, color='gray', markersize=50, label='No Region')

# Label each point with its corresponding new key
for i, (x, y) in enumerate(zip(mds_points_gdf['x'], mds_points_gdf['y'])):
    ax.text(x, y, new_keys[i], fontsize=9, color='blue', ha='right', va='bottom')

# Add a legend for the regions
ax.legend(title="French Regions")
plt.show()





####################################################
# Fuzzy distance matrix
####################################################

# Create an adjacency dictionary keyed by region name. So we know which regions touch and we know which intervention is in each region.
region_adjacency = {r["NAME_1"]: set() for idx, r in france_gdf.iterrows()} # Key = the values of the column NAME_1 of each row
# Value = Empty set

for i, region1 in france_gdf.iterrows():
    for j, region2 in france_gdf.iterrows():
        if i != j:
            # Check if the two region polygons touch
            if region1.geometry.touches(region2.geometry):
                region_adjacency[region1["NAME_1"]].add(region2["NAME_1"])


import numpy as np
import pandas as pd

# Number of interventions
n = len(new_keys)

# Prepare an empty array of object type (so we can store strings)
dist_matrix = np.empty((n, n), dtype=object)

# Extract the region name for each intervention in a list
intervention_regions = mds_points_gdf["region"].tolist()

# Build the distance matrix
for i in range(n):
    region_i = intervention_regions[i]
    for j in range(n):
        region_j = intervention_regions[j]

        # Handle cases where region might be NaN
        # (for points that fall outside of known regions)
        if pd.isna(region_i) or pd.isna(region_j):
            dist_matrix[i, j] = "far"  # or some special label
        else:
            if region_i == region_j:
                dist_matrix[i, j] = "close"
            elif region_j in region_adjacency[region_i]:
                dist_matrix[i, j] = "medium"
            else:
                dist_matrix[i, j] = "far"

# Optionally convert to a DataFrame for readability
dist_df = pd.DataFrame(dist_matrix, index=new_keys, columns=new_keys)
print(dist_df)


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Assume dist_df is already defined with "close", "medium", "far"
label_to_num = {"close": 0, "medium": 1, "far": 2}
num_matrix = dist_df.replace(label_to_num).values

# Create a custom colormap
cmap = mcolors.ListedColormap(["green", "yellow", "red"])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(10, 8))
sns.heatmap(
    num_matrix,
    cmap=cmap,
    norm=norm,
    xticklabels=False,  # Remove x labels
    yticklabels=False,  # Remove y labels
    cbar=False          # Remove default color bar
)

# Add a general label instead of specifying each intervention
plt.xlabel("Interventions")
plt.ylabel("Interventions")

# Create a simple legend mapping colors to labels
patches = [
    mpatches.Patch(color="green",  label="close"),
    mpatches.Patch(color="yellow", label="medium"),
    mpatches.Patch(color="red",    label="far")
]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

plt.title("Distance Matrix Heatmap")
plt.tight_layout()
plt.show()


