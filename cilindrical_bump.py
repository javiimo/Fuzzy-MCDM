import numpy as np
import matplotlib.pyplot as plt

# Assume off_center_bump function is defined as corrected previously:
def off_center_bump(x, y, a, b, c, d, R, k):
    """
    Calculates the value of the off-center bump function.
    Handles scalar or numpy array inputs for x, y.
    """
    # Ensure input are numpy arrays (even 0-d for scalars)
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate squared distances
    v = (x - c)**2 + (y - d)**2
    u = (x - a)**2 + (y - b)**2
    w = R**2 - u

    # Initialize output with zeros. Use w's shape/dtype for compatibility.
    z = np.zeros_like(w, dtype=float)

    # Identify points strictly inside the circle (use a small tolerance)
    inside_mask = w > 1e-15

    # Calculate the denominator (v + w)
    denominator = v + w

    # Create a mask for points that are BOTH inside AND have a valid denominator
    valid_calculation_mask = inside_mask & (denominator > 1e-15)

    # --- Perform calculation only where valid ---
    if np.any(valid_calculation_mask):
        numerator_valid = w[valid_calculation_mask]
        denominator_valid = denominator[valid_calculation_mask]
        result_valid = (numerator_valid / denominator_valid)**k
        z[valid_calculation_mask] = result_valid

    # Clip final result
    z = np.clip(z, 0, 1)

    # Return scalar if input was scalar, otherwise return the array
    if x.ndim == 0 and y.ndim == 0:
        return z.item()
    else:
        return z

# --- Function 1: Get Slice Along Line ---

def get_slice_along_line(r_values, a, b, c, d, R, k, e, f):
    """
    Calculates the bump's value along the line passing through (e, f)
    and (c, d), parameterized by distance r from (e, f).

    Args:
        r_values (np.ndarray): 1D array of distances from (e, f) along the line.
                               r=0 corresponds to the point (e, f).
                               Positive r is towards (c, d) if (e,f) != (c,d).
        a, b: Original bump support center.
        c, d: Original bump peak location.
        R: Original bump support radius.
        k: Original bump exponent.
        e, f: The reference point on the line (where r=0).

    Returns:
        np.ndarray: 1D array of z-values corresponding to r_values.
    """
    r_values = np.asarray(r_values)
    if r_values.ndim != 1:
        raise ValueError("r_values must be a 1D numpy array.")

    # Calculate direction vector from (e, f) to (c, d)
    dx = c - e
    dy = d - f
    dist_peak_ref = np.sqrt(dx**2 + dy**2)

    if dist_peak_ref < 1e-15:
        # (e, f) is the same as the peak (c, d).
        # The line direction is undefined. We can arbitrarily choose one,
        # e.g., positive x-axis, but it feels more consistent to just evaluate
        # at points radially outwards from (c, d) along x-axis for example.
        # Or, maybe simpler: evaluate the bump at the peak for all r?
        # Let's evaluate radially along x from the peak (c,d).
        print("Warning: Slice reference point (e, f) coincides with peak (c, d). "
              "Using radial distance from peak along x-axis.")
        x_points = c + r_values # Move along x-axis from peak
        y_points = d + np.zeros_like(r_values) # Stay at peak's y-level
    else:
        # Calculate the unit direction vector
        ux = dx / dist_peak_ref
        uy = dy / dist_peak_ref

        # Calculate the (x, y) coordinates corresponding to each r
        x_points = e + r_values * ux
        y_points = f + r_values * uy

    # Evaluate the original bump function at these points
    z_values = off_center_bump(x_points, y_points, a, b, c, d, R, k)

    return z_values

# --- Function 2: Revolved Bump Function ---

def revolved_bump(x, y, a, b, c, d, R, k, e, f):
    """
    Calculates the value of a function formed by revolving the slice
    of the original bump (taken along the line through (e,f) and (c,d))
    around the vertical axis passing through (e, f).

    Args:
        x, y: Coordinates (scalar or numpy arrays) where to evaluate.
        a, b: Original bump support center.
        c, d: Original bump peak location.
        R: Original bump support radius.
        k: Original bump exponent.
        e, f: The center of revolution.

    Returns:
        float or np.ndarray: The z-value(s) of the revolved function.
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate direction vector from (e, f) to (c, d)
    dx_ref = c - e
    dy_ref = d - f
    dist_peak_ref = np.sqrt(dx_ref**2 + dy_ref**2)

    if dist_peak_ref < 1e-15:
        # Revolution center coincides with the original peak.
        # The reference slice direction is undefined.
        # In this specific case, the 'revolved' function might be interpreted
        # as a radially symmetric function around (c,d) based on the bump's
        # value along *some* radial line from (c,d). However, the original bump
        # is NOT radially symmetric around (c,d) due to the boundary (a,b,R).
        # A strict interpretation of "revolve the slice" fails here.
        # Let's return 0 or raise error, as the concept is ill-defined.
        print(f"Warning/Error: Revolution center ({e},{f}) coincides with peak ({c},{d}). "
              "The revolved function definition is ambiguous. Returning 0.")
        return np.zeros_like(x, dtype=float).item() if x.ndim == 0 else np.zeros_like(x, dtype=float)
        # Alternatively: raise ValueError("Revolution center cannot coincide with the peak for this definition.")

    # Calculate the unit direction vector of the reference slice line
    ux_ref = dx_ref / dist_peak_ref
    uy_ref = dy_ref / dist_peak_ref

    # Calculate the radial distance of the input point(s) (x, y) from (e, f)
    # This 'r' is the distance used for the slice.
    r = np.sqrt((x - e)**2 + (y - f)**2)

    # Calculate the corresponding point(s) (x_prime, y_prime) on the
    # reference slice line (e,f) -> (c,d) at distance r from (e,f).
    x_prime = e + r * ux_ref
    y_prime = f + r * uy_ref

    # Evaluate the *original* off-center bump function at these mapped points
    z_revolved = off_center_bump(x_prime, y_prime, a, b, c, d, R, k)

    # Return scalar if input was scalar
    # Note: off_center_bump already handles this, so z_revolved will be correct type
    return z_revolved


# --- Example Usage ---
def main():
    # Original Bump Parameters
    a, b = 0.0, 0.0
    R = 4.0
    c, d = 1.5, 1.0
    k = 2

    # Point for slice/revolution center
    e, f = 10, 10

    # --- 1. Test the Slice Function ---
    # Extend visualization range to see full bump and some padding
    r_max_vis = np.sqrt((c-e)**2 + (d-f)**2) + 2*R  # Double the padding beyond support
    r_vals = np.linspace(-2*R, r_max_vis, 400)  # Increased points and range
    z_slice = get_slice_along_line(r_vals, a, b, c, d, R, k, e, f)

    # Plot the slice
    plt.figure(figsize=(12, 7))  # Larger figure
    plt.plot(r_vals, z_slice, label=f'Slice along line ({e},{f}) to ({c},{d})')
    plt.xlabel(f'Distance r from ({e},{f})')
    plt.ylabel('Bump Value (z)')
    plt.title('1D Slice of the Off-Center Bump')
    plt.grid(True)
    plt.axvline(0, color='gray', linestyle='--', label=f'Point (e,f)=({e},{f})')
    dist_peak_on_line = np.sqrt((c-e)**2 + (d-f)**2) # distance from e,f to c,d
    plt.axvline(dist_peak_on_line, color='red', linestyle='--', label=f'Original Peak (c,d)=({c},{d}) projected')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()


    # --- 2. Test the Revolved Function ---

    # Create Grid for Plotting
    grid_density = 100
    vis_extent = R + max(abs(a), abs(b), abs(e), abs(f)) + 1.0 # Ensure visualization covers area
    x_range = np.linspace(e - vis_extent, e + vis_extent, grid_density)
    y_range = np.linspace(f - vis_extent, f + vis_extent, grid_density)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate Z values for the revolved function
    Z_revolved = revolved_bump(X, Y, a, b, c, d, R, k, e, f)

    # Create the 3D Plot for the revolved function
    fig_rev = plt.figure(figsize=(15, 12))  # Larger figure
    ax_rev = fig_rev.add_subplot(111, projection='3d')
    surf_rev = ax_rev.plot_surface(X, Y, Z_revolved, cmap='viridis', edgecolor='none', alpha=0.9)

    # Add vertical line at the center of revolution
    ax_rev.plot([e, e], [f, f], [0, revolved_bump(e, f, a, b, c, d, R, k, e, f)],
                color='red', linestyle='--', linewidth=2, label=f'Revolution Center ({e},{f})')
    ax_rev.scatter(e, f, 0, color='red', s=40)

    ax_rev.set_xlabel('X axis')
    ax_rev.set_ylabel('Y axis')
    ax_rev.set_zlabel('z = revolved_bump(x, y)')
    ax_rev.set_title(f'Bump Slice Revolved Around ({e},{f})')
    ax_rev.set_zlim(0, 1.2)
    fig_rev.colorbar(surf_rev, shrink=0.5, aspect=10, label='Function Value')
    ax_rev.legend()
    plt.show()

    # --- Optional: Compare with original bump ---
    Z_original = off_center_bump(X, Y, a, b, c, d, R, k)
    fig_comp = plt.figure(figsize=(15, 12))  # Larger figure
    ax_comp = fig_comp.add_subplot(111, projection='3d')
    surf_comp = ax_comp.plot_surface(X, Y, Z_original, cmap='magma', edgecolor='none', alpha=0.9)
    ax_comp.plot([c, c], [d, d], [0, 1], color='blue', linestyle='--', linewidth=2, label=f'Original Peak ({c},{d})')
    ax_comp.scatter(c, d, 0, color='blue', s=40)
    ax_comp.plot([a, a], [b, b], [0, off_center_bump(a,b, a, b, c, d, R, k)], color='green', linestyle='--', linewidth=2, label=f'Original Support Center ({a},{b})')
    ax_comp.scatter(a, b, 0, color='green', s=40)
    ax_comp.set_xlabel('X axis')
    ax_comp.set_ylabel('Y axis')
    ax_comp.set_zlabel('z = off_center_bump(x, y)')
    ax_comp.set_title('Original Off-Center Bump')
    ax_comp.set_zlim(0, 1.2)
    fig_comp.colorbar(surf_comp, shrink=0.5, aspect=10, label='Function Value')
    ax_comp.legend()
    plt.show()


if __name__ == "__main__":
    main()