import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import time

# --- Define the Bump Function ---
def off_center_bump(x, y, a, b, c, d, R, k):
    """
    Calculates the value of the off-center bump function.

    Args:
        x, y: Coordinates (can be single values or numpy arrays).
        a, b: Center of the circular support.
        c, d: Location of the peak.
        R: Radius of the circular support.
        k: Exponent controlling smoothness at the boundary (k>=1).

    Returns:
        z: Value of the bump function (numpy array or float).
    """
    # Squared distance from the peak
    v = (x - c)**2 + (y - d)**2
    # Squared distance from the support center
    u = (x - a)**2 + (y - b)**2
    # Measure of distance to boundary (squared)
    w = R**2 - u

    # Initialize output with zeros (for points outside the support)
    z = np.zeros_like(x, dtype=float)

    # Define the condition for being strictly inside the support circle
    inside_mask = u < R**2

    # Calculate the denominator (v + w) only for points inside
    # Add a small epsilon to prevent division by zero numerically, although
    # theoretically v+w > 0 strictly inside the circle if (c,d) is strictly inside.
    denominator = v[inside_mask] + w[inside_mask]
    # Ensure denominator is not zero or negative before division
    valid_denom_mask = denominator > 1e-15 # A small tolerance

    # Calculate the function value only where inside and denominator is valid
    # Combine masks
    final_mask = np.zeros_like(x, dtype=bool)
    final_mask[inside_mask] = valid_denom_mask

    # Calculate z = (w / (v + w))^k only for valid points
    z[final_mask] = (w[final_mask] / denominator[valid_denom_mask])**k

    # Ensure the peak value is exactly 1 (handling potential floating point inaccuracies)
    # Find the grid point closest to the peak (c, d)
    # This might not be necessary if the grid resolution is high enough,
    # but it guarantees the visual peak reaches 1 if (c,d) isn't exactly a grid point.
    # However, the analytical function *does* guarantee f(c,d)=1.
    # For plotting, this step isn't strictly needed as the surface will show it.

    # Ensure values are clipped between 0 and 1
    z = np.clip(z, 0, 1)

    return z


def main1():

    # --- Parameters ---
    a, b = 0.0, 0.0   # Center of the circular support
    R = 3.0          # Radius of the circular support
    c, d = 1.0, 0.5   # Location of the peak (must be inside the circle)
    k = 5       # Exponent for the polynomial bump (k>=1, higher k is smoother at boundary)

    # --- Check if peak is inside the support circle ---
    peak_dist_sq = (c - a)**2 + (d - b)**2
    if peak_dist_sq >= R**2:
        raise ValueError(f"Peak (c,d)=({c},{d}) is not strictly inside the support circle with center (a,b)=({a},{b}) and radius R={R}")

    # --- Create Grid for Plotting ---
    grid_density = 150 # Increase for smoother plot
    x_range = np.linspace(a - R * 1.2, a + R * 1.2, grid_density)
    y_range = np.linspace(b - R * 1.2, b + R * 1.2, grid_density)
    X, Y = np.meshgrid(x_range, y_range)

    # --- Calculate Z values ---
    Z = off_center_bump(X, Y, a, b, c, d, R, k)

    # --- Create the 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8, rstride=1, cstride=1) # rstride/cstride for surface detail

    # --- Add Vertical Lines ---

    # Line at the support center (a, b)
    # Calculate the function value at (a, b) - this won't be 1 unless a=c, b=d
    z_center = off_center_bump(np.array([a]), np.array([b]), a, b, c, d, R, k)[0]
    ax.plot([a, a], [b, b], [0, z_center], color='red', linestyle='--', linewidth=2.5, label=f'Support Center ({a},{b})')
    ax.scatter(a, b, 0, color='red', s=50, marker='o') # Mark base of line

    # Line at the peak location (c, d)
    # The peak value is defined to be 1
    ax.plot([c, c], [d, d], [0, 1], color='blue', linestyle='--', linewidth=2.5, label=f'Peak Location ({c},{d})')
    ax.scatter(c, d, 0, color='blue', s=50, marker='o') # Mark base of line

    # --- Customize Plot ---
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Off-Center Bump Function with Circular Support')
    ax.set_zlim(0, 1.2) # Ensure the peak at z=1 is clearly visible
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Function Value')
    ax.legend()

    # Adjust view angle (optional)
    # ax.view_init(elev=30., azim=-60)

    plt.show()







def off_center_bump2(x, y, a, b, c, d, R, k):
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
    # This mask will work for both scalar (0-d array) and higher-dim arrays
    inside_mask = w > 1e-15

    # --- Important: Only proceed with calculations where inside_mask is True ---
    # We need to handle the case where inside_mask might be a single boolean (scalar input)
    # or an array of booleans.

    # Calculate the denominator (v + w)
    # This calculation itself is safe for both scalars and arrays
    denominator = v + w

    # Create a mask for points that are BOTH inside AND have a valid denominator
    # Avoid division by zero or near-zero, esp. if peak is near boundary.
    # The denominator v+w should be > 0 strictly inside if the peak is strictly inside.
    # Use a tolerance for numerical stability.
    valid_calculation_mask = inside_mask & (denominator > 1e-15)

    # --- Perform calculation only where valid ---
    # Check if there are *any* valid points to calculate (works for scalar bool or array)
    if np.any(valid_calculation_mask):
        # If the input was scalar, valid_calculation_mask is a single True.
        # If array input, valid_calculation_mask is a boolean array.

        # Select numerator and denominator ONLY where the calculation is valid.
        # This works correctly whether the mask/arrays are 0-d or higher-d.
        numerator_valid = w[valid_calculation_mask]
        denominator_valid = denominator[valid_calculation_mask]

        # Perform the core calculation on the selected valid values
        result_valid = (numerator_valid / denominator_valid)**k

        # Place the calculated results back into the corresponding locations in z.
        # This assignment also works correctly for scalar or array masks.
        z[valid_calculation_mask] = result_valid

    # Clip final result (redundant if logic above is correct, but safe)
    z = np.clip(z, 0, 1)

    # Return scalar if input was scalar, otherwise return the array
    if x.ndim == 0 and y.ndim == 0:
        # If input was scalar, z is now a 0-d array. Extract the scalar value.
        return z.item()
    else:
        # If input was array, return the result array.
        return z
    
# --- Define the Integrand ---
# The integrand function for dblquad should take y then x
def integrand_min(y, x, p1, p2):
    """
    Calculates min(f1(x, y), f2(x, y)).

    Args:
        y, x: Coordinates passed by dblquad.
        p1: Tuple of parameters for bump 1 (a1, b1, c1, d1, R1, k1)
        p2: Tuple of parameters for bump 2 (a2, b2, c2, d2, R2, k2)
    """
    a1, b1, c1, d1, R1, k1 = p1
    a2, b2, c2, d2, R2, k2 = p2

    f1_val = off_center_bump2(x, y, a1, b1, c1, d1, R1, k1)
    f2_val = off_center_bump2(x, y, a2, b2, c2, d2, R2, k2)

    return np.minimum(f1_val, f2_val)


def main2():
    # --- Parameters for the two bumps ---
    # Bump 1
    a1, b1 = 0.0, 0.0
    R1 = 3.0
    c1, d1 = 1.0, 0.5
    k1 = 2

    # Bump 2
    a2, b2 = 2.0, 1.0
    R2 = 2.5
    c2, d2 = 1.5, 2.0
    k2 = 2

    # --- Check peak locations (optional but good practice) ---
    if (c1 - a1)**2 + (d1 - b1)**2 >= R1**2:
        raise ValueError("Peak 1 is not inside its support.")
    if (c2 - a2)**2 + (d2 - b2)**2 >= R2**2:
        raise ValueError("Peak 2 is not inside its support.")

    # --- Determine Integration Bounds ---
    # Find a bounding box that covers both circular supports
    xmin = min(a1 - R1, a2 - R2)
    xmax = max(a1 + R1, a2 + R2)
    ymin = min(b1 - R1, b2 - R2)
    ymax = max(b1 + R1, b2 + R2)

    print(f"Integration bounds: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")

    # --- Perform Numerical Integration ---
    params1 = (a1, b1, c1, d1, R1, k1)
    params2 = (a2, b2, c2, d2, R2, k2)

    print("Starting numerical integration (this might take a moment)...")
    start_time = time.time()

    # dblquad integrates dy first (inner), then dx (outer)
    # It expects integrand(y, x, ...), xlow, xhigh, ylow(x), yhigh(x)
    # Since our y bounds are constant, we use lambda functions
    integral_value, integral_error = dblquad(
        integrand_min,
        xmin,                 # x lower bound
        xmax,                 # x upper bound
        lambda x: ymin,       # y lower bound (can be function of x)
        lambda x: ymax,       # y upper bound (can be function of x)
        args=(params1, params2), # Pass bump parameters to the integrand
        epsabs=1.49e-8,       # Absolute error tolerance
        epsrel=1.49e-8        # Relative error tolerance
    )

    end_time = time.time()
    print(f"Integration finished in {end_time - start_time:.2f} seconds.")

    # --- Output the Result ---
    print(f"\nParameters for Bump 1:")
    print(f"  Support Center (a1, b1): ({a1}, {b1}), Radius R1: {R1}")
    print(f"  Peak Location (c1, d1): ({c1}, {d1}), Exponent k1: {k1}")
    print(f"\nParameters for Bump 2:")
    print(f"  Support Center (a2, b2): ({a2}, {b2}), Radius R2: {R2}")
    print(f"  Peak Location (c2, d2): ({c2}, {d2}), Exponent k2: {k2}")

    print(f"\nNumerical integral of min(f1(x, y), f2(x, y)):")
    print(f"  Value: {integral_value:.6f}")
    print(f"  Estimated Error: {integral_error:.2e}")


    # --- Optional: Visualization (to understand the intersection) ---
    grid_density = 100
    vis_x_range = np.linspace(xmin, xmax, grid_density)
    vis_y_range = np.linspace(ymin, ymax, grid_density)
    X, Y = np.meshgrid(vis_x_range, vis_y_range)

    Z1 = off_center_bump2(X, Y, *params1)
    Z2 = off_center_bump2(X, Y, *params2)
    Z_min = np.minimum(Z1, Z2)

    fig = plt.figure(figsize=(12, 6))

    # Plot the minimum surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z_min, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title('Intersection Surface: min(f1, f2)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_zlim(0, 1.1)

    # Plot contour of the minimum
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z_min, cmap='viridis', levels=20)
    # Overlay circles for reference
    circle1 = plt.Circle((a1, b1), R1, color='red', fill=False, linestyle='--', label='Support 1')
    circle2 = plt.Circle((a2, b2), R2, color='blue', fill=False, linestyle='--', label='Support 2')
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    ax2.plot(c1, d1, 'rX', markersize=8, label='Peak 1')
    ax2.plot(c2, d2, 'bX', markersize=8, label='Peak 2')
    ax2.set_title('Intersection Contour: min(f1, f2)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    fig.colorbar(contour, ax=ax2, label='min(f1, f2)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main2()