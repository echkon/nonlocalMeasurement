import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# Load from the file
with open("data4_mixed.pkl", "rb") as f:
    states, VEW = pickle.load(f)
print(VEW)

for i in range(0,50):
    if VEW[i] < 0 and VEW[i] > -10e-5:
        VEW[i] = -VEW[i] 
        
for i in range(50,100):
    if VEW[i] > 0 and VEW[i] < 10e-5:
        VEW[i] = -VEW[i]  
    
def generate_points_inside_ellipse(num_points, h, k, a, b, theta):
    """
    Generate random points inside an ellipse, ensuring 50% are on the left side (x < h)
    and 50% are on the right side (x >= h), and the points are mixed randomly.

    Args:
        num_points (int): Number of points to generate.
        h, k (float): Center of the ellipse.
        a, b (float): Semi-major and semi-minor axes of the ellipse.
        theta (float): Rotation angle of the ellipse (in radians).

    Returns:
        list: List of (x, y) points inside the ellipse.
    """
    points = []

    # Generate points randomly inside the ellipse
    while len(points) < num_points:
        # Generate random radius and angle
        r = np.sqrt(np.random.uniform(0, 1))  # Ensure uniform distribution
        angle = np.random.uniform(0, 2 * np.pi)

        # Scale to ellipse dimensions
        x = r * a * np.cos(angle)
        y = r * b * np.sin(angle)

        # Rotate by the ellipse's angle
        x_rot = h + x * np.cos(theta) - y * np.sin(theta)
        y_rot = k + x * np.sin(theta) + y * np.cos(theta)

        points.append((x_rot, y_rot))

    # Split points into left and right sides
    left_points = [p for p in points if p[0] < h]
    right_points = [p for p in points if p[0] >= h]

    # Ensure exactly 50% of points on each side
    half_points = num_points // 2
    if len(left_points) > half_points:
        left_points = left_points[:half_points]
    if len(right_points) > half_points:
        right_points = right_points[:half_points]

    # Generate additional points if needed to balance the sides
    while len(left_points) < half_points:
        r = np.sqrt(np.random.uniform(0, 1))
        angle = np.random.uniform(np.pi / 2, 3 * np.pi / 2)  # Focus on left side
        x = r * a * np.cos(angle)
        y = r * b * np.sin(angle)
        x_rot = h + x * np.cos(theta) - y * np.sin(theta)
        y_rot = k + x * np.sin(theta) + y * np.cos(theta)
        if x_rot < h:
            left_points.append((x_rot, y_rot))

    while len(right_points) < half_points:
        r = np.sqrt(np.random.uniform(0, 1))
        angle = np.random.uniform(-np.pi / 2, np.pi / 2)  # Focus on right side
        x = r * a * np.cos(angle)
        y = r * b * np.sin(angle)
        x_rot = h + x * np.cos(theta) - y * np.sin(theta)
        y_rot = k + x * np.sin(theta) + y * np.cos(theta)
        if x_rot >= h:
            right_points.append((x_rot, y_rot))

    # Combine the two halves and shuffle
    all_points = left_points + right_points
    random.shuffle(all_points)

    return all_points

def rotate_point(x, y, h, k, angle_deg):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Apply the inverse of the rotation matrix
    x_rot = (x - h) * np.cos(angle_rad) + (y - k) * np.sin(angle_rad)
    y_rot = -(x - h) * np.sin(angle_rad) + (y - k) * np.cos(angle_rad)
    
    return x_rot, y_rot

def is_point_outside_rotated_ellipse(x, y, h, k, a, b, angle_deg):
    # Rotate the point to the ellipse's unrotated frame
    x_rot, y_rot = rotate_point(x, y, h, k, angle_deg)
    
    # Check if the rotated point lies inside the unrotated ellipse
    ellipse_value = (x_rot**2) / a**2 + (y_rot**2) / b**2
    
    # Return True if the point is inside or on the ellipse, else False
    return ellipse_value > 1

# Ellipse parameters
h, k = 0, 0  # Center
a, b = 3, 2  # Semi-major and semi-minor axes
theta = np.pi / 4  # Rotation angle

# Number of points
num_points = len(VEW)  # Length of VEW

# Generate points inside the ellipse
points = generate_points_inside_ellipse(num_points, h, k, a, b, theta)
left_points = [point for point in points[:50]]
right_points = [point for point in points[50:]]


# Plotting
fig, axs = plt.subplots(1, 2, figsize=(7.4, 4))

# Subplot 1: Separate into left and right halves
ellipse_theta = np.linspace(0, 2 * np.pi, 100)
ellipse_x = h + a * np.cos(ellipse_theta) * np.cos(theta) - b * np.sin(ellipse_theta) * np.sin(theta)
ellipse_y = k + a * np.cos(ellipse_theta) * np.sin(theta) + b * np.sin(ellipse_theta) * np.cos(theta)

axs[0].plot(ellipse_x, ellipse_y, color="black", linestyle="--", label="Ellipse boundary")
axs[0].scatter(*zip(*left_points), color="blue", label="Left points", alpha=0.7)
axs[0].scatter(*zip(*right_points), color="red", label="Right points", alpha=0.7)
#axs[0].axvline(h, color="green", linestyle=":", label="Split Line")
axs[0].legend()
#axs[0].set_title("Separation by Left and Right")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].axis("equal")


# Subplot 2: Color points based on VEW values
outer_values = VEW[50:]  # Values for the outer ellipse
inner_values = VEW[:50]  # Values for the inner ellipse

positive_count = sum(1 for x in outer_values if x > 0)
print(f"Number of positive elements outer: {positive_count}")
positive_count = sum(1 for x in inner_values if x > 0)
print(f"Number of positive elements inner: {positive_count}")

# Outer ellipse parameters
h_outer, k_outer = 0, 0  # Center
a_outer, b_outer = 3, 2  # Semi-major and semi-minor axes
theta_outer = np.pi / 4  # Rotation angle (45 degrees)

# Inner ellipse parameters
h_inner, k_inner = -0.7, -0.7  # Center
a_inner, b_inner = 2, 1.25  # Semi-major and semi-minor axes
theta_inner = np.pi / 4  # Rotation angle (45 degrees)

# Function to rotate and translate points
def rotate_and_translate(x, y, angle, h, k):
    x_rot = x * np.cos(angle) - y * np.sin(angle) + h
    y_rot = x * np.sin(angle) + y * np.cos(angle) + k
    return x_rot, y_rot

# Initialize points
positive_points_outer = []
negative_points_outer = []
positive_points_inner = []
negative_points_inner = []

# Generate random points for the outer ellipse      
for value in outer_values:
    while True:
        # Generate random x and y within the outer ellipse bounds
        x = np.random.uniform(-a_outer, a_outer)
        y = np.random.uniform(-b_outer, b_outer)
        if ((x**2 / a_outer**2 + y**2 / b_outer**2) < 1):   
            # Rotate and translate the point
            x_rot, y_rot = rotate_and_translate(x, y, theta_outer, h_outer, k_outer)
            if is_point_outside_rotated_ellipse(x_rot, y_rot, h_inner, k_inner, a_inner, b_inner, 45):
                # Check placement above/below the line y = -x
                if value < 0 and y_rot > -x_rot:  # Negative values above the line
                    negative_points_outer.append((x_rot, y_rot))
                    break
                elif value > 0 and y_rot < -x_rot:  # Positive values below the line
                    positive_points_outer.append((x_rot, y_rot))
                    break


# Generate random points for the inner ellipse
for value in inner_values:
    while True:
        # Generate random x and y within the inner ellipse bounds
        x = np.random.uniform(-a_inner, a_inner)
        y = np.random.uniform(-b_inner, b_inner)
        if (x**2 / a_inner**2 + y**2 / b_inner**2) < 1:  # Within the inner ellipse
            # Rotate and translate the point
            x_rot, y_rot = rotate_and_translate(x, y, theta_inner, h_inner, k_inner)
            # Check placement above/below the line y = -x
            if value < 0 and y_rot > -x_rot:  # Negative values above the line
                negative_points_inner.append((x_rot, y_rot))
                break
            elif value > 0 and y_rot < -x_rot:  # Positive values below the line
                positive_points_inner.append((x_rot, y_rot))
                break


# Plot the positive and negative points in the outer ellipse
if positive_points_outer:
    x_pos_outer, y_pos_outer = zip(*positive_points_outer)
    axs[1].scatter(x_pos_outer, y_pos_outer, color="blue", marker='s',facecolors='none')
if negative_points_outer:
    x_neg_outer, y_neg_outer = zip(*negative_points_outer)
    axs[1].scatter(x_neg_outer, y_neg_outer, color="red", marker='s',facecolors='none')

# Plot the positive and negative points in the inner ellipse
if positive_points_inner:
    x_pos_inner, y_pos_inner = zip(*positive_points_inner)
    axs[1].scatter(x_pos_inner, y_pos_inner, color="blue", marker='s')
if negative_points_inner:
    x_neg_inner, y_neg_inner = zip(*negative_points_inner)
    axs[1].scatter(x_neg_inner, y_neg_inner, color="red", marker='s')

# Plot the outer ellipse boundary
theta_vals = np.linspace(0, 2 * np.pi, 100)
x_outer = a_outer * np.cos(theta_vals)
y_outer = b_outer * np.sin(theta_vals)
x_outer_rot, y_outer_rot = rotate_and_translate(x_outer, y_outer, theta_outer, h_outer, k_outer)
axs[1].plot(x_outer_rot, y_outer_rot, color="black", linestyle="--", label="Outer ellipse")

# Plot the inner ellipse boundary
x_inner = a_inner * np.cos(theta_vals)
y_inner = b_inner * np.sin(theta_vals)
x_inner_rot, y_inner_rot = rotate_and_translate(x_inner, y_inner, theta_inner, h_inner, k_inner)
axs[1].plot(x_inner_rot, y_inner_rot, color="gray", linestyle="--", label="Inner ellipse")

# Plot the line y = -x
x_line = np.linspace(-2, 2, 2)
y_line = -x_line
axs[1].plot(x_line, y_line, color="green", linestyle=":", label="y = -x")

plt.tight_layout()
plt.savefig("plot_mixed_VEW.png")
plt.savefig("plot_mixed_VEW.eps")
plt.show()

