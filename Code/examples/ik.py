import numpy as np
import matplotlib.pyplot as plt

# --- Helper function for angle wrapping ---
# It's useful to wrap angles for clarity, e.g., to [-180, 180]
def wrap_to_180(angle_deg):
    """Wraps an angle in degrees to the range [-180, 180]."""
    # Normalizes angle to [0, 360) then shifts to [-180, 180)
    return (angle_deg + 180) % 360 - 180

# --- Robot Parameters ---
L1 = 50.0  # Length of the hip link (blue), converted from 5 cm to mm
L2 = 40.97 # Length of the knee link (red) in mm

# --- Local Frame Definitions (Degrees) ---
# These define the zero-angle direction for your local joint sensors/encoders
HIP_LOCAL_FRAME_OFFSET_DEG = 225.0  # Zero Hip Local angle corresponds to 225 deg CCW from global +X
KNEE_LOCAL_FRAME_OFFSET_DEG = 315.0 # Zero Knee Local angle corresponds to 315 deg CCW from global +X

# --- Inverse Kinematics Function (Calculates standard angles) ---
# theta1_global_rad: angle of hip link from global horizontal +X (CCW positive)
# theta2_relative_rad: angle of knee link from hip link (CCW positive, negative for inward bend)
def inverse_kinematics(x, y, L1, L2):
    """
    Calculate standard joint angles (theta1_global_rad, theta2_relative_rad)
    for a given end-effector position (x, y) in the global frame.

    Args:
        x (float): Desired x-coordinate of the foot (mm) in global frame.
        y (float): Desired y-coordinate of the foot (mm) in global frame.
        L1 (float): Length of the first link (mm).
        L2 (float): Length of the second link (mm).

    Returns:
        tuple or None: (theta1_global_rad, theta2_relative_rad) in radians,
                       or None if unreachable.
                       theta1_global_rad: hip angle (rad) relative to global horizontal
                       theta2_relative_rad: knee angle (rad) relative to hip link (negative for inward bend)
    """
    D_squared = x**2 + y**2
    D = np.sqrt(D_squared)

    # Check reachability with small tolerance
    if D > L1 + L2 + 1e-6 or D < abs(L1 - L2) - 1e-6:
        # print(f"Warning: Target point ({x:.2f}, {y:.2f}) is unreachable. D={D:.2f}")
        return None

    # Law of Cosines for internal angles alpha (at hip), beta (at knee)
    # Handle case where D is zero to avoid division by zero
    cos_alpha = (L1**2 + D_squared - L2**2) / (2 * L1 * D) if D > 1e-9 else 1.0
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    cos_beta = (L1**2 + L2**2 - D_squared) / (2 * L1 * L2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # Angle of hip-foot vector relative to global horizontal +X
    gamma = np.arctan2(y, x)

    # Calculate theta1 (hip angle relative to global horizontal) - using 'elbow up' configuration
    theta1_global_rad = gamma + alpha

    # Calculate theta2 (knee angle relative to hip link) - negative for inward bend configuration
    theta2_relative_rad = -(np.pi - beta)

    return theta1_global_rad, theta2_relative_rad

# --- Trajectory Definition (Triangle) ---
# P1: Ground, Back
# P2: Lifted, Peak
# P3: Ground, Front

# Define trajectory parameters (Adjusted for reachability with L1=50, L2=40.97)
x_center = 50.0      # Approximate horizontal center of the base
stroke_length = 40.0 # Horizontal distance of the base
y_ground = -40.0     # Vertical position of the ground level
lift_height = 30.0   # Height the foot is lifted

# Define the 3 vertices of the triangle in the global (x,y) frame
P1 = (x_center - stroke_length / 2, y_ground)           # Back-Ground (30, -40)
P2 = (x_center, y_ground + lift_height)                 # Center-Lifted (50, -10)
P3 = (x_center + stroke_length / 2, y_ground)           # Front-Ground (70, -40)

# Number of intermediate steps for each segment
num_steps_segment = 40
total_steps = num_steps_segment * 3

# Generate sequence of (x, y) points along the trajectory segments (P1->P2->P3->P1)
x_traj = []
y_traj = []

# Segment 1: P1 to P2 (Diagonal Lift/Forward)
for i in range(num_steps_segment):
    t = i / (num_steps_segment - 1)
    x = P1[0] + t * (P2[0] - P1[0])
    y = P1[1] + t * (P2[1] - P1[1])
    x_traj.append(x)
    y_traj.append(y)

# Segment 2: P2 to P3 (Diagonal Down/Forward)
for i in range(num_steps_segment):
    t = i / (num_steps_segment - 1)
    x = P2[0] + t * (P3[0] - P2[0])
    y = P2[1] + t * (P3[1] - P2[1])
    x_traj.append(x)
    y_traj.append(y)

# Segment 3: P3 to P1 (Horizontal Backward - Stance)
for i in range(num_steps_segment):
    t = i / (num_steps_segment - 1)
    x = P3[0] + t * (P1[0] - P3[0])
    y = P1[1] # Stay at P1's y-coordinate (ground level)
    x_traj.append(x)
    y_traj.append(y)


# --- Calculate Joint Angles for Each Trajectory Point ---
theta1_global_traj_rad = []     # Standard hip angle (global horizontal ref)
theta2_relative_traj_rad = []   # Standard knee angle (relative to hip link)

# Lists to store angles in LOCAL frames (in degrees)
theta1_local_traj_deg = []
theta2_local_traj_deg = []

reachable_x_actual = [] # Store the x,y points that were actually reachable
reachable_y_actual = []

for i in range(total_steps):
    x = x_traj[i]
    y = y_traj[i]

    standard_angles = inverse_kinematics(x, y, L1, L2)

    if standard_angles is not None:
        theta1_global_rad, theta2_relative_rad = standard_angles

        # Store standard angles (useful for debugging or plotting global)
        theta1_global_traj_rad.append(theta1_global_rad)
        theta2_relative_traj_rad.append(theta2_relative_rad)

        # Convert standard angles to degrees
        theta1_global_deg = np.rad2deg(theta1_global_rad)
        theta2_relative_deg = np.rad2deg(theta2_relative_rad)

        # --- Convert to Local Frame Angles ---
        # Hip Local Angle = Standard Hip Angle (Global Ref) - Hip Local Frame Offset
        theta1_local_deg = theta1_global_deg - HIP_LOCAL_FRAME_OFFSET_DEG
        theta1_local_traj_deg.append(wrap_to_180(theta1_local_deg)) # Store wrapped angle

        # Knee Link Global Angle = Standard Hip Angle (Global Ref) + Standard Knee Angle (Relative to Hip Link)
        knee_link_global_deg = theta1_global_deg + theta2_relative_deg
        # Knee Local Angle = Knee Link Global Angle - Knee Local Frame Offset
        theta2_local_deg = knee_link_global_deg - KNEE_LOCAL_FRAME_OFFSET_DEG
        theta2_local_traj_deg.append(wrap_to_180(theta2_local_deg)) # Store wrapped angle

        reachable_x_actual.append(x)
        reachable_y_actual.append(y)
    else:
        # Handle unreachable points - append NaN to all angle lists
        print(f"Step {i}: Point ({x:.2f}, {y:.2f}) is unreachable. Appending NaN.")
        theta1_global_traj_rad.append(np.nan)
        theta2_relative_traj_rad.append(np.nan)
        theta1_local_traj_deg.append(np.nan) # Append NaN to local angle lists too
        theta2_local_traj_deg.append(np.nan)
        reachable_x_actual.append(np.nan) # Keep lists same length
        reachable_y_actual.append(np.nan)

# --- Plotting ---

# Plot the desired trajectory and the reachable points calculated (Global Frame)
plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, 'b--', label='Desired Trajectory')
plt.plot(reachable_x_actual, reachable_y_actual, 'g-', label='Calculated Trajectory (Reachable)')
plt.plot(P1[0], P1[1], 'ro', markersize=5, label='P1 (Ground Back)')
plt.plot(P2[0], P2[1], 'ro', markersize=5, label='P2 (Lifted Peak)')
plt.plot(P3[0], P3[1], 'ro', markersize=5, label='P3 (Ground Front)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Robot Foot Trajectory (Global Frame)')
plt.axis('equal') # Keep the aspect ratio equal for correct representation
plt.grid(True)
plt.legend()
plt.show()

# Plot the LOCAL joint angles over the trajectory steps
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
plt.plot(theta1_local_traj_deg)
plt.ylabel(f'Hip Local Angle (degrees)\n(Offset by {HIP_LOCAL_FRAME_OFFSET_DEG} deg)')
plt.title('Hip and Knee Angles Over Trajectory Steps (Local Frames)')
plt.grid(True)
# Mark segment transitions (indices are 0, num_steps_segment-1, num_steps_segment*2-1, total_steps-1)
plt.axvline(x=num_steps_segment -1, color='r', linestyle='--', lw=1, label='End P1->P2')
plt.axvline(x=num_steps_segment*2 -1, color='m', linestyle='--', lw=1, label='End P2->P3')
plt.legend()


plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
plt.plot(theta2_local_traj_deg)
plt.ylabel(f'Knee Local Angle (degrees)\n(Offset by {KNEE_LOCAL_FRAME_OFFSET_DEG} deg)')
plt.xlabel('Trajectory Step Number')
plt.grid(True)
plt.axvline(x=num_steps_segment -1, color='r', linestyle='--', lw=1, label='End P1->P2')
plt.axvline(x=num_steps_segment*2 -1, color='m', linestyle='--', lw=1, label='End P2->P3')
plt.legend()


plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()

# --- Simulate and plot leg poses at key points (Global Frame) ---
# We use standard angles and plot in the global frame to visualize the physical leg
vis_steps = np.linspace(0, total_steps - 1, 9, dtype=int) # Get 9 evenly spaced steps

plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, 'b--', alpha=0.5, label='Desired Trajectory (Global)')
plt.plot(reachable_x_actual, reachable_y_actual, 'g-', alpha=0.7, label='Calculated Trajectory (Global)')
plt.plot(0, 0, 'ko', markersize=8, label='Hip Origin (Global Frame)')

for i in vis_steps:
    # Need standard angles for plotting poses in global frame
    if i < len(theta1_global_traj_rad) and not np.isnan(theta1_global_traj_rad[i]):
        t1_global_rad = theta1_global_traj_rad[i]
        t2_relative_rad = theta2_relative_traj_rad[i]
        x_f, y_f = x_traj[i], y_traj[i] # Use desired point for plotting leg pose

        # Calculate knee position using forward kinematics based on standard angles
        x_k = L1 * np.cos(t1_global_rad)
        y_k = L1 * np.sin(t1_global_rad)

        # Plot links in the global frame
        plt.plot([0, x_k], [0, y_k], 'k-', lw=2) # Hip link (L1)
        plt.plot([x_k, x_f], [y_k, y_f], 'k-', lw=2) # Knee link (L2)

        # Plot joints/foot
        plt.plot(x_k, y_k, 'ko', markersize=5) # Knee joint
        plt.plot(x_f, y_f, 'yo', markersize=7) # Foot position (yellow dot)

plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Robot Leg Poses Along Trajectory (Global Frame)')
plt.axis('equal')
plt.grid(True)
plt.legend()
# Set axis limits
plt.xlim(min(x_traj) - 20, max(x_traj) + 20)
plt.ylim(min(y_traj) - 20, max(L1, L2) + 20) # Ensure enough space above ground
plt.show()


# --- Display angles for the 3 key points (P1, P2, P3) in LOCAL FRAMES ---
# Find the indices corresponding to the vertices P1, P2, P3
# P1 (start) is index 0
# P2 is the end of the first segment (index num_steps_segment - 1)
# P3 is the end of the second segment (index num_steps_segment * 2 - 1)
# P1 (end of cycle) is the end of the third segment (index total_steps - 1)

idx_p1_start = 0
idx_p2 = num_steps_segment - 1
idx_p3 = num_steps_segment * 2 - 1
idx_p1_end_cycle = total_steps - 1


print("\n--- Calculated Joint Angles (degrees) for the 3 Vertices of the Triangle Trajectory ---")
print("Angles are relative to the specified local frames:")
print(f"  - Hip Local Frame Zero: {HIP_LOCAL_FRAME_OFFSET_DEG} deg CCW from global +X.")
print(f"  - Knee Local Frame Zero: {KNEE_LOCAL_FRAME_OFFSET_DEG} deg CCW from global +X.")
print("Angles are wrapped to [-180, 180] for readability.")


points_of_interest = {
    "P1 (Ground Back) - Start of Cycle": (idx_p1_start, P1),
    "P2 (Lifted Peak)": (idx_p2, P2),
    "P3 (Ground Front)": (idx_p3, P3),
    "P1 (Ground Back) - End of Cycle": (idx_p1_end_cycle, P1) # Note: Same physical point as P1 Start
}

for name, (idx, global_pos) in points_of_interest.items():
    # Ensure the index is valid and the point was reachable (angle is not NaN)
    if idx < len(theta1_local_traj_deg) and not np.isnan(theta1_local_traj_deg[idx]):
         hip_local_angle = theta1_local_traj_deg[idx]
         knee_local_angle = theta2_local_traj_deg[idx]
         print(f"{name} (Global Pos: ({global_pos[0]:.2f}, {global_pos[1]:.2f})): Hip Local Angle = {hip_local_angle:.2f} deg, Knee Local Angle = {knee_local_angle:.2f} deg")
    else:
         # If the vertex point itself was unreachable (shouldn't happen with adjusted trajectory)
         print(f"{name} (Global Pos: ({global_pos[0]:.2f}, {global_pos[1]:.2f})): Point was unreachable, cannot provide angles.")


# --- Optional: Verify angles for the diagram's reference pose in LOCAL FRAMES ---
print("\n--- Angles for the Diagram's Reference Pose (in Local Frames) ---")
# Assuming diagram pose is: Hip Global 45 deg, Knee Relative to Hip -45 deg
diagram_theta1_global_deg = 45.0
diagram_theta2_relative_to_L1_deg = -45.0 # Angle between blue (45) and red (0 global) is -45

# Convert these standard angles to your specified local frames
diagram_theta1_local_deg = diagram_theta1_global_deg - HIP_LOCAL_FRAME_OFFSET_DEG
diagram_theta1_local_deg_wrapped = wrap_to_180(diagram_theta1_local_deg)

diagram_knee_link_global_deg = diagram_theta1_global_deg + diagram_theta2_relative_to_L1_deg # 45 + (-45) = 0 deg Global
diagram_theta2_local_deg = diagram_knee_link_global_deg - KNEE_LOCAL_FRAME_OFFSET_DEG # 0 - 315 = -315 deg
diagram_theta2_local_deg_wrapped = wrap_to_180(diagram_theta2_local_deg) # Wrapped: 45 deg

# Calculate the foot position for this diagram pose using FK
diagram_theta1_global_rad = np.deg2rad(diagram_theta1_global_deg)
diagram_theta2_relative_to_L1_rad = np.deg2rad(diagram_theta2_relative_to_L1_deg)
diagram_x_fk = L1 * np.cos(diagram_theta1_global_rad) + L2 * np.cos(diagram_theta1_global_rad + diagram_theta2_relative_to_L1_rad)
diagram_y_fk = L1 * np.sin(diagram_theta1_global_rad) + L2 * np.sin(diagram_theta1_global_rad + diagram_theta2_relative_to_L1_rad)


print(f"Diagram Pose (Assumed Standard Angles: Hip Global {diagram_theta1_global_deg} deg, Knee Relative {diagram_theta2_relative_to_L1_deg} deg; Foot at ({diagram_x_fk:.2f}, {diagram_y_fk:.2f})):")
print(f"  Hip Local Angle = {diagram_theta1_local_deg_wrapped:.2f} deg")
print(f"  Knee Local Angle = {diagram_theta2_local_deg_wrapped:.2f} deg")
print("Note: In this specific diagram pose, Hip Local Angle is -180 deg, and Knee Local Angle is 45 deg.")