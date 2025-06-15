import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Replicate the "Expert" Algorithm from your script ---
# We need to reproduce the core logic to generate training data and for verification.

# Constants from quad_autocorrect.py
CORRECTION_GAIN_KP = 1.5
NEUTRAL_LIFT_ANGLE = 30.0
MIN_LIFT_ANGLE = 20.0
MAX_LIFT_ANGLE = 45.0

def clamp(value, min_val, max_val):
    """Clamps a value within a specified range."""
    return max(min_val, min(value, max_val))

def get_expert_action(yaw_error: float) -> tuple[float, float]:
    """
    This is the original algorithm we want the neural network to learn.
    It takes a yaw error and returns the calculated (N, Y) lift angles.
    """
    correction = CORRECTION_GAIN_KP * yaw_error
    
    # These signs assume a positive correction (from a right turn) should steer LEFT.
    # To steer LEFT: increase lift/push of right-side legs (Y), decrease left-side (N).
    N = clamp(NEUTRAL_LIFT_ANGLE - correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    Y = clamp(NEUTRAL_LIFT_ANGLE + correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    
    return N, Y

# --- 2. Define the Neural Network (The "Policy") ---
# A very simple network is enough to learn this function.
# Input: 1 value (yaw_error)
# Output: 2 values (N, Y)

class WalkPolicy(nn.Module):
    def __init__(self):
        super(WalkPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),      # Input layer (1 neuron) -> Hidden layer (64 neurons)
            nn.ReLU(),             # Activation function
            nn.Linear(64, 64),     # Hidden layer -> Hidden layer
            nn.ReLU(),             # Activation function
            nn.Linear(64, 2)       # Hidden layer -> Output layer (2 neurons for N and Y)
        )

    def forward(self, x):
        return self.network(x)

# --- 3. Training Setup ---

def generate_training_data(num_samples=5000):
    """
    Generates training data by sampling the expert.
    We create random yaw errors and ask the expert for the correct action.
    """
    # Generate random yaw errors over a plausible range (e.g., -45 to +45 degrees)
    yaw_errors = np.random.uniform(-45.0, 45.0, num_samples)
    
    # Get the "correct" N, Y actions from our expert algorithm
    expert_actions = np.array([get_expert_action(err) for err in yaw_errors])
    
    # Convert to PyTorch tensors
    # The state (input) needs an extra dimension for the network
    states_tensor = torch.FloatTensor(yaw_errors).view(-1, 1)
    actions_tensor = torch.FloatTensor(expert_actions)
    
    return states_tensor, actions_tensor

# --- 4. The Main Training and Verification Script ---

if __name__ == "__main__":
    # --- Training ---
    print("--- Starting Policy Training (Behavioral Cloning) ---")
    
    # Instantiate the policy, loss function, and optimizer
    policy = WalkPolicy()
    criterion = nn.MSELoss()  # Mean Squared Error is perfect for regression
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Generate our training dataset
    states, expert_actions = generate_training_data()
    print(f"Generated {len(states)} training samples.")
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        # 1. Forward pass: compute predicted actions by passing states to the policy
        predicted_actions = policy(states)
        
        # 2. Compute loss
        loss = criterion(predicted_actions, expert_actions)
        
        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
    print("--- Training Finished ---")

    # Save the trained model (optional)
    torch.save(policy.state_dict(), 'walk_policy.pth')
    print("Policy saved to walk_policy.pth\n")


    # --- Verification ---
    print("--- Verifying Trained Policy vs. Expert Algorithm ---")
    policy.eval() # Set the policy to evaluation mode

    # Test with some specific yaw errors
    test_errors = [-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0]

    with torch.no_grad(): # We don't need to calculate gradients for verification
        for error in test_errors:
            print(f"\nTesting with Yaw Error = {error:.1f} degrees")
            
            # Get the ground truth from the original algorithm
            expert_N, expert_Y = get_expert_action(error)
            print(f"  Expert Algorithm -> N={expert_N:6.2f}, Y={expert_Y:6.2f}")
            
            # Get the action from our trained neural network
            error_tensor = torch.FloatTensor([[error]]) # Must be a 2D tensor [[val]]
            predicted_N, predicted_Y = policy(error_tensor).squeeze().tolist()
            print(f"  Trained Policy   -> N={predicted_N:6.2f}, Y={predicted_Y:6.2f}")