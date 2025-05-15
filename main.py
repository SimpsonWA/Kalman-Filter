import numpy as np
import matplotlib.pyplot as plt

def circle(samples = 50, origin = np.array([0,0]), radius = 1 ):
    
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    x = origin[0] + radius * np.cos(angles)
    y = origin[1] + radius * np.sin(angles)

    return np.vstack((x, y)).T

timeStep = 1.0
# Intal state and covaraince
mu = np.array([[0.0],
              [0.0]])
sigma = np.array([[1.0 , 0],
                 [0, 1.0]])

# System matrices 
A = np.array([[1,0],
             [0,1]])
B = np.array([[timeStep, 0],
             [0, timeStep]])
C = np.array([[1,0],
             [0,1]])

# Noise Covariances
R = np.array([[0.01, 0],
              [0, 0.01]])  # Process noise
Q = np.array([[0.1, 0],
              [0, 0.1]])   # Measurement noise

pos_real = []
measurements = []
estimates = []

np.random.seed(42)
inital_pos = mu.copy()

# Circle Control input
u_t = circle()


for i in range(50):
    
    # Simulate the real position wit noisy measurement
    process_noise = np.random.multivariate_normal([0,0],R).reshape(2,1)
    true_pos = A @ mu + B @ u_t[i] + process_noise
    measurement_noise = np.random.multivariate_normal([0,0],Q).reshape(2,1)
    z_t = C @ true_pos + measurement_noise

    # Prediction
    mu_bar = A @ mu + B @ u_t[i]
    sigma_bar = A @ sigma @ A.transpose() + R 

    # Update step 
    K_t = sigma_bar @ C.transpose() @ np.linalg.inv(C @ sigma_bar @ C.transpose() + Q)
    mu = mu_bar + K_t@(z_t - C @ mu_bar) 
    sigma = (np.eye(2) - K_t @ C) @ sigma_bar

    pos_real.append(true_pos.flatten())
    measurements.append(z_t.flatten())
    estimates.append(mu.flatten())

# --- Plotting ---
true_positions = np.array(pos_real)
measurements = np.array(measurements)
estimates = np.array(estimates)

plt.figure(figsize=(10, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'k-', label='True Position')
plt.plot(measurements[:, 0], measurements[:, 1], 'rx', label='Measurements')
plt.plot(estimates[:, 0], estimates[:, 1], 'b--', label='Kalman Estimate')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig("kalman_output.png")

# Error predicted
rmse_predicted = np.sqrt(np.mean((estimates-true_positions)**2))
print(rmse_predicted)

# Error measurement
rmse_measurements = np.sqrt(np.mean((measurements-true_positions)**2))
print(rmse_measurements)
