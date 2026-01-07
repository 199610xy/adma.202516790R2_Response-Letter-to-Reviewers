import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Your data
data = [
    # viscosity, ba, frequency, predicted_velocity, uncertainty
    [1, 10, 0.5, 1.0701511, 0.085871704],
    [1, 10, 1, 1.8474572, 0.04071184],
    [1, 10, 2, 3.0587406, 0.24664229],
    [1, 10, 3.5, 4.2636733, 0.09200105],
    [1, 10, 5, 4.800153, 0.12995589],

    [10, 10, 0.5, 1.0479642, 0.04547473],
    [10, 10, 1, 1.6013759, 0.057603955],
    [10, 10, 2, 2.3362281, 0.08094917],
    [10, 10, 3.5, 3.0811615, 0.16320932],
    [10, 10, 5, 3.4682827, 0.16261606],

    [25, 10, 0.5, 0.701007, 0.04338036],
    [25, 10, 1, 0.9572779, 0.036549706],
    [25, 10, 2, 1.2327287, 0.05929856],
    [25, 10, 3.5, 1.6784718, 0.03624813],
    [25, 10, 5, 2.0064025, 0.08086927],

    [50, 10, 0.5, 0.5351178, 0.029881813],
    [50, 10, 1, 0.70257586, 0.022922168],
    [50, 10, 2, 0.9606311, 0.029201124],
    [50, 10, 3.5, 1.1262496, 0.085616246],
    [50, 10, 5, 1.3151946, 0.13596606],

    [100, 10, 0.5, 0.38268498, 0.025582632],
    [100, 10, 1, 0.47191516, 0.02306815],
    [100, 10, 2, 0.7229158, 0.03202234],
    [100, 10, 3.5, 1.0939249, 0.06411118],
    [100, 10, 5, 1.3336176, 0.08521616],

    [150, 10, 0.5, 0.32645276, 0.025059333],
    [150, 10, 1, 0.38370714, 0.022672717],
    [150, 10, 2, 0.562308, 0.021023143],
    [150, 10, 3.5, 0.8992034, 0.041054115],
    [150, 10, 5, 1.2121229, 0.057239488]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['viscosity', 'ba', 'frequency', 'predicted_velocity', 'uncertainty'])

# Frequency points
frequencies = [0.5, 1, 2, 3.5, 5]

# Viscosity values
viscosities = [1, 10, 25, 50, 100, 150]

# Velocity measurement error (standard deviation)
sigma_v = 0.1  # mm/s


def calculate_resolution(df, frequencies, viscosities, sigma_v):
    """
    Calculate the resolution of viscosity inversion using neural network.

    Parameters:
    - df: DataFrame containing predicted velocities
    - frequencies: List of frequency points
    - viscosities: List of viscosity values
    - sigma_v: Velocity measurement error (standard deviation)

    Returns:
    - DataFrame containing resolution results
    """

    # Extract velocity matrix (viscosity × frequency)
    velocity_matrix = np.zeros((len(viscosities), len(frequencies)))
    for i, visc in enumerate(viscosities):
        for j, freq in enumerate(frequencies):
            # Find corresponding data
            row = df[(df['viscosity'] == visc) & (df['frequency'] == freq)]
            if len(row) > 0:
                velocity_matrix[i, j] = row['predicted_velocity'].values[0]

    # Calculate sensitivity matrix (derivative ∂v/∂η)
    # Using finite difference method
    sensitivity_matrix = np.zeros((len(viscosities), len(frequencies)))

    for j in range(len(frequencies)):
        for i in range(len(viscosities)):
            if i == 0:  # First viscosity point (η=1), forward difference
                if i + 1 < len(viscosities):
                    d_eta = viscosities[i + 1] - viscosities[i]
                    d_v = velocity_matrix[i + 1, j] - velocity_matrix[i, j]
                    sensitivity_matrix[i, j] = d_v / d_eta
            elif i == len(viscosities) - 1:  # Last viscosity point (η=150), backward difference
                d_eta = viscosities[i] - viscosities[i - 1]
                d_v = velocity_matrix[i, j] - velocity_matrix[i - 1, j]
                sensitivity_matrix[i, j] = d_v / d_eta
            else:  # Interior points, using central difference (more accurate)
                d_eta_forward = viscosities[i + 1] - viscosities[i]
                d_eta_backward = viscosities[i] - viscosities[i - 1]
                d_v_forward = velocity_matrix[i + 1, j] - velocity_matrix[i, j]
                d_v_backward = velocity_matrix[i, j] - velocity_matrix[i - 1, j]
                # Use weighted average
                sensitivity_matrix[i, j] = (d_v_forward / d_eta_forward + d_v_backward / d_eta_backward) / 2

    # Calculate resolution (error propagation)
    resolutions = []
    for i in range(len(viscosities)):
        # Calculate sum of squared sensitivities
        sum_sq_sensitivity = np.sum(sensitivity_matrix[i, :] ** 2)

        # Calculate viscosity resolution (standard deviation)
        if sum_sq_sensitivity > 0:
            sigma_eta = sigma_v / np.sqrt(sum_sq_sensitivity)
        else:
            sigma_eta = np.inf

        # Calculate relative error
        rel_error = (sigma_eta / viscosities[i]) * 100

        resolutions.append({
            'viscosity': viscosities[i],
            'resolution_cP': sigma_eta,
            'relative_error_%': rel_error,
            'sum_sq_sensitivity': sum_sq_sensitivity
        })

    # Convert to DataFrame
    result_df = pd.DataFrame(resolutions)

    # Add sensitivity information (optional)
    sensitivity_df = pd.DataFrame(
        sensitivity_matrix,
        index=[f'η={v}cP' for v in viscosities],
        columns=[f'f={f}Hz' for f in frequencies]
    )

    return result_df, sensitivity_df


# Calculate resolution
result_df, sensitivity_df = calculate_resolution(df, frequencies, viscosities, sigma_v)

print("=" * 60)
print("Neural Network Viscosity Inversion Resolution Analysis")
print("=" * 60)
print(f"Velocity measurement error σ_v = {sigma_v} mm/s")
print(f"Frequency points: {frequencies} Hz")
print("\nSensitivity Matrix (∂v/∂η, unit: (mm/s)/cP):")
print(sensitivity_df.round(4))
print("\nResolution Results:")

# Display results with 4 decimal places for resolution_cP and 2 decimal places for others
display_df = result_df.copy()
display_df['resolution_cP'] = display_df['resolution_cP'].round(4)
display_df['relative_error_%'] = display_df['relative_error_%'].round(2)
display_df['sum_sq_sensitivity'] = display_df['sum_sq_sensitivity'].round(6)
print(display_df)



# Print detailed mathematical derivation
print("\n" + "=" * 60)
print("Mathematical Derivation Process")
print("=" * 60)
print("1. Local Linear Approximation:")
print("   Assume that near viscosity η, the relationship between velocity v and η is linear:")
print("   v(η) ≈ v(η₀) + (∂v/∂η)|η₀ * (η - η₀)")
print("")
print("2. Error Propagation Formula:")
print("   When inverting viscosity through velocity measurements at multiple frequency points,")
print("   each velocity measurement error is independent, with variance σ_v²")
print("   Total error propagation formula:")
print("   σ_η² = σ_v² / ∑[ (∂v_i/∂η)² ]")
print("   where the summation is over all frequency points i")
print("")
print("3. Derivative Calculation Method:")
print("   Interior points use central difference: ∂v/∂η ≈ [v(η+Δη) - v(η-Δη)] / (2Δη)")
print("   Endpoints use forward/backward difference")
print("")
print("4. Final Resolution:")
print("   σ_η = σ_v / √[∑(∂v_i/∂η)²]")

# Example calculation (take η=1 cP as example)
print("\n" + "=" * 60)
print("Detailed Calculation for η=1 cP:")
print("=" * 60)
eta_index = 0  # η=1 cP
print(f"Viscosity η = {viscosities[eta_index]} cP")
print(f"Sensitivity vector (∂v/∂η):")
for j, freq in enumerate(frequencies):
    sens = sensitivity_df.iloc[eta_index, j]
    print(f"  f={freq}Hz: {sens:.6f} (mm/s)/cP")

sum_sq = np.sum(sensitivity_df.iloc[eta_index, :] ** 2)
print(f"\nSum of squared sensitivities: ∑(∂v/∂η)² = {sum_sq:.6f}")
print(f"Velocity measurement error: σ_v = {sigma_v} mm/s")
resolution = sigma_v / np.sqrt(sum_sq)
print(f"Viscosity resolution: σ_η = {sigma_v} / √{sum_sq:.6f} = {resolution:.4f} cP")