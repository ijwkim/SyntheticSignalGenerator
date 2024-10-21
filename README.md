# Magnetic Sensor Fusion Test Signal Generator

A Python tool for generating synthetic magnetic field (`B`) signals and their rates of change (`dB/dt`) to evaluate the effectiveness of magnetic sensor fusion algorithms.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Example](#example)
- [Sample Plots](#sample-plots)
- [License](#license)

## Features

- **Bumps and Piecewise-Regular Signals:** Generate complex magnetic field signals combining exponential trends and Gaussian-like pulses.
- **Noise Simulation:** Add configurable Gaussian noise to both `B` and `dB/dt` signals.
- **Offset Biases:** Apply customizable offsets to simulate sensor biases or external influences.
- **Visualization:** Optional plotting of generated signals for easy analysis.
- **Modularity:** Separate functions for signal generation and plotting for enhanced flexibility.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/MagneticSensorFusionTestSignal.git
   cd MagneticSensorFusionTestSignal


from signal_generator import generate_test_signals, plot_signals

# Generate synthetic test signals
signals = generate_test_signals()

# Access the generated signals
time = signals['t']                # Time array in seconds
b_true = signals['b_true']         # True magnetic field signal (T)
b_meas = signals['b_meas']         # Measured magnetic field signal with noise (T)
dBdt_true = signals['dBdt_true']   # True rate of change of magnetic field (T/s)
dBdt_meas = signals['dBdt_meas']   # Measured rate of change with noise and offset (T/s)
offset = signals['offset']         # Applied offset to dB/dt (T/s)
measured_sigma = signals['measured_sigma']               # Noise standard deviation for B
diff_measured_sigma = signals['diff_measured_sigma']       # Noise standard deviation for dB/dt

# Plot the generated signals
plot_signals(t, b_true, b_meas, dBdt_true, dBdt_meas, offset, measured_sigma, diff_measured_sigma)
