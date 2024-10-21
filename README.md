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
   git clone https://github.com/ijwkim/MagneticSensorFusionTestSignal.git
   cd MagneticSensorFusionTestSignal
   ```

### Usage
Import the `generate_test_signals` function from `signal_generator.py` to create synthetic signals. You can also visualize the signals by using the `plot_signals` function.

```python
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
```

### Parameters
The `generate_test_signals` function initializes and returns a dictionary of synthetic signals with the following default parameters:

- `sampling_freq` (float): Sampling frequency in Hz. Default: 1000 Hz
- `duration_hours` (int): Duration of the signal in hours. Default: 1 hour
- `measured_sigma` (float): Standard deviation of noise added to the magnetic field (B). Default: 0.1 T
- `diff_measured_sigma` (float): Standard deviation of noise added to the rate of change of the magnetic field (dB/dt). Default: 0.02 T/s
- These parameters can be adjusted within the `generate_test_signals` function as needed.

### Example
Here is an example script demonstrating how to generate and plot the synthetic signals:

```python
from signal_generator import generate_test_signals, plot_signals

def main():
    # Generate synthetic test signals
    signals = generate_test_signals()
    
    # Extract signals from the returned dictionary
    t = signals['t']
    b_true = signals['b_true']
    b_meas = signals['b_meas']
    dBdt_true = signals['dBdt_true']
    dBdt_meas = signals['dBdt_meas']
    offset = signals['offset']
    measured_sigma = signals['measured_sigma']
    diff_measured_sigma = signals['diff_measured_sigma']
    
    # Plot the signals
    plot_signals(t, b_true, b_meas, dBdt_true, dBdt_meas, offset, measured_sigma, diff_measured_sigma)

if __name__ == "__main__":
    main()
```
