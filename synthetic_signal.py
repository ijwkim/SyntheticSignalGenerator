import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

def piece_regular_signal(n):
    """
    Generates a piece-regular signal similar to the one in the MATLAB WaveLab toolbox.
    
    This function constructs a complex signal by combining exponential trends, Gaussian-like 
    pulses (bumps), and constant offsets to simulate realistic magnetic field variations.
    
    Parameters:
    - n (int): Length of the signal (number of samples).
    
    Returns:
    - f (np.ndarray): Generated piece-regular signal.
    """
    f = np.zeros(n)
    
    # Define segment lengths based on fractions of the total signal length
    n_12 = int(np.fix(n / 12))
    n_7 = int(np.fix(n / 7))
    n_5 = int(np.fix(n / 5))
    n_3 = int(np.fix(n / 3))
    n_2 = int(np.fix(n / 2))
    n_20 = int(np.fix(n / 20))

    # Generate the 'bumps' component and scale it
    f1 = -15 * demo_signal(n, 'bumps')
    
    # Exponential decay component
    t = np.arange(1, n_12 + 1) / n_12
    f2 = -np.exp(4 * t)
    
    # Exponential growth component
    t = np.arange(1, n_7 + 1) / n_7
    f5 = np.exp(4 * t) - np.exp(4)
    
    # Gaussian-like pulses
    t = np.arange(1, n_3 + 1) / n_3
    fma = 6 / 40
    f6 = -70 * np.exp(-((t - 0.5) ** 2) / (2 * fma ** 2))
    
    # Assemble the signal by assigning different components to specific segments
    f[:n_7] = f6[:n_7]
    f[n_7:n_5] = 0.5 * f6[n_7:n_5]
    f[n_5:n_3] = f6[n_5:n_3]
    f[n_3:n_2] = f1[n_3:n_2]
    
    f[n_2:n_2 + n_12] = f2
    f[n_2 + n_12:n_2 + 2 * n_12] = f2[::-1]
    
    # Add constant negative offsets
    f[n_2 + 2 * n_12:n_2 + 2 * n_12 + n_20] = -25 * np.ones(n_20)
    
    # Insert the exponential growth component
    k = n_2 + 2 * n_12 + n_20
    f[k:k + n_7] = f5[:n_7]
    
    # Handle remaining samples by mirroring the initial part of the signal
    diff = n - 5 * n_5
    f[5 * n_5:n] = f[diff - 1::-1]
    
    # Zero-mean the signal to remove any DC bias
    bias = np.sum(f) / n
    f = bias - f
    
    return f

def demo_signal(n, name='bumps'):
    """
    Generates demo signals, including 'Bumps'.
    
    This helper function creates specific signal components used in generating the piece-regular signal.
    
    Parameters:
    - n (int): Number of samples in the signal.
    - name (str): Type of the signal to generate ('bumps' by default).
    
    Returns:
    - f (np.ndarray): Generated signal based on the specified type.
    
    Raises:
    - NotImplementedError: If an unsupported signal type is requested.
    """
    # Create a time array from 1/n to 1 with n samples
    t = np.arange(1 / n, 1 + 1 / n, 1 / n)
    
    if name.lower() == 'bumps':
        # Define parameters for the 'bumps' signal
        t0s = [.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]
        hs = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2]
        ws = [.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005]
        f = np.zeros_like(t)
        
        # Superimpose multiple 'bumps' onto the signal
        for t0, h, w in zip(t0s, hs, ws):
            f += h / (1 + np.abs((t - t0) / w))**4
        return f
    else:
        raise NotImplementedError(f"Signal type '{name}' is not implemented.")

def offset_signal_long_pulse(t):
    """
    Generates an offset signal using a combination of long pulses modeled by hyperbolic tangent functions.
    
    This function creates realistic bias variations over time to simulate sensor biases or external influences.
    
    Parameters:
    - t (np.ndarray): Time array.
    
    Returns:
    - offset (np.ndarray): Generated offset signal.
    """
    offset = (
        (np.tanh((t - 60 * 5) / 40) + 1) / 2 +
        (np.tanh((t - 60 * 10) / 40) + 1) / 2 +
        (np.tanh((t - 60 * 15) / 40) + 1) / 2 +
        (np.tanh((t - 60 * 20) / 40) + 1) / 2 -
        (np.tanh((t - 60 * 40) / 40) + 1) / 2 -
        (np.tanh((t - 60 * 45) / 40) + 1) / 2 -
        (np.tanh((t - 60 * 50) / 40) + 1) / 2 -
        (np.tanh((t - 60 * 55) / 40) + 1) / 2
    ) * 3 / 50  # Scale the offset appropriately
    
    return offset

def generate_test_signals():
    """
    Generates synthetic test signals for magnetic sensor fusion analysis.
    
    This function creates true and measured magnetic field (`B`) signals, calculates their rates of change (`dB/dt`),
    adds Gaussian noise, and applies an offset to simulate realistic sensor measurements.
    
    Returns:
    - signals (dict): A dictionary containing:
        - 't': Time array in seconds.
        - 'b_true': True magnetic field signal.
        - 'b_meas': Measured magnetic field signal with noise.
        - 'dBdt_true': True rate of change of magnetic field.
        - 'dBdt_meas': Measured rate of change of magnetic field with noise and offset.
        - 'offset': Applied offset to `dBdt`.
        - 'measured_sigma': Standard deviation of `B` noise.
        - 'diff_measured_sigma': Standard deviation of `dBdt` noise.
    """
    # Define sampling parameters
    sampling_freq = 1e3  # Sampling frequency in Hz (1000 Hz)
    duration_hours = 1   # Duration of the signal in hours
    datasize = int(sampling_freq * 60 * 60 * duration_hours)  # Total number of samples
    t = np.arange(datasize) / sampling_freq  # Time array in seconds
    
    # Define noise parameters
    measured_sigma = 0.1       # Standard deviation of B noise (T)
    diff_measured_sigma = 0.02  # Standard deviation of dB/dt noise (T/s)
    
    # Generate true magnetic field signal using piece-regular components
    b_true = piece_regular_signal(t.size + 1) / 100  # Scale down the signal for realistic values
    
    # Simulate measured magnetic field by adding Gaussian noise
    b_meas = b_true[:-1] + np.random.normal(0, measured_sigma, t.shape)
    
    # Generate offset signal to simulate sensor bias
    offset = offset_signal_long_pulse(t)
    
    # Calculate true rate of change of magnetic field (dB/dt)
    dBdt_true = np.diff(b_true) * sampling_freq  # dB/dt in T/s
    
    # Simulate measured dB/dt by adding Gaussian noise and offset
    dBdt_meas = dBdt_true + np.random.normal(0, diff_measured_sigma, t.shape) + offset
    
    # Adjust b_true to match the measured signal length
    b_true = b_true[:-1]
    
    return {
        't': t,
        'b_true': b_true,
        'b_meas': b_meas,
        'dBdt_true': dBdt_true,
        'dBdt_meas': dBdt_meas,
        'offset': offset,
        'measured_sigma': measured_sigma,
        'diff_measured_sigma': diff_measured_sigma
    }

def plot_signals(t, b_true, b_meas, dBdt_true, dBdt_meas, offset, measured_sigma, diff_measured_sigma):
    """
    Plots the generated magnetic field and its rate of change signals for visualization.
    
    The plot includes:
    - True and measured magnetic field (`B`) over time.
    - True and measured rate of change of magnetic field (`dB/dt`) over time.
    - Applied offset to the `dB/dt` measurements.
    
    Parameters:
    - t (np.ndarray): Time array in seconds.
    - b_true (np.ndarray): True magnetic field signal.
    - b_meas (np.ndarray): Measured magnetic field signal with noise.
    - dBdt_true (np.ndarray): True rate of change of magnetic field.
    - dBdt_meas (np.ndarray): Measured rate of change of magnetic field with noise and offset.
    - offset (np.ndarray): Offset applied to `dB/dt`.
    - measured_sigma (float): Standard deviation of `B` noise.
    - diff_measured_sigma (float): Standard deviation of `dB/dt` noise.
    """
    # Initialize the figure and define grid specifications
    fig = plt.figure(dpi=200, figsize=(11, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1.0])

    # Subplot (a): Magnetic Field B over time
    ax3 = fig.add_subplot(gs[:, 0])
    ax3.plot(t / 60, b_meas, lw=1, alpha=0.5, label=r"Hall probe $B$ [$\sigma_{H}=$" + f"{measured_sigma} T]")
    ax3.plot(t / 60, b_true, 'k--', label=r"True $B$", zorder=10)
    ax3.set_xlim([0, 60])  # Time in minutes
    ax3.set_ylim([-0.7, 1.1])  # Magnetic field in Tesla
    ax3.set_xlabel('Time [min]')
    ax3.set_ylabel(r"Magnetic Field $B$ [T]")
    ax3.text(0.01, 0.99, '(a)', transform=ax3.transAxes, fontsize=12, va='top')

    # Subplot (b): Rate of Change dB/dt over time
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t / 60, dBdt_meas, color='C0', linestyle='-', lw=1, alpha=0.5,
             label=r"Coil $\dot{B}$ [$\sigma_C=$" + f"{diff_measured_sigma} T/s]")
    ax1.plot(t / 60, dBdt_true, color='C1', linestyle='--', label=r"True $\dot{B}$")
    ax1.plot(t / 60, offset, color='k', linestyle='-.', label="True bias")
    ax1.set_ylim([-0.1, 0.5])  # Rate of change in Tesla per second
    ax1.set_xlim([0, 60])  # Time in minutes
    ax1.set_ylabel(r"$\dot{B}$ [T/s]")
    ax1.text(0.01, 0.98, '(b)', transform=ax1.transAxes, fontsize=12, va='top')

    # Subplot (c): Offset over time
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t / 60, offset, color='k', linestyle='--', label="True bias")
    ax2.set_ylim([-0.1, 0.5])  # Offset in Tesla per second
    ax2.set_xlim([0, 60])  # Time in minutes
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel(r"Bias [T/s]")
    ax2.text(0.01, 0.98, '(c)', transform=ax2.transAxes, fontsize=12, va='top')

    # Configure legends for each subplot
    ax1.legend(loc='upper left', fontsize=10.5, frameon=False, ncol=1,
               bbox_to_anchor=(0.05, 0.1, 0.1, 0.95))
    ax2.legend(loc='upper left', fontsize=10.5, frameon=False, ncol=1,
               bbox_to_anchor=(0.05, 0.1, 0.1, 0.92))
    ax3.legend(loc='upper left', fontsize=10.5, frameon=False, ncol=2,
               bbox_to_anchor=(0.05, 0.1, 0.1, 0.92))

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("one_hour_long_test_signal.png")  # Uncomment to save the plot as an image
    plt.show()

def example_usage():
    """
    Demonstrates example usage of the signal generation and plotting functions.
    
    This function generates synthetic signals and visualizes them using the plotting function.
    """
    # Generate synthetic test signals
    signals = generate_test_signals()
    
    # Plot the generated signals
    plot_signals(
        signals['t'],
        signals['b_true'],
        signals['b_meas'],
        signals['dBdt_true'],
        signals['dBdt_meas'],
        signals['offset'],
        signals['measured_sigma'],
        signals['diff_measured_sigma']
    )

if __name__ == "__main__":
    example_usage()
