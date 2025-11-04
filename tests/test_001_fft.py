import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO
import pytest
from scipy.constants import c as c_light

sys.path.append('../')
import iddefix

def _load_data():
    """Load and prepare data once for all tests."""
    data_wake_potential = np.loadtxt(
        '../examples/data/004_SPS_model_transitions_q26.txt',
        comments='#', delimiter='\t')
    
    data_wake_time = data_wake_potential[:, 0] * 1e-9  # [s]
    data_wake_dipolar = data_wake_potential[:, 2]
    sigma = 1e-10

    DE_model = iddefix.EvolutionaryAlgorithm(
        data_wake_time,
        data_wake_dipolar * c_light,
        N_resonators=10,
        parameterBounds=None,
        plane='transverse',
        fitFunction='wake potential',
        sigma=sigma
    )

    # Preload DE parameters
    data_str = StringIO('''
        1     |        2.22e+00        |      76.87       |    1.005e+09
        2     |        7.62e+00        |      138.95      |    1.176e+09
        3     |        1.15e+00        |      15.49       |    1.268e+09
        4     |        1.19e+00        |      39.99       |    1.657e+09
        5     |        1.54e+00        |      169.72      |    2.075e+09
        6     |        1.79e+00        |      177.73      |    2.199e+09
        7     |        1.67e+00        |      53.54       |    2.251e+09
        8     |        1.87e+00        |      39.01       |    2.431e+09
        9     |        1.84e+00        |       5.01       |    2.675e+09
        10    |        1.99e+00        |      178.88      |    2.908e+09
        11    |        1.99e+00        |      38.55       |    3.184e+09
    '''.strip())
    
    DE_model.minimizationParameters = np.loadtxt(
        data_str, skiprows=0, usecols=(1, 2, 3),
        delimiter='|', dtype=float
    ).flatten()

    return DE_model, data_wake_time, data_wake_dipolar, sigma

# ---------- Pytest fixture ----------
@pytest.fixture(scope="module")
def load_data():
    return _load_data()

def test_compare_wakes(load_data, plot=False, adaptative=False):
    """Check that neffint wake and DE model wake are consistent."""
    DE_model, _, _, _ = load_data
    time = np.linspace(1e-11, 50e-9, 1000)
    f_fd = np.linspace(0, 5e9, 1000)

    global t, W
    Z_fd = DE_model.get_impedance(frequency_data=f_fd, wakelength=None)
    t, W = iddefix.compute_ineffint(
        f_fd, Z_fd, times=time, plane='transverse', adaptative=adaptative,
    )
    W_de = DE_model.get_wake(time)

    # Test numerical similarity (correlation > 0.95)
    corr = np.corrcoef(W_de, W)[0, 1]
    assert corr > 0.95, f"Wake correlation too low: {corr:.3f}"

    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time, W_de, lw=2, c='tab:red', label='DE wake')
        ax.plot(time, W, lw=1.5, c='tab:blue', ls='--', label='neffint wake')
        ax.legend(fontsize=14)
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('Wake [V/C/m]')
        fig.tight_layout()
        plt.show()
        fig.savefig('001_compare_wakes_adaptative.png')


def test_compare_impedances(load_data, plot=False, adaptative=False):
    """Compare impedance computed from DE model, FFT, and neffint."""
    DE_model, _, _, _ = load_data
    f_de = np.linspace(1, 5e9, 10000)
    time = np.linspace(0, 50e-9, 1000)
    Z_de = DE_model.get_impedance(frequency_data=f_de, wakelength=None)
    W_de = DE_model.get_wake(time)

    f_fft, Z_fft = iddefix.compute_fft(time, W_de/c_light, fmax=5e9)

    f_inft, Z_inft = iddefix.compute_fft(time, W/c_light, fmax=5e9)

    f_nft, Z_nft = iddefix.compute_neffint(time, DE_model.get_wake(time),
                                    frequencies=f_de,
                                    adaptative=adaptative)

    Z_fft *= 1j #transverse
    Z_inft *= 1j #transverse
    Z_nft *= 1j #transverse


    # Test that |Z| distributions are roughly consistent
    rel_error = np.mean(np.abs(np.abs(Z_nft) - np.abs(Z_de))) / np.mean(np.abs(Z_de))
    assert rel_error < 0.1, f"Relative impedance error too high: {rel_error:.2%}"

    if plot:
        fig = plt.figure(figsize=(12, 7))
        plt.plot(f_de, np.real(Z_de), color='tab:red', lw=3, alpha=0.7, label='Fully decayed real impedance')
        plt.plot(f_de, np.imag(Z_de), color='tab:blue', lw=3, alpha=0.7, label='Fully decayed imag. impedance')
        plt.plot(f_de, np.abs(Z_de), color='tab:green', lw=3, alpha=0.7, label='Fully decayed Abs. impedance')

        plt.plot(f_fft, np.real(Z_fft), color='tab:red', ls='--', label='numpy FFT real impedance')
        plt.plot(f_fft, np.imag(Z_fft), color='tab:blue', ls='--', label='numpy FFT imag. impedance')
        plt.plot(f_fft, np.abs(Z_fft), color='tab:green', ls='--', label='numpy FFT Abs. impedance')

        plt.plot(f_inft, np.real(Z_inft), color='tab:red', ls='-.', label='iNeffint real impedance')
        plt.plot(f_inft, np.imag(Z_inft), color='tab:blue', ls='-.', label='iNeffint imag. impedance')
        plt.plot(f_inft, np.abs(Z_inft), color='tab:green', ls='-.', label='iNeffint Abs. impedance')

        plt.plot(f_nft, np.real(Z_nft), color='tab:red', ls=':', label='Neffint real impedance')
        plt.plot(f_nft, np.imag(Z_nft), color='tab:blue',  ls=':', label='Neffint imag. impedance')
        plt.plot(f_nft, np.abs(Z_nft), color='tab:green',  ls=':', label='Neffint Abs. impedance')

        plt.legend()
        plt.xlabel('f [Hz]')
        plt.ylabel('$Z_{Transverse}$ [$\Omega$]')
        plt.show()
        fig.savefig('001_compare_imp_adaptative.png')


if __name__ == "__main__":
    # Run with plots for visual inspection
    print("Running wake and impedance comparison plots...")
    adaptative = False  # Set to True to use adaptative neffint
    test_compare_wakes(_load_data(), plot=True, adaptative=adaptative)
    test_compare_impedances(_load_data(), plot=True, adaptative=adaptative)
