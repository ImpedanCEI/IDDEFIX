"""Wake deconvolution workflows and their sensitivity to sampling choices.

The tests use SPS and optional WAKIS data to check spectral reconstruction.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from packaging.version import Version
from scipy.constants import c as c_light

import iddefix

if Version(np.__version__) >= Version("2.0.0"):
    _trapezoid = np.trapezoid
else:
    _trapezoid = np.trapz


@pytest.fixture(scope="module")
def wake_data_time():
    """Load the real SPS wake potential data for testing."""
    data = np.loadtxt(
        "examples/data/004_SPS_model_transitions_q26.txt",
        comments="#",
        delimiter="\t",
    )
    time = data[:, 0] * 1e-9  # convert ns → s
    wake = data[:, 2]  # in V/C and time sampling
    return time, wake


@pytest.fixture(scope="module")
def wake_data_from_wakis():
    """Load wake data from wakis simulations. Skip if wakis is not installed."""
    wakis = pytest.importorskip(
        "wakis", reason="wakis not installed; skipping wake tests"
    )
    wake = wakis.WakeSolver(save=False)
    wake.load_results("tests/data/002_wakis_example/")
    data_time = wake.s / c_light  # convert from m to s
    data_wake = wake.WP / 1e12 / c_light
    return data_time, data_wake


def compute_norm(f, Z):
    """Helper to compute a scalar 'size' measure of impedance spectrum."""
    return _trapezoid(np.abs(Z), f) / (f[-1] - f[0])


def test_deconvolution_rolloff():
    """A positive rolloff extends the spectrum and applies the raised cosine."""
    sigma = 0.1e-9
    fmax = 1e9
    rolloff = 0.5
    time = (np.arange(512) - 256) * 1e-11
    gaussian = iddefix.gaussian_bunch(time, sigma)

    f_default, _ = iddefix.compute_deconvolution(time, gaussian, sigma, fmax=fmax)
    f_rolloff, impedance = iddefix.compute_deconvolution(
        time, gaussian, sigma, fmax=fmax, rolloff=rolloff
    )

    cutoff = (1 + rolloff) * fmax
    assert f_default[-1] < fmax
    assert np.any((f_rolloff > fmax) & (f_rolloff < cutoff))

    expected = np.ones_like(f_rolloff)
    tapered = f_rolloff > fmax
    expected[tapered] = 0.5 * (
        1 + np.cos(np.pi * (f_rolloff[tapered] - fmax) / (cutoff - fmax))
    )
    np.testing.assert_allclose(impedance, expected, rtol=1e-12, atol=1e-12)


def test_deconvolution_fmax_defaults_from_sigma():
    """An omitted fmax gives the same result as 1/(3 sigma)."""
    sigma = 0.12e-9
    time = (np.arange(256) - 128) * 8e-12
    potential = np.exp(-0.5 * (time / sigma) ** 2)
    derived_fmax = 1 / (3 * sigma)

    default_f, default_z = iddefix.compute_deconvolution(time, potential, sigma)
    explicit_f, explicit_z = iddefix.compute_deconvolution(
        time, potential, sigma, fmax=derived_fmax
    )

    np.testing.assert_array_equal(default_f, explicit_f)
    np.testing.assert_array_equal(default_z, explicit_z)


def test_gaussian_bunch_spectrum_starts_at_dc_and_stops_at_fmax():
    """The Gaussian spectrum includes DC and excludes frequencies at fmax."""
    time = (np.arange(128) - 64) * 2e-12
    sigma = 0.1e-9
    fmax = 4e9

    frequency, spectrum = iddefix.gaussian_bunch_spectrum(time, sigma, fmax, samples=16)

    assert frequency[0] == 0
    assert np.all(frequency >= 0)
    assert np.all(frequency < fmax)
    assert frequency.shape == spectrum.shape
    np.testing.assert_allclose(spectrum[0], np.sum(iddefix.gaussian_bunch(time, sigma)))


def stability_with_samples(time_data, wake_data, plot=False):
    sigma = 1e-10
    samples_list = [501, 1001, 5001, 10001]

    norms = []
    spectra = {}
    for s in samples_list:
        f, Z = iddefix.compute_deconvolution(time_data, wake_data, sigma, samples=s)
        norms.append(compute_norm(f, Z))
        spectra[s] = (f, Z)

    # Compare ratios of norms — they should not diverge
    ratios = np.array(norms) / norms[-1]
    max_dev = np.max(np.abs(ratios - 1))

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for n, (f, Z) in spectra.items():
            ax1.plot(f, np.abs(Z), label=f"samples={n}")
        ax1.set_title("Impedance magnitude vs samples")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("|Z(f)| [Ohm]")
        ax1.legend()
        ax1.grid(True, which="both")

        ax2.plot(samples_list, norms, "o-", label="∫|Z| df / Δf")
        ax2.set_title("Convergence of impedance norm with samples")
        ax2.set_xlabel("Number of samples")
        ax2.set_ylabel("Integrated impedance [a.u.]")
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    assert max_dev < 0.05, f"Impedance norm varies too much with samples: {ratios}"


@pytest.mark.parametrize(
    "wake_data",
    [
        pytest.param("wake_data_time", id="sps-data"),
        pytest.param("wake_data_from_wakis", id="wakis-data"),
    ],
)
def test_stability_with_samples(wake_data, request):
    """The impedance should converge as the number of FFT samples increases."""
    time, wake = request.getfixturevalue(wake_data)
    stability_with_samples(time, wake, plot=False)


def stability_with_sigma(data_time, data_wake, plot=False):
    sigmas = [0.5e-10, 1e-10, 2e-10]

    norms = []
    spectra = {}

    for s in sigmas:
        f, Z = iddefix.compute_deconvolution(data_time, data_wake, s)
        norms.append(compute_norm(f, Z))
        spectra[s] = (f, Z)

    ratios = np.array(norms) / norms[1]  # compare to central sigma
    max_dev = np.max(np.abs(ratios - 1))

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for s, (f, Z) in spectra.items():
            ax1.loglog(f, np.abs(Z), label=f"sigma={s * 1e9:.2f} ns")
        ax1.set_title("Impedance magnitude vs sigma")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("|Z(f)| [Ohm]")
        ax1.legend()
        ax1.grid(True, which="both")

        ax2.plot([s * 1e9 for s in sigmas], norms, "o-", label="∫|Z| df / Δf")
        ax2.set_title("Sensitivity of impedance norm to sigma")
        ax2.set_xlabel("Sigma [ns]")
        ax2.set_ylabel("Integrated impedance [a.u.]")
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    assert max_dev < 20, f"Impedance varies too strongly with sigma: {ratios}"


@pytest.mark.parametrize(
    "wake_data",
    [
        pytest.param("wake_data_time", id="sps-data"),
        pytest.param("wake_data_from_wakis", id="wakis-data"),
    ],
)
def test_stability_with_sigma(wake_data, request):
    """Changing sigma moderately should not cause large jumps in the impedance magnitude."""
    time, wake = request.getfixturevalue(wake_data)
    stability_with_sigma(time, wake, plot=False)


# ==============================================================
# Optional diagnostic plotting when running directly
# ==============================================================

if __name__ == "__main__":
    print("Running impedance stability diagnostics with real SPS data...\n")
    data = np.loadtxt(
        "../examples/data/004_SPS_model_transitions_q26.txt",
        comments="#",
        delimiter="\t",
    )
    data_time = data[:, 0] * 1e-9  # convert ns → s
    data_wake = data[:, 2]  # in V/C and time sampling

    # Call test functions with plotting enabled
    stability_with_samples(data_time, data_wake, plot=True)
    stability_with_sigma(data_time, data_wake, plot=True)

    print("\n✅ All stability diagnostics completed successfully.")

    print("Running impedance stability diagnostics with Wakis data...\n")

    from wakis import WakeSolver

    wake = WakeSolver(save=False)
    wake.load_results("data/002_wakis_example/")
    data_time = wake.s / c_light  # convert from m to s
    data_wake = wake.WP / 1e12 / c_light

    # Call test functions with plotting enabled
    stability_with_samples(data_time, data_wake, plot=True)
    stability_with_sigma(data_time, data_wake, plot=True)

    print("\n✅ All stability diagnostics completed successfully.")
