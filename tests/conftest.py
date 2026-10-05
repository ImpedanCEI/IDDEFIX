"""Shared pytest options and diagnostic fixtures."""

from collections.abc import Mapping

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--debug-plots",
        action="store_true",
        help="Show plots of the values compared by analytical regression tests",
    )


@pytest.fixture
def debug_plot(request):
    """Plot asserted data only when pytest is run with ``--debug-plots``."""
    if not request.config.getoption("--debug-plots"):
        return lambda *args, **kwargs: None

    import matplotlib.pyplot as plt
    import numpy as np

    def plot(
        x,
        actual,
        expected,
        *,
        title,
        xlabel,
        ylabel,
        expected_label="Analytical reference",
    ):
        x = np.asarray(x)
        expected = np.asarray(expected)
        actual_curves = actual if isinstance(actual, Mapping) else {"Computed": actual}
        curves = {label: np.asarray(values) for label, values in actual_curves.items()}
        is_complex = np.iscomplexobj(expected) or any(
            np.iscomplexobj(values) for values in curves.values()
        )
        components = (
            (("Real", np.real), ("Imaginary", np.imag))
            if is_complex
            else ((None, lambda values: values),)
        )
        fig, axes = plt.subplots(
            len(components), 1, figsize=(9, 4 * len(components)), squeeze=False
        )

        for ax, (component_name, component) in zip(axes[:, 0], components):
            ax.plot(x, component(expected), color="black", lw=2, label=expected_label)
            for label, values in curves.items():
                ax.plot(x, component(values), "--", lw=1.5, label=label)
            subtitle = f" — {component_name.lower()} part" if component_name else ""
            ax.set(
                title=f"{request.node.name}: {title}{subtitle}",
                xlabel=xlabel,
                ylabel=ylabel,
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        plt.show()
        plt.close(fig)

    return plot
