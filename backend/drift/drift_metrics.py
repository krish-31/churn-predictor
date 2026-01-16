import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


def calculate_psi(expected, actual, bins=10):
    """
    Population Stability Index (PSI)
    Measures shift in feature distribution
    """
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, bins + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    psi_value = 0.0

    for i in range(len(expected_perc) - 1):
        expected_count = np.sum(
            (expected >= expected_perc[i]) & (expected < expected_perc[i + 1])
        ) / len(expected)

        actual_count = np.sum(
            (actual >= actual_perc[i]) & (actual < actual_perc[i + 1])
        ) / len(actual)

        expected_count = max(expected_count, 1e-6)
        actual_count = max(actual_count, 1e-6)

        psi_value += (actual_count - expected_count) * np.log(
            actual_count / expected_count
        )

    return round(psi_value, 5)


def calculate_ks_test(expected, actual):
    """
    Kolmogorov-Smirnov Test
    Returns p-value
    """
    _, p_value = ks_2samp(expected, actual)
    return round(p_value, 5)


def calculate_js_divergence(expected, actual, bins=10):
    """
    Jensen-Shannon Divergence
    Measures probabilistic distance
    """
    expected_hist, _ = np.histogram(expected, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bins, density=True)

    expected_hist += 1e-6
    actual_hist += 1e-6

    js_value = jensenshannon(expected_hist, actual_hist)
    return round(float(js_value), 5)
