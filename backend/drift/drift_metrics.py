import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(expected, actual, bins=10):
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc)
        * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )
    return float(psi)


def calculate_ks_test(expected, actual):
    _, p_value = ks_2samp(expected, actual)
    return float(p_value)


def calculate_js_divergence(expected, actual, bins=10):
    expected = expected.dropna()
    actual = actual.dropna()

    hist_expected, _ = np.histogram(expected, bins=bins, density=True)
    hist_actual, _ = np.histogram(actual, bins=bins, density=True)

    hist_expected += 1e-6
    hist_actual += 1e-6

    m = 0.5 * (hist_expected + hist_actual)
    js = 0.5 * (
        np.sum(hist_expected * np.log(hist_expected / m)) +
        np.sum(hist_actual * np.log(hist_actual / m))
    )
    return float(js)