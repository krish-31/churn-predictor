import pandas as pd
from .drift_metrics import (
    calculate_psi,
    calculate_ks_test,
    calculate_js_divergence
)


class DriftDetector:

    def detect_drift(self, baseline_df: pd.DataFrame, recent_df: pd.DataFrame):
        results = {}
        severe = 0
        warning = 0

        for col in baseline_df.columns:
            if col not in recent_df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(baseline_df[col]):
                continue

            psi = calculate_psi(baseline_df[col], recent_df[col])
            ks = calculate_ks_test(baseline_df[col], recent_df[col])
            js = calculate_js_divergence(baseline_df[col], recent_df[col])

            if psi > 0.25:
                status = "Severe"
                severe += 1
            elif psi > 0.1:
                status = "Warning"
                warning += 1
            else:
                status = "Normal"

            results[col] = {
                "psi": psi,
                "ks_pvalue": ks,
                "js_divergence": js,
                "status": status
            }

        total = max(len(results), 1)

        overall = (
            "Severe" if severe / total > 0.3
            else "Warning" if (severe + warning) / total > 0.1
            else "Normal"
        )

        return {
            "overall_status": overall,
            "features": results
        }