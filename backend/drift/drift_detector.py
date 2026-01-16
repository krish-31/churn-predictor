import pandas as pd
from typing import Dict

from .drift_metrics import (
    calculate_psi,
    calculate_ks_test,
    calculate_js_divergence
)


class DriftDetector:
    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_severe: float = 0.25,
        ks_warning: float = 0.05,
        ks_severe: float = 0.01
    ):
        # Thresholds are configurable (NOT hard-coded in logic)
        self.psi_warning = psi_warning
        self.psi_severe = psi_severe
        self.ks_warning = ks_warning
        self.ks_severe = ks_severe

    def evaluate_feature(self, baseline: pd.Series, recent: pd.Series) -> Dict:
        psi = calculate_psi(baseline, recent)
        ks_pvalue = calculate_ks_test(baseline, recent)
        js_divergence = calculate_js_divergence(baseline, recent)

        # Drift severity logic
        if psi >= self.psi_severe or ks_pvalue <= self.ks_severe:
            status = "Severe"
        elif psi >= self.psi_warning or ks_pvalue <= self.ks_warning:
            status = "Warning"
        else:
            status = "Normal"

        return {
            "psi": psi,
            "ks_pvalue": ks_pvalue,
            "js_divergence": js_divergence,
            "status": status
        }

    def detect_drift(
        self,
        baseline_df: pd.DataFrame,
        recent_df: pd.DataFrame
    ) -> Dict:
        feature_results = {}
        severe_count = 0
        warning_count = 0

        common_features = [
            col for col in baseline_df.columns
            if col in recent_df.columns
        ]

        for feature in common_features:
            # Skip non-numeric features safely
            if not pd.api.types.is_numeric_dtype(baseline_df[feature]):
                continue

            result = self.evaluate_feature(
                baseline_df[feature].dropna(),
                recent_df[feature].dropna()
            )

            feature_results[feature] = result

            if result["status"] == "Severe":
                severe_count += 1
            elif result["status"] == "Warning":
                warning_count += 1

        total_features = len(feature_results)

        # Overall drift decision
        if severe_count / max(total_features, 1) > 0.3:
            overall_status = "Severe"
        elif (severe_count + warning_count) / max(total_features, 1) > 0.1:
            overall_status = "Warning"
        else:
            overall_status = "Normal"

        return {
            "overall_status": overall_status,
            "total_features_checked": total_features,
            "severe_features": severe_count,
            "warning_features": warning_count,
            "features": feature_results
        }
