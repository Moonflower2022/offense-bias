import json
import os
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
#  Paper‑friendly version of the ModelBiasAnalyzer
#  ------------------------------------------------
#  Key changes (vs. your original script)
#  1.  Horizontal bars & trimmed views (top‑k / bottom‑k + "Other") so all labels are
#      readable in print.
#  2.  Automatic figure height (0.25 inch per row, capped) and 600 dpi output.
#  3.  Log‑scaled sample‑count chart _and_ a Lorenz curve to show inequality succinctly.
#  4.  Outlier‑only annotations removed – colour encodes significance instead.
#  5.  A new `create_paper_plots()` method that drops all figures directly to a folder
#      ready for LaTeX / Word import.  No interactive libraries required.
#  6.  Modified middle-k selection to be closest to target value instead of mean.
# ----------------------------------------------------------------------------------

class ModelBiasAnalyzerPaper:
    """Comprehensive model performance & bias analysis with paper‑ready plots."""

    # -------------------------------------------------------------------------
    #  Initialization & data extraction (same as original)
    # -------------------------------------------------------------------------
    def __init__(self, data_path: str = None, data_dict: dict = None):
        if data_dict:
            self.data = data_dict
        elif data_path:
            with open(data_path, "r", encoding="utf‑8") as f:
                self.data = json.load(f)
        else:
            raise ValueError("Either data_path or data_dict must be provided")

        self.overall_metrics = self._extract_overall_metrics()
        self.country_metrics = self._extract_country_metrics()
        self.bias_analysis: Dict = {}

    # ---------------- basic metric extraction --------------------------------
    def _extract_overall_metrics(self) -> Dict:
        return {
            "total_samples": self.data["total_samples"],
            "accuracy": self.data["accuracy"],
            "precision": self.data["precision"],
            "recall": self.data["recall"],
            "f1_score": self.data["f1_score"],
            "confusion_matrix": self.data["confusion_matrix"],
        }

    def _extract_country_metrics(self) -> pd.DataFrame:
        country_data = []
        for country, m in self.data["country_prediction_rates"].items():
            country_data.append(
                {
                    "country": country,
                    "total_samples": m["total_valid_predictions"],
                    "predicted_positive": m["predicted_positive_count"],
                    "predicted_negative": m["predicted_negative_count"],
                    "predicted_positive_rate": m["predicted_positive_rate"],
                    "predicted_negative_rate": m["predicted_negative_rate"],
                    "ground_truth_positive_rate": m["ground_truth_positive_rate"],
                    "accuracy": m["accuracy"],
                    "true_positive": m["true_positive_count"],
                    "true_negative": m["true_negative_count"],
                    "false_positive": m["false_positive_count"],
                    "false_negative": m["false_negative_count"],
                }
            )
        df = pd.DataFrame(country_data)

        # extra metrics
        df["precision"] = df.apply(
            lambda r: r.true_positive / (r.true_positive + r.false_positive)
            if (r.true_positive + r.false_positive) > 0
            else 0,
            axis=1,
        )
        df["recall"] = df.apply(
            lambda r: r.true_positive / (r.true_positive + r.false_negative)
            if (r.true_positive + r.false_negative) > 0
            else 0,
            axis=1,
        )
        df["specificity"] = df.apply(
            lambda r: r.true_negative / (r.true_negative + r.false_positive)
            if (r.true_negative + r.false_positive) > 0
            else 0,
            axis=1,
        )
        df["false_positive_rate"] = df.apply(
            lambda r: r.false_positive / (r.false_positive + r.true_negative)
            if (r.false_positive + r.true_negative) > 0
            else 0,
            axis=1,
        )
        return df

    # ---------------- high‑level bias helpers --------------------------------
    def analyze_sample_distribution(self):
        total_samples = self.country_metrics["total_samples"].sum()
        distribution = {
            "total_samples": total_samples,
            "country_sample_sizes": self.country_metrics[[
                "country",
                "total_samples",
            ]].sort_values("total_samples", ascending=False),
            "representation_ratios": (self.country_metrics["total_samples"] / total_samples).sort_values(
                ascending=False
            ),
        }
        return distribution

    def calculate_accuracy_differences(self):
        overall_acc = self.overall_metrics["accuracy"]
        df = self.country_metrics.copy()
        df["accuracy_difference"] = df["accuracy"] - overall_acc
        df["abs_accuracy_difference"] = df["accuracy_difference"].abs()
        return df[[
            "country",
            "accuracy",
            "accuracy_difference",
            "abs_accuracy_difference",
        ]].sort_values("abs_accuracy_difference", ascending=False)

    def calculate_disparate_impact_ratio(self):
        overall_ppv = self.overall_metrics["precision"]
        di = self.country_metrics.copy()
        di["disparate_impact_ratio"] = di["predicted_positive_rate"] / overall_ppv
        di["disparate_impact_flag"] = di["disparate_impact_ratio"].apply(
            lambda x: "Significant" if (x < 0.8 or x > 1.25) else "Acceptable"
        )
        return di[[
            "country",
            "predicted_positive_rate",
            "disparate_impact_ratio",
            "disparate_impact_flag",
        ]].sort_values("disparate_impact_ratio")

    def calculate_equal_opportunity_difference(self):
        overall_tpr = self.overall_metrics["recall"]
        eo = self.country_metrics.copy()
        eo["equal_opportunity_diff"] = eo["recall"] - overall_tpr
        eo["abs_eo_diff"] = eo["equal_opportunity_diff"].abs()
        eo["eo_flag"] = eo["abs_eo_diff"].apply(lambda x: "Significant" if x > 0.1 else "Acceptable")
        return eo[[
            "country",
            "recall",
            "equal_opportunity_diff",
            "abs_eo_diff",
            "eo_flag",
        ]].sort_values("abs_eo_diff", ascending=False)

    def calculate_ppv_bias(self):
        overall_ppv = self.overall_metrics["precision"]
        ppv = self.country_metrics.copy()
        ppv["ppv_bias"] = ppv["precision"] - overall_ppv
        ppv["abs_ppv_bias"] = ppv["ppv_bias"].abs()
        ppv["ppv_flag"] = ppv["abs_ppv_bias"].apply(lambda x: "Significant" if x > 0.1 else "Acceptable")
        return ppv[[
            "country",
            "precision",
            "ppv_bias",
            "abs_ppv_bias",
            "ppv_flag",
        ]].sort_values("abs_ppv_bias", ascending=False)

    def comprehensive_bias_analysis(self):
        self.bias_analysis = {
            "sample_distribution": self.analyze_sample_distribution(),
            "accuracy_differences": self.calculate_accuracy_differences(),
            "disparate_impact": self.calculate_disparate_impact_ratio(),
            "equal_opportunity": self.calculate_equal_opportunity_difference(),
            "ppv_bias": self.calculate_ppv_bias(),
        }
        return self.bias_analysis

    # -------------------------------------------------------------------------
    #  Plot helpers ------------------------------------------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def bucket_top_mid_other(df: pd.DataFrame, col: str, k: int, target_value: float = None) -> pd.DataFrame:
        """Return *top‑k*, *middle‑k* (closest to target), *bottom‑k* rows and a
        single aggregated **Other** row.

        * Top‑k  : largest *k* values.
        * Bottom : smallest *k* values.
        * Middle : rows whose value is closest to the *target_value*,
                excluding those already in Top/Bottom.
        * Other  : one row with country='Other' and the **mean** value of the
                remaining rows.
        
        Args:
            df: DataFrame to process
            col: Column name to bucket on
            k: Number of entries for each bucket
            target_value: Target value for middle selection. If None, uses mean.
        """
        n = len(df)
        if k * 3 >= n:
            # Not enough rows to slice; just return the whole DF untouched.
            return df.copy()

        # Compute slices
        top = df.nlargest(k, col).copy()
        bottom = df.nsmallest(k, col).copy()

        remaining = df.drop(top.index.union(bottom.index))
        if remaining.empty:
            # Edge‑case: top+bottom consumed all rows
            middle = pd.DataFrame(columns=df.columns)
            other_rows = pd.DataFrame(columns=df.columns)
        else:
            # Use target_value if provided, otherwise fall back to mean
            if target_value is not None:
                reference_val = target_value
            else:
                reference_val = remaining[col].mean()
            
            # Middle‑k closest to the reference value
            middle = remaining.iloc[(remaining[col] - reference_val).abs().argsort()[:k]].copy()
            rest = remaining.drop(middle.index)
            # Aggregated Other (mean of remaining rows)
            other_val = rest[col].mean() if not rest.empty else np.nan
            other_rows = pd.DataFrame({
                "country": ["Other"],
                col: [other_val],
            })

        # Tag buckets for potential downstream use
        top["bucket"] = "Top"
        middle["bucket"] = "Middle"
        bottom["bucket"] = "Bottom"
        other_rows["bucket"] = "Other"

        return pd.concat([top, middle, bottom, other_rows], ignore_index=True)        

    # -------------------------------------------------------------------------
    #  Paper‑ready plotting pipeline ------------------------------------------
    # -------------------------------------------------------------------------
    def create_paper_plots(self, plot_dir: str = "outputs/paper_figs", k: int = 20):
        """Generate publication‑quality static figures (600 dpi)."""
        if not self.bias_analysis:
            self.comprehensive_bias_analysis()

        os.makedirs(plot_dir, exist_ok=True)
        sns.set_palette("muted")

        # ---------- 1. Accuracy Δ (horizontal, trimmed) ----------------------
        # Target value is 0 (no difference from overall accuracy)
        adf = self.bucket_top_mid_other(
            self.bias_analysis["accuracy_differences"], 
            "accuracy_difference", 
            k, 
            target_value=0.0
        )
        adf = adf.sort_values("accuracy_difference")
        palette = ["#d62728" if v < 0 else "#2ca02c" for v in adf["accuracy_difference"]]

        fig_h = min(10, max(4, 0.25 * len(adf)))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        sns.barplot(data=adf, x="accuracy_difference", y="country", palette=palette, ax=ax)
        ax.axvline(0, color="k", linewidth=0.8)
        
        # Add dotted lines at 15th position from top and bottom
        n_entries = len(adf)
        if n_entries > 30:  # Only add lines if we have more than 30 entries
            ax.axhline(k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from top
            ax.axhline(n_entries - k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from bottom
        
        ax.set_xlabel("Δ Accuracy")
        ax.set_ylabel("")
        ax.set_title(f"Accuracy Δ vs Overall (Top, Middle, & Bottom {k} + Other)")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/accuracy_diff_paper.png", dpi=600)
        plt.close(fig)

        # ---------- 2. Sample distribution (log & Lorenz) -------------------
        sd = self.bias_analysis["sample_distribution"]["country_sample_sizes"]
        top_n = 30 if len(sd) > 30 else len(sd)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Bar (log‑scale)
        sns.barplot(x="country", y="total_samples", data=sd.head(top_n), ax=ax1, color="#5DADE2")
        ax1.set_yscale("log")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax1.set_xlabel("")
        ax1.set_ylabel("Samples (log)")
        ax1.set_title(f"Top {top_n} sample counts")

        # Lorenz curve
        share = sd["total_samples"].sort_values().cumsum() / sd["total_samples"].sum()
        share.index = np.arange(1, len(share) + 1)
        ax2.plot(share.index / len(share), share.values, drawstyle="steps-post")
        ax2.plot([0, 1], [0, 1], "--k", alpha=0.5)
        ax2.set_xlabel("Cumulative share of countries")
        ax2.set_ylabel("Cumulative share of samples")
        ax2.set_title("Lorenz curve (inequality)")

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/sample_distribution_paper.png", dpi=600)
        plt.close(fig)

        # ---------- 3. Disparate impact (scatter, trimmed) -------------------
        # Target value is 1.0 (no disparate impact)
        di = self.bucket_top_mid_other(
            self.bias_analysis["disparate_impact"], 
            "disparate_impact_ratio", 
            k, 
            target_value=1.0
        )
        di = di.sort_values("disparate_impact_ratio")
        fig_h = min(10, max(4, 0.22 * len(di)))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        colors = ["#2ca02c" if f == "Acceptable" else "#d62728" for f in di["disparate_impact_flag"]]
        ax.scatter(di["disparate_impact_ratio"], di["country"], c=colors, s=30)
        ax.axvline(1.0, color="k", linewidth=0.8)
        ax.axvline(0.8, color="k", linestyle="--", linewidth=0.8)
        ax.axvline(1.25, color="k", linestyle="--", linewidth=0.8)
        
        # Add dotted lines at 15th position from top and bottom
        n_entries = len(di)
        if n_entries > 30:  # Only add lines if we have more than 30 entries
            ax.axhline(k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from top
            ax.axhline(n_entries - k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from bottom
        
        ax.set_xlabel("Disparate Impact Ratio")
        ax.set_ylabel("")
        ax.set_title(f"Disparate Impact (Top, Middle, & Bottom {k} + Other)")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/disparate_impact_paper.png", dpi=600)
        plt.close(fig)

        # ---------- 4. Equal opportunity diff --------------------------------
        # Target value is 0.0 (no difference in equal opportunity)
        eo = self.bucket_top_mid_other(
            self.bias_analysis["equal_opportunity"], 
            "equal_opportunity_diff", 
            k, 
            target_value=0.0
        )
        eo = eo.sort_values("equal_opportunity_diff")
        fig, ax = plt.subplots(figsize=(8, fig_h))
        colors = ["#2ca02c" if f == "Acceptable" else "#d62728" for f in eo["eo_flag"]]
        ax.scatter(eo["equal_opportunity_diff"], eo["country"], c=colors, s=30)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.axvline(0.1, color="k", linestyle="--", linewidth=0.8)
        ax.axvline(-0.1, color="k", linestyle="--", linewidth=0.8)
        
        n_entries = len(di)
        if n_entries > 30:  # Only add lines if we have more than 30 entries
            ax.axhline(k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from top
            ax.axhline(n_entries - k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from bottom
        
        ax.set_xlabel("Equal Opportunity Diff")
        ax.set_ylabel("")
        ax.set_title(f"Equal Opportunity Diff (Top, Middle, & Bottom {k} + Other)")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/eo_diff_paper.png", dpi=600)
        plt.close(fig)

        # ---------- 5. PPV bias ----------------------------------------------
        # Target value is 0.0 (no PPV bias)
        ppv = self.bucket_top_mid_other(
            self.bias_analysis["ppv_bias"], 
            "ppv_bias", 
            k, 
            target_value=0.0
        )
        ppv = ppv.sort_values("ppv_bias")
        fig, ax = plt.subplots(figsize=(8, fig_h))
        colors = ["#2ca02c" if f == "Acceptable" else "#d62728" for f in ppv["ppv_flag"]]
        ax.scatter(ppv["ppv_bias"], ppv["country"], c=colors, s=30)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.axvline(0.1, color="k", linestyle="--", linewidth=0.8)
        ax.axvline(-0.1, color="k", linestyle="--", linewidth=0.8)

        n_entries = len(di)
        if n_entries > 30:  # Only add lines if we have more than 30 entries
            ax.axhline(k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from top
            ax.axhline(n_entries - k - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)  # 15th from bottom

        ax.set_xlabel("PPV Bias")
        ax.set_ylabel("")
        ax.set_title(f"PPV Bias (Top, Middle, & Bottom {k} + Other)")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/ppv_bias_paper.png", dpi=600)
        plt.close(fig)

        # ---------- 6. Accuracy histogram ------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(self.country_metrics["accuracy"], bins=20, edgecolor="black")
        ax.axvline(
            self.overall_metrics["accuracy"],
            color="red",
            linestyle="--",
            label=f"Overall Accuracy: {self.overall_metrics['accuracy']:.3f}",
        )
        ax.set_title("Distribution of Country Accuracies", fontsize=14)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/accuracy_hist_paper.png", dpi=600)
        plt.close(fig)

        # ---------- 7. Confusion matrix (overall) ----------------------------
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = np.array(self.overall_metrics["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred Neg", "Pred Pos"],
            yticklabels=["Actual Neg", "Actual Pos"],
            cbar_kws={"label": "Count"},
            ax=ax,
        )
        ax.set_title("Confusion Matrix", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/confusion_matrix_paper.png", dpi=600)
        plt.close(fig)

        print(f"Paper‑ready figures saved to → {plot_dir}")


# -----------------------------------------------------------------------------
#  Simple CLI entry‑point for quick testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate publication‑quality bias & performance figures."
    )
    parser.add_argument(
        "--json", required=True, help="Path to detailed_metrics.json (or similar)"
    )
    parser.add_argument(
        "--out", default="outputs/paper_figs", help="Output directory for figures"
    )
    parser.add_argument(
        "-k", type=int, default=20, help="Top & bottom k countries to show"
    )
    args = parser.parse_args()

    analyzer = ModelBiasAnalyzerPaper(data_path=args.json)
    analyzer.comprehensive_bias_analysis()
    analyzer.create_paper_plots(plot_dir=args.out, k=args.k)