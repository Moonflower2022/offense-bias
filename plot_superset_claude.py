import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

class ModelBiasAnalyzer:
    """
    Comprehensive model performance and bias analysis tool following AI Fairness 360 framework.
    Analyzes performance metrics and bias across different demographic groups (countries).
    """
    
    def __init__(self, data_path: str = None, data_dict: dict = None):
        """
        Initialize analyzer with model evaluation data.
        
        Args:
            data_path: Path to JSON file containing evaluation results
            data_dict: Dictionary containing evaluation results (alternative to file path)
        """
        if data_dict:
            self.data = data_dict
        elif data_path:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError("Either data_path or data_dict must be provided")
        
        self.overall_metrics = self._extract_overall_metrics()
        self.country_metrics = self._extract_country_metrics()
        self.bias_analysis = {}
        
    def _extract_overall_metrics(self) -> Dict:
        """Extract overall model performance metrics."""
        return {
            'total_samples': self.data['total_samples'],
            'accuracy': self.data['accuracy'],
            'precision': self.data['precision'],
            'recall': self.data['recall'],
            'f1_score': self.data['f1_score'],
            'confusion_matrix': self.data['confusion_matrix']
        }
    
    def _extract_country_metrics(self) -> pd.DataFrame:
        """Extract country-specific metrics into a DataFrame."""
        country_data = []
        for country, metrics in self.data['country_prediction_rates'].items():
            country_data.append({
                'country': country,
                'total_samples': metrics['total_valid_predictions'],
                'predicted_positive': metrics['predicted_positive_count'],
                'predicted_negative': metrics['predicted_negative_count'],
                'predicted_positive_rate': metrics['predicted_positive_rate'],
                'predicted_negative_rate': metrics['predicted_negative_rate'],
                'ground_truth_positive_rate': metrics['ground_truth_positive_rate'],
                'accuracy': metrics['accuracy'],
                'true_positive': metrics['true_positive_count'],
                'true_negative': metrics['true_negative_count'],
                'false_positive': metrics['false_positive_count'],
                'false_negative': metrics['false_negative_count']
            })
        
        df = pd.DataFrame(country_data)
        
        # Calculate additional metrics
        df['precision'] = df.apply(lambda row: 
            row['true_positive'] / (row['true_positive'] + row['false_positive']) 
            if (row['true_positive'] + row['false_positive']) > 0 else 0, axis=1)
        
        df['recall'] = df.apply(lambda row: 
            row['true_positive'] / (row['true_positive'] + row['false_negative']) 
            if (row['true_positive'] + row['false_negative']) > 0 else 0, axis=1)
        
        df['specificity'] = df.apply(lambda row: 
            row['true_negative'] / (row['true_negative'] + row['false_positive']) 
            if (row['true_negative'] + row['false_positive']) > 0 else 0, axis=1)
        
        df['false_positive_rate'] = df.apply(lambda row: 
            row['false_positive'] / (row['false_positive'] + row['true_negative']) 
            if (row['false_positive'] + row['true_negative']) > 0 else 0, axis=1)
        
        return df
    
    def analyze_sample_distribution(self) -> Dict:
        """Analyze sample size distribution across countries."""
        total_samples = self.country_metrics['total_samples'].sum()
        
        distribution_analysis = {
            'total_samples': total_samples,
            'country_sample_sizes': self.country_metrics[['country', 'total_samples']].sort_values('total_samples', ascending=False),
            'sample_size_stats': {
                'mean': self.country_metrics['total_samples'].mean(),
                'median': self.country_metrics['total_samples'].median(),
                'std': self.country_metrics['total_samples'].std(),
                'min': self.country_metrics['total_samples'].min(),
                'max': self.country_metrics['total_samples'].max()
            },
            'representation_ratios': (self.country_metrics['total_samples'] / total_samples).sort_values(ascending=False)
        }
        
        return distribution_analysis
    
    def calculate_accuracy_differences(self) -> pd.DataFrame:
        """Calculate accuracy differences between overall model accuracy and country-specific accuracies."""
        # Overall accuracy across all countries
        overall_accuracy = self.overall_metrics['accuracy']

        # Copy country-level metrics
        df = self.country_metrics.copy()

        # Compute the difference: overall accuracy minus country accuracy
        df['accuracy_difference'] = df['accuracy'] - overall_accuracy
        df['abs_accuracy_difference'] = df['accuracy_difference'].abs()

        # Return the relevant columns sorted by magnitude of the difference
        return df[['country', 'accuracy', 'accuracy_difference', 'abs_accuracy_difference']].sort_values('abs_accuracy_difference', ascending=False)
        
    def calculate_disparate_impact_ratio(self) -> pd.DataFrame:
        """Calculate disparate impact ratio for each country."""
        # Overall positive prediction rate
        overall_positive_rate = self.overall_metrics['precision']  # Using precision as overall positive prediction rate
        
        disparate_impact = self.country_metrics.copy()
        disparate_impact['disparate_impact_ratio'] = disparate_impact['predicted_positive_rate'] / overall_positive_rate
        
        # Flag countries with significant disparate impact (typically < 0.8 or > 1.25)
        disparate_impact['disparate_impact_flag'] = disparate_impact['disparate_impact_ratio'].apply(
            lambda x: 'Significant' if x < 0.8 or x > 1.25 else 'Acceptable'
        )
        
        return disparate_impact[['country', 'predicted_positive_rate', 'disparate_impact_ratio', 'disparate_impact_flag']].sort_values('disparate_impact_ratio')
    
    def calculate_equal_opportunity_difference(self) -> pd.DataFrame:
        """Calculate equal opportunity difference (difference in recall/TPR)."""
        overall_tpr = self.overall_metrics['recall']
        
        eo_analysis = self.country_metrics.copy()
        eo_analysis['equal_opportunity_diff'] = eo_analysis['recall'] - overall_tpr
        eo_analysis['abs_eo_diff'] = np.abs(eo_analysis['equal_opportunity_diff'])
        
        # Flag countries with significant equal opportunity differences (typically > 0.1)
        eo_analysis['eo_flag'] = eo_analysis['abs_eo_diff'].apply(
            lambda x: 'Significant' if x > 0.1 else 'Acceptable'
        )
        
        return eo_analysis[['country', 'recall', 'equal_opportunity_diff', 'abs_eo_diff', 'eo_flag']].sort_values('abs_eo_diff', ascending=False)
    
    def calculate_ppv_bias(self) -> pd.DataFrame:
        """Calculate Positive Predictive Value (PPV) bias."""
        overall_ppv = self.overall_metrics['precision']
        
        ppv_analysis = self.country_metrics.copy()
        ppv_analysis['ppv_bias'] = ppv_analysis['precision'] - overall_ppv
        ppv_analysis['abs_ppv_bias'] = np.abs(ppv_analysis['ppv_bias'])
        
        # Flag countries with significant PPV bias
        ppv_analysis['ppv_flag'] = ppv_analysis['abs_ppv_bias'].apply(
            lambda x: 'Significant' if x > 0.1 else 'Acceptable'
        )
        
        return ppv_analysis[['country', 'precision', 'ppv_bias', 'abs_ppv_bias', 'ppv_flag']].sort_values('abs_ppv_bias', ascending=False)
    
    def comprehensive_bias_analysis(self) -> Dict:
        """Perform comprehensive bias analysis."""
        analysis = {
            'sample_distribution': self.analyze_sample_distribution(),
            'accuracy_differences': self.calculate_accuracy_differences(),
            'disparate_impact': self.calculate_disparate_impact_ratio(),
            'equal_opportunity': self.calculate_equal_opportunity_difference(),
            'ppv_bias': self.calculate_ppv_bias()
        }
        
        self.bias_analysis = analysis
        return analysis
    
    def generate_summary_report(self) -> Dict:
        """Generate summary report of key findings."""
        if not self.bias_analysis:
            self.comprehensive_bias_analysis()
        
        # Countries with most significant biases
        di_significant = self.bias_analysis['disparate_impact'][
            self.bias_analysis['disparate_impact']['disparate_impact_flag'] == 'Significant'
        ]
        
        eo_significant = self.bias_analysis['equal_opportunity'][
            self.bias_analysis['equal_opportunity']['eo_flag'] == 'Significant'
        ]
        
        ppv_significant = self.bias_analysis['ppv_bias'][
            self.bias_analysis['ppv_bias']['ppv_flag'] == 'Significant'
        ]
        
        # Top countries by sample size
        top_countries = self.bias_analysis['sample_distribution']['country_sample_sizes'].head(5)
        
        summary = {
            'overall_performance': {
                'accuracy': self.overall_metrics['accuracy'],
                'precision': self.overall_metrics['precision'],
                'recall': self.overall_metrics['recall'],
                'f1_score': self.overall_metrics['f1_score']
            },
            'bias_flags': {
                'disparate_impact_issues': len(di_significant),
                'equal_opportunity_issues': len(eo_significant),
                'ppv_bias_issues': len(ppv_significant),
                'countries_with_di_issues': di_significant['country'].tolist() if len(di_significant) > 0 else [],
                'countries_with_eo_issues': eo_significant['country'].tolist() if len(eo_significant) > 0 else [],
                'countries_with_ppv_issues': ppv_significant['country'].tolist() if len(ppv_significant) > 0 else []
            },
            'data_representation': {
                'total_countries': len(self.country_metrics),
                'top_5_countries_by_sample_size': top_countries[['country', 'total_samples']].to_dict('records'),
                'sample_size_inequality': {
                    'max_samples': self.country_metrics['total_samples'].max(),
                    'min_samples': self.country_metrics['total_samples'].min(),
                    'ratio': self.country_metrics['total_samples'].max() / self.country_metrics['total_samples'].min()
                }
            }
        }
        
        return summary
    
    def create_individual_plots(self, plot_dir: str = 'outputs/figures/'):
        """Create individual plots and save them separately, with fixed country labels."""
        if not self.bias_analysis:
            self.comprehensive_bias_analysis()
        
        os.makedirs(plot_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Helper to annotate points
        def annotate_all(ax, xs, ys, labels, x_offset=0, y_offset=4):
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(
                    lab,
                    (x + x_offset, y + y_offset),
                    textcoords='offset points',
                    ha='center', fontsize=7, alpha=0.8
                )
        
        # 1. Sample Distribution (All Countries)
        sd = self.bias_analysis['sample_distribution']['country_sample_sizes']
        n = len(sd)
        # Make figure wider if many countries so labels stay readable
        fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 6))
        xs = np.arange(n)
        ax.bar(xs, sd['total_samples'], alpha=0.8, edgecolor='black')
        ax.set_xticks(xs)
        ax.set_xticklabels(sd['country'], rotation=90, ha='right', fontsize=8)
        ax.set_title('Sample Distribution by Country', fontsize=16, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/sample_distribution_all.png', dpi=300)
        plt.close(fig)
        
        # 2. Accuracy Differences (All Countries)
        br = self.bias_analysis['accuracy_differences']  # no .head(10), include everyone
        n = len(br)
        # Make figure wider if many countries so labels stay readable
        fig, ax = plt.subplots(figsize=(max(12, n * 0.4), 6))
        xs = np.arange(n)
        colors = ['red' if val > 0 else 'blue' for val in br['accuracy_difference']]
        ax.bar(xs, br['accuracy_difference'], color=colors, alpha=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels(br['country'], rotation=90, ha='right', fontsize=8)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Accuracy Differences by Country', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy Differences', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/accuracy_differences.png', dpi=300)
        plt.close(fig)
        
        # 3. Disparate Impact Ratios
        di = self.bias_analysis['disparate_impact']
        fig, ax = plt.subplots(figsize=(14, 6))
        xs = np.arange(len(di))
        ys = di['disparate_impact_ratio']
        colors = ['red' if f=='Significant' else 'green' for f in di['disparate_impact_flag']]
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=60)
        ax.set_xticks(xs)
        ax.set_xticklabels(di['country'], rotation=90, ha='right')
        annotate_all(ax, xs, ys, di['country'])
        ax.axhline(0.8, linestyle='--', alpha=0.5, label='Lower Threshold (0.8)')
        ax.axhline(1.25, linestyle='--', alpha=0.5, label='Upper Threshold (1.25)')
        ax.axhline(1.0, linestyle='-', alpha=0.3, label='Parity Line')
        ax.set_title('Disparate Impact Ratios by Country', fontsize=16, fontweight='bold')
        ax.set_ylabel('Disparate Impact Ratio', fontsize=14)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/disparate_impact_ratios.png', dpi=300)
        plt.close(fig)
        
        # 4. Equal Opportunity Differences
        eo = self.bias_analysis['equal_opportunity']
        fig, ax = plt.subplots(figsize=(14, 6))
        xs = np.arange(len(eo))
        ys = eo['equal_opportunity_diff']
        colors = ['red' if f=='Significant' else 'green' for f in eo['eo_flag']]
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=60)
        ax.set_xticks(xs)
        ax.set_xticklabels(eo['country'], rotation=90, ha='right')
        annotate_all(ax, xs, ys, eo['country'])
        ax.axhline(0, linestyle='-', alpha=0.3, label='Parity Line')
        ax.axhline(0.1, linestyle='--', alpha=0.5, label='Upper Threshold (0.1)')
        ax.axhline(-0.1, linestyle='--', alpha=0.5, label='Lower Threshold (-0.1)')
        ax.set_title('Equal Opportunity Differences by Country', fontsize=16, fontweight='bold')
        ax.set_ylabel('Equal Opportunity Difference', fontsize=14)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/equal_opportunity_differences.png', dpi=300)
        plt.close(fig)
        
        # 5. PPV Bias
        ppv = self.bias_analysis['ppv_bias']
        fig, ax = plt.subplots(figsize=(14, 6))
        xs = np.arange(len(ppv))
        ys = ppv['ppv_bias']
        colors = ['red' if f=='Significant' else 'green' for f in ppv['ppv_flag']]
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=60)
        ax.set_xticks(xs)
        ax.set_xticklabels(ppv['country'], rotation=90)
        annotate_all(ax, xs, ys, ppv['country'])
        ax.axhline(0, linestyle='-', alpha=0.3, label='Parity Line')
        ax.axhline(0.1, linestyle='--', alpha=0.5, label='Upper Threshold (0.1)')
        ax.axhline(-0.1, linestyle='--', alpha=0.5, label='Lower Threshold (-0.1)')
        ax.set_title('Positive Predictive Value Bias by Country', fontsize=16, fontweight='bold')
        ax.set_ylabel('PPV Bias', fontsize=14)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/ppv_bias.png', dpi=300)
        plt.close(fig)
        
        # 6. Accuracy Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.country_metrics['accuracy'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(self.overall_metrics['accuracy'], color='red', linestyle='--',
                   label=f'Overall Accuracy: {self.overall_metrics["accuracy"]:.3f}')
        ax.set_title('Distribution of Country-Level Accuracies', fontsize=16, fontweight='bold')
        ax.set_xlabel('Accuracy', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/accuracy_distribution.png', dpi=300)
        plt.close(fig)
        
        # 7. Performance vs Sample Size
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(self.country_metrics['total_samples'], self.country_metrics['accuracy'],
                   alpha=0.7, s=60)
        ax.set_xscale('log')
        ax.set_title('Accuracy vs Sample Size by Country', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sample Size (log scale)', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/accuracy_vs_sample_size.png', dpi=300)
        plt.close(fig)
        
        # 8. Confusion Matrix Heatmap (Overall)
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = np.array(self.overall_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Neg', 'Pred Pos'],
                    yticklabels=['Actual Neg', 'Actual Pos'],
                    cbar_kws={'label': 'Count'}, ax=ax)
        ax.set_title('Overall Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('Actual Label', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/confusion_matrix.png', dpi=300)
        plt.close(fig)
        
        # 9. Bias Summary Overview
        summary = self.generate_summary_report()
        counts = [
            summary['bias_flags']['disparate_impact_issues'],
            summary['bias_flags']['equal_opportunity_issues'],
            summary['bias_flags']['ppv_bias_issues']
        ]
        total_countries = summary['data_representation']['total_countries']
        types = ['Disparate Impact', 'Equal Opportunity', 'PPV Bias']
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(types, counts, alpha=0.7, edgecolor='black', color=['red','orange','yellow'])
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, c+0.1, str(c),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_title(f'Number of Countries (out of {total_countries}) with Significant Bias Issues', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Countries', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/bias_summary.png', dpi=300)
        plt.close(fig)
        
        print(f"All plots saved to {plot_dir}:")
        print(" 1. sample_distribution.png")
        print(" 2. accuracy_differences.png")
        print(" 3. disparate_impact_ratios.png")
        print(" 4. equal_opportunity_differences.png")
        print(" 5. ppv_bias.png")
        print(" 6. accuracy_distribution.png")
        print(" 7. accuracy_vs_sample_size.png")
        print(" 8. confusion_matrix.png")
        print(" 9. bias_summary.png")
    
    def print_detailed_report(self):
        """Print a detailed analysis report."""
        if not self.bias_analysis:
            self.comprehensive_bias_analysis()
        
        summary = self.generate_summary_report()
        
        print("="*80)
        print("COMPREHENSIVE MODEL BIAS AND PERFORMANCE ANALYSIS")
        print("="*80)
        
        print("\n1. OVERALL PERFORMANCE METRICS")
        print("-" * 40)
        for metric, value in summary['overall_performance'].items():
            print(f"{metric.upper()}: {value:.4f}")
        
        print("\n2. DATASET REPRESENTATION ANALYSIS")
        print("-" * 40)
        print(f"Total Countries: {summary['data_representation']['total_countries']}")
        print(f"Total Samples: {self.bias_analysis['sample_distribution']['total_samples']}")
        print(f"Sample Size Range: {summary['data_representation']['sample_size_inequality']['min_samples']} - {summary['data_representation']['sample_size_inequality']['max_samples']}")
        print(f"Sample Size Ratio (Max/Min): {summary['data_representation']['sample_size_inequality']['ratio']:.2f}")
        
        print("\nTop 5 Countries by Sample Size:")
        for country_data in summary['data_representation']['top_5_countries_by_sample_size']:
            print(f"  {country_data['country']}: {country_data['total_samples']} samples")
        
        print("\n3. BIAS ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Countries with Disparate Impact Issues: {summary['bias_flags']['disparate_impact_issues']}")
        if summary['bias_flags']['countries_with_di_issues']:
            print(f"  Affected Countries: {', '.join(summary['bias_flags']['countries_with_di_issues'])}")
        
        print(f"Countries with Equal Opportunity Issues: {summary['bias_flags']['equal_opportunity_issues']}")
        if summary['bias_flags']['countries_with_eo_issues']:
            print(f"  Affected Countries: {', '.join(summary['bias_flags']['countries_with_eo_issues'])}")
        
        print(f"Countries with PPV Bias Issues: {summary['bias_flags']['ppv_bias_issues']}")
        if summary['bias_flags']['countries_with_ppv_issues']:
            print(f"  Affected Countries: {', '.join(summary['bias_flags']['countries_with_ppv_issues'])}")
        
        print("\n4. DETAILED BIAS METRICS")
        print("-" * 40)
        
        print("\nTop 10 Countries by Base Rate Gap:")
        accuracy_difference_top = self.bias_analysis['accuracy_differences'].head(10)
        for _, row in accuracy_difference_top.iterrows():
            print(f"  {row['country']}: {row['accuracy_difference']:+.4f}")
        
        print("\nCountries with Significant Disparate Impact:")
        di_issues = self.bias_analysis['disparate_impact'][
            self.bias_analysis['disparate_impact']['disparate_impact_flag'] == 'Significant'
        ]
        for _, row in di_issues.iterrows():
            print(f"  {row['country']}: {row['disparate_impact_ratio']:.4f}")
        
        print("\nCountries with Significant Equal Opportunity Differences:")
        eo_issues = self.bias_analysis['equal_opportunity'][
            self.bias_analysis['equal_opportunity']['eo_flag'] == 'Significant'
        ]
        for _, row in eo_issues.iterrows():
            print(f"  {row['country']}: {row['equal_opportunity_diff']:+.4f}")
        
        print("\n" + "="*80)


# Example usage and demonstration
def main():
    """Main function to demonstrate the analysis."""
    
    # Initialize analyzer
    print("Initializing Model Bias Analyzer...")
    
    # For demonstration, we'll use the data from the document
    # In practice, you would load your JSON file like this:
    directory = "outputs/0612_16:29"

    analyzer = ModelBiasAnalyzer(data_path=f'{directory}/detailed_metrics.json')

    # Run comprehensive analysis
    analyzer.comprehensive_bias_analysis()
    
    # Print detailed report
    analyzer.print_detailed_report()
    
    # Create individual plots (new method)
    analyzer.create_individual_plots(plot_dir=f'{directory}/figures')
    
    # Optionally, still create the comprehensive plot (original method)
    # analyzer.create_visualizations(save_plots=True, plot_dir='outputs/figures/')


if __name__ == "__main__":
    main()