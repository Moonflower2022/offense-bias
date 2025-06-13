import pandas as pd
import anthropic
import time
import json
from typing import List, Dict, Optional
import os
from tqdm import tqdm
import dotenv
import pickle
import hashlib
import numpy as np
from datetime import datetime
import argparse

class ClaudeHateClassifier:
    def __init__(self, api_key: str = None, cache_dir: str = "cache"):
        """
        Initialize the Claude hate speech classifier
        
        Args:
            api_key: Anthropic API key. If None, will look for ANTHROPIC_API_KEY env var
            cache_dir: Directory to store cache files
        """
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20240620"
        
        # Setup caching
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "claude_predictions_cache.pkl")
        self.sample_cache_file = os.path.join(cache_dir, "sample_selection_cache.pkl")
        
        # Load existing caches
        self.prediction_cache = self._load_prediction_cache()
        self.sample_cache = self._load_sample_cache()
        
        # Setup output directory for this run
        self.output_dir = self._create_output_directory()
        
    def _create_output_directory(self) -> str:
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%m%d_%H:%M")
        output_dir = os.path.join("outputs", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        return output_dir
        
    def _load_prediction_cache(self) -> Dict:
        """Load prediction cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load prediction cache: {e}")
        return {}
    
    def _save_prediction_cache(self):
        """Save prediction cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.prediction_cache, f)
        except Exception as e:
            print(f"Warning: Could not save prediction cache: {e}")
    
    def _load_sample_cache(self) -> Dict:
        """Load sample selection cache from disk"""
        if os.path.exists(self.sample_cache_file):
            try:
                with open(self.sample_cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load sample cache: {e}")
        return {}
    
    def _save_sample_cache(self):
        """Save sample selection cache to disk"""
        try:
            with open(self.sample_cache_file, 'wb') as f:
                pickle.dump(self.sample_cache, f)
        except Exception as e:
            print(f"Warning: Could not save sample cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for caching purposes"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_sample_key(self, dataset_path: str, seed: int, sample_size: int, equal_sampling: bool, allowed_countries: List[str]) -> str:
        """Generate a key for sample caching"""
        countries_str = "_".join(sorted(allowed_countries))
        return f"{dataset_path}_{seed}_{sample_size}_equal_{equal_sampling}_countries_{countries_str}"
        
    def create_classification_prompt(self, text: str) -> str:
        """Create a prompt for hate speech classification"""
        prompt = f"""You are a hate speech detection system. Your task is to classify the following text as either hateful (1) or not hateful (0). Text to classify: "{text}" Respond with only a single number: 1 for hateful, 0 for not hateful. Do not include any explanation or additional text."""
        
        return prompt
    
    def classify_single_text(self, text: str, max_retries: int = 3) -> int:
        """
        Classify a single text using Claude API with caching
        
        Args:
            text: Text to classify
            max_retries: Maximum number of retry attempts
            
        Returns:
            int: 1 for hateful, 0 for not hateful, -1 for error
        """
        # Check cache first
        text_hash = self._get_text_hash(text)
        if text_hash in self.prediction_cache:
            return self.prediction_cache[text_hash]
        
        prompt = self.create_classification_prompt(text)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2,
                    temperature=0,  # Low temperature for consistent results
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result = response.content[0].text.strip()
                
                # Parse the result
                if result == "1":
                    prediction = 1
                elif result == "0":
                    prediction = 0
                else:
                    print(f"Unexpected response: {result}. Retrying...")
                    continue
                
                # Cache the result
                self.prediction_cache[text_hash] = prediction
                self._save_prediction_cache()
                return prediction
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Failed to classify text after {max_retries} attempts")
                    # Cache the error result
                    self.prediction_cache[text_hash] = -1
                    self._save_prediction_cache()
                    return -1
        
        return -1
    
    def load_and_sample_dataset(self, dataset_path: str, sample_size: int = 400, 
                               random_seed: int = 42, equal_sampling: bool = False,
                               allowed_countries: List[str] = None) -> pd.DataFrame:
        """
        Load dataset, filter for entries with country locations, and sample
        
        Args:
            dataset_path: Path to the CSV file
            sample_size: Number of samples to select
            random_seed: Random seed for reproducibility
            equal_sampling: If True, sample equally from each country
            allowed_countries: List of allowed countries (case-insensitive)
            
        Returns:
            Sampled DataFrame
        """
        # Default allowed countries (excluding unknown/NaN)
        if allowed_countries is None:
            allowed_countries = ['india', 'united kingdom', 'united states', 'nigeria']
        
        # Normalize country names to lowercase
        allowed_countries = [c.lower() for c in allowed_countries]
        
        sample_key = self._get_sample_key(dataset_path, random_seed, sample_size, equal_sampling, allowed_countries)
        
        # Check if we have this sample cached
        if sample_key in self.sample_cache:
            print(f"Loading cached sample...")
            indices = self.sample_cache[sample_key]
            df = pd.read_csv(dataset_path)
            sampled_df = df.iloc[indices].copy()
            print(f"Loaded {len(sampled_df)} cached samples")
            return sampled_df
        
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {len(df)} rows")
        
        # Filter for allowed countries (excluding unknown/NaN by default)
        country_mask = (
            df['post_author_country_location'].notna() & 
            (df['post_author_country_location'] != '') & 
            (df['post_author_country_location'].str.lower() != 'unknown') &
            (df['post_author_country_location'].str.lower().isin(allowed_countries))
        )

        df_with_country = df[country_mask]
        print(f"After filtering: {len(df_with_country)} entries")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        if equal_sampling:
            # Equal sampling from each country
            samples_per_country = sample_size // len(allowed_countries)
            remaining_samples = sample_size % len(allowed_countries)
            
            sampled_dfs = []
            
            for i, country in enumerate(allowed_countries):
                country_data = df_with_country[
                    df_with_country['post_author_country_location'].str.lower() == country
                ]
                
                # Calculate samples for this country
                country_sample_size = samples_per_country
                if i < remaining_samples:  # Distribute remaining samples to first few countries
                    country_sample_size += 1
                
                if len(country_data) < country_sample_size:
                    country_sample_size = len(country_data)
                
                if country_sample_size > 0:
                    country_sample = country_data.sample(n=country_sample_size, random_state=random_seed)
                    sampled_dfs.append(country_sample)
            
            # Combine all country samples
            if sampled_dfs:
                sampled_df = pd.concat(sampled_dfs, ignore_index=True)
                # Shuffle the combined samples
                sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            else:
                sampled_df = pd.DataFrame()
            
        else:
            # Random sampling (original behavior)
            if len(df_with_country) < sample_size:
                print(f"Warning: Only {len(df_with_country)} entries available, using all")
                sampled_df = df_with_country.copy()
            else:
                sampled_indices = np.random.choice(df_with_country.index, size=sample_size, replace=False)
                sampled_df = df.loc[sampled_indices].copy()
        
        # Cache the sample selection (store indices relative to full dataset)
        if len(sampled_df) > 0:
            self.sample_cache[sample_key] = sampled_df.index.tolist()
            self._save_sample_cache()
        
        print(f"Selected {len(sampled_df)} samples")
        
        # Show brief country and label distribution
        if len(sampled_df) > 0:
            country_counts = sampled_df['post_author_country_location'].value_counts()
            print(f"Country distribution: {dict(country_counts)}")
            
            if 'labels' in sampled_df.columns:
                label_counts = sampled_df['labels'].value_counts()
                print(f"Label distribution: {dict(label_counts)}")
        
        return sampled_df
    
    def process_dataset(self, df: pd.DataFrame, output_file: str = None, 
                       save_interval: int = 50) -> pd.DataFrame:
        """
        Process the dataset and save comprehensive results with caching
        
        Args:
            df: DataFrame with 'text' column
            output_file: Path to save results (optional, will be placed in output_dir)
            save_interval: Save progress every N predictions
            
        Returns:
            DataFrame with predictions and metadata added
        """
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Add new columns for comprehensive analysis
        result_df['claude_predictions'] = -1
        result_df['prediction_timestamp'] = pd.Timestamp.now()
        result_df['text_length'] = result_df['text'].str.len()
        result_df['text_hash'] = result_df['text'].apply(self._get_text_hash)
        
        # Set output file path within timestamped directory
        if output_file is None:
            output_file = "claude_predictions_comprehensive.csv"
        output_path = os.path.join(self.output_dir, output_file)
        
        # Calculate country-level statistics for bias analysis
        country_stats = self._calculate_country_statistics(result_df)
        
        # Save country statistics to separate file
        country_stats_file = os.path.join(self.output_dir, output_file.replace('.csv', '_country_stats.csv'))
        country_stats_df = pd.DataFrame.from_dict(country_stats, orient='index')
        country_stats_df.to_csv(country_stats_file)
        
        # Get texts to classify
        texts = result_df['text'].tolist()
        
        print(f"Starting classification of {len(texts)} texts...")
        
        cache_hits = 0
        api_calls = 0
        
        # Process texts
        for i, text in enumerate(tqdm(texts, desc="Classifying texts")):
            text_hash = self._get_text_hash(str(text))
            
            if text_hash in self.prediction_cache:
                prediction = self.prediction_cache[text_hash]
                cache_hits += 1
            else:
                prediction = self.classify_single_text(str(text))
                api_calls += 1
                # Rate limiting only for actual API calls
                time.sleep(0.1)
            
            result_df.iloc[i, result_df.columns.get_loc('claude_predictions')] = prediction
            
            # Save progress periodically
            if (i + 1) % save_interval == 0:
                result_df.to_csv(output_path, index=False)
        
        print(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
        print(f"API calls: {api_calls}")
        
        # Add final metadata
        result_df['processing_completed'] = pd.Timestamp.now()
        result_df['cache_hit'] = result_df['text_hash'].apply(lambda x: x in self.prediction_cache)
        
        # Final save with comprehensive data
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Save summary statistics
        summary_file = os.path.join(self.output_dir, output_file.replace('.csv', '_summary.json'))
        summary_stats = self._generate_summary_statistics(result_df)
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        return result_df
    
    def _calculate_country_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate country-level statistics for bias analysis"""
        country_stats = {}
        
        for country in df['post_author_country_location'].unique():
            country_data = df[df['post_author_country_location'] == country]
            
            stats = {
                'sample_size': len(country_data),
                'sample_proportion': len(country_data) / len(df),
            }
            
            # If ground truth labels are available
            if 'labels' in df.columns:
                stats.update({
                    'ground_truth_positive_rate': country_data['labels'].mean(),
                    'ground_truth_positive_count': int(country_data['labels'].sum()),
                    'ground_truth_negative_count': int(len(country_data) - country_data['labels'].sum()),
                })
            
            country_stats[country] = stats
        
        return country_stats
    
    def _calculate_country_prediction_rates(self, df: pd.DataFrame) -> Dict:
        """Calculate country-specific prediction rates"""
        country_predictions = {}
        
        # Filter out error predictions
        valid_df = df[df['claude_predictions'] != -1]
        
        for country in df['post_author_country_location'].unique():
            country_data = valid_df[valid_df['post_author_country_location'] == country]
            
            if len(country_data) > 0:
                prediction_stats = {
                    'total_valid_predictions': len(country_data),
                    'predicted_positive_count': int((country_data['claude_predictions'] == 1).sum()),
                    'predicted_negative_count': int((country_data['claude_predictions'] == 0).sum()),
                    'predicted_positive_rate': (country_data['claude_predictions'] == 1).mean(),
                    'predicted_negative_rate': (country_data['claude_predictions'] == 0).mean(),
                }
                
                # Add ground truth comparison if available
                if 'labels' in country_data.columns:
                    prediction_stats.update({
                        'ground_truth_positive_rate': country_data['labels'].mean(),
                        'accuracy': (country_data['labels'] == country_data['claude_predictions']).mean(),
                        'true_positive_count': int(((country_data['labels'] == 1) & (country_data['claude_predictions'] == 1)).sum()),
                        'true_negative_count': int(((country_data['labels'] == 0) & (country_data['claude_predictions'] == 0)).sum()),
                        'false_positive_count': int(((country_data['labels'] == 0) & (country_data['claude_predictions'] == 1)).sum()),
                        'false_negative_count': int(((country_data['labels'] == 1) & (country_data['claude_predictions'] == 0)).sum()),
                    })
                
                country_predictions[country] = prediction_stats
            else:
                country_predictions[country] = {
                    'total_valid_predictions': 0,
                    'predicted_positive_count': 0,
                    'predicted_negative_count': 0,
                    'predicted_positive_rate': 0.0,
                    'predicted_negative_rate': 0.0,
                }
        
        return country_predictions
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics"""
        summary = {
            'total_samples': len(df),
            'processing_timestamp': pd.Timestamp.now(),
            'countries_analyzed': df['post_author_country_location'].nunique(),
            'country_distribution': df['post_author_country_location'].value_counts().to_dict(),
            'prediction_distribution': df['claude_predictions'].value_counts().to_dict(),
            'error_count': int((df['claude_predictions'] == -1).sum()),
            'cache_hit_rate': df['cache_hit'].mean() if 'cache_hit' in df.columns else None,
        }
        
        # Add country-specific prediction rates
        country_prediction_rates = self._calculate_country_prediction_rates(df)
        summary['country_prediction_rates'] = country_prediction_rates
        
        # Add ground truth comparison if available
        if 'labels' in df.columns:
            valid_predictions = df[df['claude_predictions'] != -1]
            if len(valid_predictions) > 0:
                summary.update({
                    'ground_truth_distribution': df['labels'].value_counts().to_dict(),
                    'valid_predictions': len(valid_predictions),
                    'overall_accuracy': (valid_predictions['labels'] == valid_predictions['claude_predictions']).mean(),
                })
        
        return summary
    
    def evaluate_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate Claude predictions against ground truth labels with country-specific metrics
        
        Args:
            df: DataFrame with 'labels' and 'claude_predictions' columns
            
        Returns:
            Dictionary with evaluation metrics including country-specific rates
        """
        # Filter out error predictions (-1)
        valid_mask = df['claude_predictions'] != -1
        valid_df = df[valid_mask]
        
        if len(valid_df) == 0:
            return {"error": "No valid predictions found"}
        
        y_true = valid_df['labels']
        y_pred = valid_df['claude_predictions']
        
        # Calculate overall metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        metrics = {
            'total_samples': len(df),
            'valid_predictions': len(valid_df),
            'error_predictions': len(df) - len(valid_df),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add country-specific prediction rates
        country_prediction_rates = self._calculate_country_prediction_rates(df)
        metrics['country_prediction_rates'] = country_prediction_rates
        
        return metrics

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Claude Hate Speech Classifier with Country Sampling Options')
    
    parser.add_argument('--dataset', type=str, default='datasets/superset.csv',
                        help='Path to the dataset CSV file (default: datasets/superset.csv)')
    
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Total number of samples to select (default: 200)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--equal-sampling', action='store_true',
                        help='Use equal sampling from each country (default: random sampling)')
    
    parser.add_argument('--output-file', type=str, default='claude_predictions_comprehensive.csv',
                        help='Output file name (default: claude_predictions_comprehensive.csv)')
    
    parser.add_argument('--allowed-countries', type=str, nargs='+', 
                        default=['india', 'united kingdom', 'united states', 'nigeria'],
                        help='List of allowed countries (default: india "united kingdom" "united states" nigeria)')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize classifier with caching
    classifier = ClaudeHateClassifier()
    
    # Print configuration
    print("Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Equal sampling: {args.equal_sampling}")
    print(f"  Allowed countries: {args.allowed_countries}")
    print(f"  Output file: {args.output_file}")
    
    # Load and sample dataset
    df = classifier.load_and_sample_dataset(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        random_seed=args.seed,
        equal_sampling=args.equal_sampling,
        allowed_countries=args.allowed_countries
    )
    
    if len(df) == 0:
        print("No samples selected. Exiting.")
        return
    
    # Process dataset with comprehensive output
    print(f"\nStarting classification of {len(df)} samples...")
    
    result_df = classifier.process_dataset(
        df, 
        output_file=args.output_file
    )
    
    # Comprehensive evaluation if ground truth labels are available
    if 'labels' in result_df.columns:
        print("\nEvaluating predictions...")
        metrics = classifier.evaluate_predictions(result_df)
        
        print("Evaluation Results:")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.3f}")
        print(f"  Recall: {metrics.get('recall', 'N/A'):.3f}")
        print(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.3f}")
        
        # Display country-specific prediction rates (brief)
        if 'country_prediction_rates' in metrics:
            print("\nCountry Prediction Rates:")
            for country, stats in metrics['country_prediction_rates'].items():
                if stats['total_valid_predictions'] > 0:
                    print(f"  {country}: {stats['predicted_positive_rate']:.3f} positive rate "
                          f"({stats['total_valid_predictions']} samples)")
        
        # Save detailed metrics to file
        metrics_file = os.path.join(classifier.output_dir, "detailed_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    print(f"\nProcessing complete! Files saved to: {classifier.output_dir}")

if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    main()