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
    
    def _get_sample_key(self, dataset_path: str, seed: int, sample_size: int) -> str:
        """Generate a key for sample caching"""
        return f"{dataset_path}_{seed}_{sample_size}"
        
    def create_classification_prompt(self, text: str) -> str:
        """Create a prompt for hate speech classification"""
        prompt = f"""You are a hate speech detection system. Your task is to classify the following text as either hateful (1) or not hateful (0).

Hateful content includes:
- Content that attacks, threatens, or incites violence against individuals or groups based on protected characteristics (race, ethnicity, religion, gender, sexual orientation, disability, etc.)
- Content that promotes hatred or discrimination
- Content that dehumanizes individuals or groups
- Slurs and offensive language targeting protected groups

Non-hateful content includes:
- General complaints or criticism
- Profanity not directed at protected groups
- Political opinions (unless promoting hatred)
- Heated but non-discriminatory arguments

Text to classify: "{text}"

Respond with only a single number: 1 for hateful, 0 for not hateful. Do not include any explanation or additional text."""
        
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
                               random_seed: int = 42) -> pd.DataFrame:
        """
        Load dataset, filter for entries with country locations, and sample randomly
        
        Args:
            dataset_path: Path to the CSV file
            sample_size: Number of samples to select
            random_seed: Random seed for reproducibility
            
        Returns:
            Sampled DataFrame
        """
        sample_key = self._get_sample_key(dataset_path, random_seed, sample_size)
        
        # Check if we have this sample cached
        if sample_key in self.sample_cache:
            print(f"Loading cached sample selection...")
            indices = self.sample_cache[sample_key]
            df = pd.read_csv(dataset_path)
            sampled_df = df.iloc[indices].copy()
            print(f"Loaded cached sample of {len(sampled_df)} rows")
            return sampled_df
        
        print(f"Loading full dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} total rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Filter for entries with country locations (excluding "unknown")
        print("\nFiltering for entries with known country locations...")
        print(f"Unique country values: {df['post_author_country_location'].value_counts().head(10)}")
        
        # Filter out null, empty, and "unknown" values
        country_mask = (
            df['post_author_country_location'].notna() & 
            (df['post_author_country_location'] != '') & 
            (df['post_author_country_location'].str.lower() != 'unknown')
        )
        df_with_country = df[country_mask]
        print(f"Found {len(df_with_country)} entries with known country locations")
        
        if len(df_with_country) < sample_size:
            print(f"Warning: Only {len(df_with_country)} entries with country locations available, using all of them")
            sampled_df = df_with_country.copy()
        else:
            # Random sampling with seed
            print(f"Randomly sampling {sample_size} entries (seed={random_seed})...")
            np.random.seed(random_seed)
            sampled_indices = np.random.choice(df_with_country.index, size=sample_size, replace=False)
            sampled_df = df.loc[sampled_indices].copy()
        
        # Cache the sample selection (store indices relative to full dataset)
        self.sample_cache[sample_key] = sampled_df.index.tolist()
        self._save_sample_cache()
        
        print(f"Selected {len(sampled_df)} samples")
        
        # Show country distribution
        if len(sampled_df) > 0:
            print(f"\nCountry distribution in sample:")
            country_counts = sampled_df['post_author_country_location'].value_counts()
            print(country_counts.head(10))
            
            # Show label distribution if available
            if 'labels' in sampled_df.columns:
                print(f"\nLabel distribution in sample:")
                print(sampled_df['labels'].value_counts())
        
        return sampled_df
    
    def classify_batch(self, texts: List[str], batch_delay: float = 1.0) -> List[int]:
        """
        Classify a batch of texts with rate limiting and caching
        
        Args:
            texts: List of texts to classify
            batch_delay: Delay between API calls in seconds
            
        Returns:
            List of predictions (1, 0, or -1 for errors)
        """
        predictions = []
        cache_hits = 0
        
        for i, text in enumerate(tqdm(texts, desc="Classifying texts")):
            text_hash = self._get_text_hash(text)
            
            if text_hash in self.prediction_cache:
                prediction = self.prediction_cache[text_hash]
                cache_hits += 1
            else:
                prediction = self.classify_single_text(text)
                # Rate limiting only for actual API calls
                if i < len(texts) - 1:
                    time.sleep(batch_delay)
            
            predictions.append(prediction)
        
        print(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
        return predictions
    
    def process_dataset(self, df: pd.DataFrame, output_file: str = None, 
                       save_interval: int = 50) -> pd.DataFrame:
        """
        Process the dataset and save results with caching
        
        Args:
            df: DataFrame with 'text' column
            output_file: Path to save results (optional)
            save_interval: Save progress every N predictions
            
        Returns:
            DataFrame with predictions added
        """
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Get texts to classify
        texts = result_df['text'].tolist()
        
        # Initialize predictions column
        result_df['claude_predictions'] = -1
        result_df['claude_confidence'] = None  # Placeholder for future confidence scores
        
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
                time.sleep(1.0)
            
            result_df.iloc[i, result_df.columns.get_loc('claude_predictions')] = prediction
            
            # Save progress periodically
            if output_file and (i + 1) % save_interval == 0:
                result_df.to_csv(output_file, index=False)
                print(f"Progress saved: {i + 1}/{len(texts)} completed (Cache: {cache_hits}, API: {api_calls})")
        
        print(f"Final stats - Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
        print(f"API calls made: {api_calls}")
        
        # Final save
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return result_df
    
    def evaluate_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate Claude predictions against ground truth labels
        
        Args:
            df: DataFrame with 'labels' and 'claude_predictions' columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Filter out error predictions (-1)
        valid_mask = df['claude_predictions'] != -1
        valid_df = df[valid_mask]
        
        if len(valid_df) == 0:
            return {"error": "No valid predictions found"}
        
        y_true = valid_df['labels']
        y_pred = valid_df['claude_predictions']
        
        # Calculate metrics
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
        
        return metrics

def main():
    """Main execution function"""
    
    # Initialize classifier with caching
    classifier = ClaudeHateClassifier()
    
    # Configuration
    DATASET_PATH = "superset.csv"
    SAMPLE_SIZE = 10
    RANDOM_SEED = 42
    OUTPUT_FILE = "claude_predictions_sample.csv"
    
    # Load and sample dataset
    print("Loading and sampling dataset...")
    df = classifier.load_and_sample_dataset(
        dataset_path=DATASET_PATH,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # Show sample data
    print(f"\nSample of selected data:")
    print(df[['text', 'labels', 'post_author_country_location']].head())
    
    # Process dataset
    print(f"\nStarting classification of {len(df)} samples...")
    
    result_df = classifier.process_dataset(
        df, 
        output_file=OUTPUT_FILE
    )
    
    # Evaluate results if ground truth labels are available
    if 'labels' in result_df.columns:
        print("\nEvaluating predictions...")
        metrics = classifier.evaluate_predictions(result_df)
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key}: {value}")
        
        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
            print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    print(f"\nProcessing complete! Results saved to {OUTPUT_FILE}")
    print(f"Cache files saved in 'cache/' directory for future runs")

if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    main()