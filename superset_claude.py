import pandas as pd
import anthropic
import time
import json
from typing import List, Dict
import os
from tqdm import tqdm
import dotenv

class ClaudeHateClassifier:
    def __init__(self, api_key: str = None):
        """
        Initialize the Claude hate speech classifier
        
        Args:
            api_key: Anthropic API key. If None, will look for ANTHROPIC_API_KEY env var
        """
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20240620"
        
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
        Classify a single text using Claude API
        
        Args:
            text: Text to classify
            max_retries: Maximum number of retry attempts
            
        Returns:
            int: 1 for hateful, 0 for not hateful, -1 for error
        """
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
                    return 1
                elif result == "0":
                    return 0
                else:
                    print(f"Unexpected response: {result}. Retrying...")
                    continue
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Failed to classify text after {max_retries} attempts")
                    return -1
        
        return -1
    
    def classify_batch(self, texts: List[str], batch_delay: float = 1.0) -> List[int]:
        """
        Classify a batch of texts with rate limiting
        
        Args:
            texts: List of texts to classify
            batch_delay: Delay between API calls in seconds
            
        Returns:
            List of predictions (1, 0, or -1 for errors)
        """
        predictions = []
        
        for i, text in enumerate(tqdm(texts, desc="Classifying texts")):
            prediction = self.classify_single_text(text)
            predictions.append(prediction)
            
            # Rate limiting
            if i < len(texts) - 1:  # Don't sleep after the last item
                time.sleep(batch_delay)
        
        return predictions
    
    def process_dataset(self, df: pd.DataFrame, output_file: str = None, 
                       batch_size: int = None, save_interval: int = 50) -> pd.DataFrame:
        """
        Process the entire dataset and save results
        
        Args:
            df: DataFrame with 'text' column
            output_file: Path to save results (optional)
            batch_size: Process only first N rows (optional)
            save_interval: Save progress every N predictions
            
        Returns:
            DataFrame with predictions added
        """
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Limit batch size if specified
        if batch_size:
            result_df = result_df.head(batch_size)
            print(f"Processing first {batch_size} rows")
        
        # Get texts to classify
        texts = result_df['text'].tolist()
        
        # Initialize predictions column
        result_df['claude_predictions'] = -1
        result_df['claude_confidence'] = None  # Placeholder for future confidence scores
        
        print(f"Starting classification of {len(texts)} texts...")
        
        # Process texts
        for i, text in enumerate(tqdm(texts, desc="Classifying texts")):
            prediction = self.classify_single_text(str(text))
            result_df.iloc[i, result_df.columns.get_loc('claude_predictions')] = prediction
            
            # Save progress periodically
            if output_file and (i + 1) % save_interval == 0:
                result_df.to_csv(output_file, index=False)
                print(f"Progress saved: {i + 1}/{len(texts)} completed")
            
            # Rate limiting
            time.sleep(1.0)  # Adjust based on your API rate limits
        
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
    
    # Initialize classifier
    # Make sure to set your ANTHROPIC_API_KEY environment variable
    classifier = ClaudeHateClassifier()
    
    # Configuration - Adjust these values as needed
    MAX_ROWS_TO_LOAD = 100  # Set this to limit dataset size from the start
    PROCESS_BATCH_SIZE = 100  # How many of the loaded rows to actually process
    
    # Load dataset with row limit
    print(f"Loading dataset (max {MAX_ROWS_TO_LOAD} rows)...")
    df = pd.read_csv("superset.csv", nrows=MAX_ROWS_TO_LOAD)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head())
    
    # Show label distribution if available
    if 'labels' in df.columns:
        print(f"\nLabel distribution:")
        print(df['labels'].value_counts())
    
    # Process dataset
    print("\nStarting classification...")
    
    result_df = classifier.process_dataset(
        df, 
        output_file="claude_predictions.csv",
        batch_size=PROCESS_BATCH_SIZE  # Process only this many rows
    )
    
    # Evaluate results if ground truth labels are available
    if 'labels' in result_df.columns:
        print("\nEvaluating predictions...")
        metrics = classifier.evaluate_predictions(result_df)
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key}: {value}")
        
        print(f"\nConfusion Matrix:")
        print(f"TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
        print(f"FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
    
    print(f"\nProcessing complete! Results saved to claude_predictions.csv")

if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    # Example usage
    main()