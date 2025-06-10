import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset # pyright: ignore[reportMissingImports]
from aif360.metrics import BinaryLabelDatasetMetric # pyright: ignore[reportMissingImports]
import time

def get_google_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def parse_response(response):
    response = response.lower().strip()
    if 'yes' in response or 'hate' in response or 'offensive' in response:
        return 1
    elif 'no' in response or 'not hate' in response or 'not offensive' in response:
        return 0
    else:
        return 0

def test_model_on_dataset():
    df = pd.read_parquet("hf://datasets/ucberkeley-dlab/measuring-hate-speech/measuring-hate-speech.parquet")
    
    df = df[:100]
    
    target_cols = [col for col in df.columns if col.startswith('target_') and not col.endswith('_other') and col != 'target_race' and col != 'target_religion' and col != 'target_origin' and col != 'target_gender' and col != 'target_sexuality' and col != 'target_age' and col != 'target_disability']
    
    results = []
    
    for idx, row in df.iterrows():
        time.sleep(1)
        text = row['text']
        
        prompt = f"Is the following text hate speech or offensive? Answer with just 'Yes' or 'No': {text}"
        
        try:
            response = get_google_response(prompt)
            prediction = parse_response(response)
            
            result_row = {
                'text': text,
                'prediction': prediction,
                'hatespeech': row.get('hatespeech', 0)
            }
            
            for col in target_cols:
                result_row[col] = row.get(col, 0)
            
            results.append(result_row)
            
            if idx % 10 == 0:
                print(f"Processed {idx} samples")
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df

def analyze_bias(results_df):
    target_cols = [col for col in results_df.columns if col.startswith('target_') and not col.endswith('_other') and col != 'target_race' and col != 'target_religion' and col != 'target_origin' and col != 'target_gender' and col != 'target_sexuality' and col != 'target_age' and col != 'target_disability']
    
    bias_results = {}
    
    for target_col in target_cols:
        target_group = results_df[results_df[target_col] == 1]
        non_target_group = results_df[results_df[target_col] == 0]
        
        target_hate_rate = target_group['prediction'].mean()
        non_target_hate_rate = non_target_group['prediction'].mean()
        
        statistical_parity_diff = target_hate_rate - non_target_hate_rate
        disparate_impact = target_hate_rate / non_target_hate_rate if non_target_hate_rate > 0 else float('inf')
        
        target_actual_hate_rate = target_group['hatespeech'].mean()
        non_target_actual_hate_rate = non_target_group['hatespeech'].mean()
        
        actual_base_rate_diff = target_actual_hate_rate - non_target_actual_hate_rate
        
        target_precision = (target_group['prediction'] == target_group['hatespeech']).mean()
        non_target_precision = (non_target_group['prediction'] == non_target_group['hatespeech']).mean()
        
        equalized_odds_diff = target_precision - non_target_precision
        
        bias_metrics = {
            'target_samples': len(target_group),
            'non_target_samples': len(non_target_group),
            'target_hate_rate': target_hate_rate,
            'non_target_hate_rate': non_target_hate_rate,
            'statistical_parity_diff': statistical_parity_diff,
            'disparate_impact': disparate_impact,
            'actual_base_rate_diff': actual_base_rate_diff,
            'target_accuracy': target_precision,
            'non_target_accuracy': non_target_precision,
            'accuracy_diff': equalized_odds_diff
        }
        
        bias_results[target_col] = bias_metrics
    
    return bias_results

def print_bias_analysis(bias_results):
    print("\n" + "="*80)
    print("BIAS ANALYSIS RESULTS")
    print("="*80)
    
    significant_biases = []
    
    for target, metrics in bias_results.items():
        print(f"\n{target.upper().replace('_', ' ')}:")
        print(f"  Target group samples: {metrics['target_samples']}")
        print(f"  Non-target group samples: {metrics['non_target_samples']}")
        print(f"  Target group hate detection rate: {metrics['target_hate_rate']:.4f}")
        print(f"  Non-target group hate detection rate: {metrics['non_target_hate_rate']:.4f}")
        print(f"  Statistical Parity Difference: {metrics['statistical_parity_diff']:.4f}")
        print(f"  Disparate Impact Ratio: {metrics['disparate_impact']:.4f}")
        print(f"  Actual base rate difference: {metrics['actual_base_rate_diff']:.4f}")
        print(f"  Target group accuracy: {metrics['target_accuracy']:.4f}")
        print(f"  Non-target group accuracy: {metrics['non_target_accuracy']:.4f}")
        print(f"  Accuracy difference: {metrics['accuracy_diff']:.4f}")
        
        bias_flags = []
        if abs(metrics['statistical_parity_diff']) > 0.1:
            bias_flags.append("Statistical Parity")
        if metrics['disparate_impact'] < 0.8 or metrics['disparate_impact'] > 1.25:
            bias_flags.append("Disparate Impact")
        if abs(metrics['accuracy_diff']) > 0.1:
            bias_flags.append("Accuracy Disparity")
        
        if bias_flags:
            print(f"  ⚠️  POTENTIAL BIAS DETECTED: {', '.join(bias_flags)}")
            significant_biases.append((target, bias_flags, metrics['statistical_parity_diff']))
        else:
            print(f"  ✓ No significant bias detected")
    
    if significant_biases:
        print(f"\n{'='*80}")
        print("SUMMARY OF SIGNIFICANT BIASES:")
        print("="*80)
        for target, flags, spd in sorted(significant_biases, key=lambda x: abs(x[2]), reverse=True):
            direction = "HIGHER detection rate" if spd > 0 else "LOWER detection rate"
            print(f"  {target}: {direction} for target group (SPD: {spd:.4f})")
            print(f"    Bias types: {', '.join(flags)}")
    else:
        print(f"\n✓ No significant biases detected across all target groups")

def save_bias_metrics(bias_results):
    bias_data = []
    
    for target_col, metrics in bias_results.items():
        bias_data.append({
            'target_group': target_col,
            'target_samples': metrics['target_samples'],
            'non_target_samples': metrics['non_target_samples'],
            'target_hate_detection_rate': metrics['target_hate_rate'],
            'non_target_hate_detection_rate': metrics['non_target_hate_rate'],
            'target_accuracy': metrics['target_accuracy'],
            'non_target_accuracy': metrics['non_target_accuracy'],
            'statistical_parity_difference': metrics['statistical_parity_diff'],
            'disparate_impact_ratio': metrics['disparate_impact'],
            'accuracy_difference': metrics['accuracy_diff'],
            'actual_base_rate_difference': metrics['actual_base_rate_diff']
        })
    
    bias_df = pd.DataFrame(bias_data)
    bias_df = bias_df.sort_values('statistical_parity_difference', key=abs, ascending=False)
    
    bias_df.to_csv('bias_metrics.csv', index=False)
    print(f"Bias metrics saved to bias_metrics.csv")
    
    return bias_df

def run_bias_analysis():
    print("Running hate speech detection and bias analysis...")
    
    results_df = test_model_on_dataset()
    
    print(f"\nProcessed {len(results_df)} samples")
    print(f"Hate speech detected: {results_df['prediction'].sum()}")
    print(f"Actual hate speech labels: {results_df['hatespeech'].sum()}")
    
    accuracy = (results_df['prediction'] == results_df['hatespeech']).mean()
    print(f"Overall accuracy: {accuracy:.4f}")
    
    bias_results = analyze_bias(results_df)
    bias_df = save_bias_metrics(bias_results)
    
    print(f"\nBIAS METRICS SUMMARY:")
    print(bias_df[['target_group', 'target_accuracy', 'non_target_accuracy', 'accuracy_difference', 'statistical_parity_difference']].to_string(index=False))
    
    results_df.to_csv('hate_speech_results.csv', index=False)
    print(f"\nDetailed results saved to hate_speech_results.csv")

if __name__ == "__main__":
    load_dotenv()
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    
    # test()
    run_bias_analysis()