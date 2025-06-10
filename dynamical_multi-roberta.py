import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load your new dataset
df = pd.read_csv("dynamically_generated.csv")

# Load model
tokenizer = AutoTokenizer.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification")
model = AutoModelForSequenceClassification.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification")

# Check dataset structure
print("Dataset columns:", df.columns.tolist())
print("Dataset shape:", df.shape)
print("First few rows:")
print(df.head())

# Use the correct column names for your dataset
text_col = 'text'
label_col = 'label'

# Convert string labels to binary
def convert_label_to_binary(label):
    """Convert 'hate'/'nothate' to 1/0"""
    return 1 if label == 'hate' else 0

# Apply conversion
df['binary_label'] = df[label_col].apply(convert_label_to_binary)

# Check label distribution
print(f"\nOriginal label distribution:")
print(df[label_col].value_counts())
print(f"\nBinary label distribution:")
print(df['binary_label'].value_counts())

# Sample a few examples
sample_texts = df[text_col].head(10).tolist()
sample_labels = df['binary_label'].head(10).tolist()  # Use converted binary labels

# Define label mapping for the original model
original_label_names = {0: 'Neither', 1: 'Offensive', 2: 'Hate Speech'}
# Define binary label mapping for your dataset
binary_label_names = {0: 'Not Hateful', 1: 'Hateful'}

def map_to_binary(original_prediction):
    """Map 3-class prediction to binary: only 2(Hate Speech)->1, 0&1->0"""
    return 1 if original_prediction == 2 else 0

print("Testing model on sample data (Only Hate Speech -> 1, Neither/Offensive -> 0):\n")

correct_predictions = 0
for i, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = F.softmax(outputs.logits, dim=-1)
        predicted_class_original = torch.argmax(predictions, dim=-1).item()
        
        # Map to binary classification
        predicted_class_binary = map_to_binary(predicted_class_original)
    
    # Check if prediction is correct
    is_correct = predicted_class_binary == true_label
    correct_predictions += is_correct
    
    print(f"Example {i+1}:")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"True Label: {binary_label_names[true_label]} ({true_label})")
    print(f"Original Model Prediction: {original_label_names[predicted_class_original]} ({predicted_class_original})")
    print(f"Binary Prediction: {binary_label_names[predicted_class_binary]} ({predicted_class_binary})")
    print(f"Confidence (original): {predictions[0][predicted_class_original]:.3f}")
    print(f"All scores: Neither={predictions[0][0]:.3f}, Offensive={predictions[0][1]:.3f}, Hate={predictions[0][2]:.3f}")
    print(f"Correct: {'✓' if is_correct else '✗'}")
    print("-" * 80)

# Calculate accuracy
accuracy = correct_predictions / len(sample_texts)
print(f"\nAccuracy on {len(sample_texts)} samples: {correct_predictions}/{len(sample_texts)} ({accuracy*100:.1f}%)")

# Function to evaluate on larger subset (optional)
def evaluate_model(df, num_samples=100):
    """Evaluate model on a subset of the data"""
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    correct = 0
    total = 0
    
    for _, row in sample_df.iterrows():
        text = row[text_col]
        true_label = row['binary_label']  # Use converted binary label
        
        # Get prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_original = torch.argmax(outputs.logits, dim=-1).item()
            predicted_class_binary = map_to_binary(predicted_class_original)
        
        if predicted_class_binary == true_label:
            correct += 1
        total += 1
    
    return correct / total

# Evaluate on larger sample
print(f"\nEvaluating on larger sample...")
larger_accuracy = evaluate_model(df, num_samples=200)
print(f"Accuracy on 200 samples: {larger_accuracy*100:.1f}%")

# Analyze failures in detail
print(f"\n" + "="*80)
print("FAILURE ANALYSIS (Only Hate Speech -> 1, Neither/Offensive -> 0)")
print("="*80)

def analyze_failures(df, num_samples=1000):
    """Find and analyze model failures"""
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    failures = []
    false_positives = []  # Model says hateful, but it's not
    false_negatives = []  # Model says not hateful, but it is
    
    for _, row in sample_df.iterrows():
        text = row[text_col]
        true_label = row['binary_label']  # Use converted binary label
        
        # Get prediction with confidence scores
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
            predicted_class_original = torch.argmax(predictions, dim=-1).item()
            predicted_class_binary = map_to_binary(predicted_class_original)
            confidence = predictions[0][predicted_class_original].item()
        
        # Check for failures
        if predicted_class_binary != true_label:
            failure_info = {
                'text': text,
                'true_label': true_label,
                'predicted_binary': predicted_class_binary,
                'predicted_original': predicted_class_original,
                'confidence': confidence,
                'all_scores': {
                    'neither': predictions[0][0].item(),
                    'offensive': predictions[0][1].item(),
                    'hate': predictions[0][2].item()
                }
            }
            
            failures.append(failure_info)
            
            if predicted_class_binary == 1 and true_label == 0:
                false_positives.append(failure_info)
            elif predicted_class_binary == 0 and true_label == 1:
                false_negatives.append(failure_info)
    
    return failures, false_positives, false_negatives

# Get failure examples
failures, false_positives, false_negatives = analyze_failures(df, num_samples=1000)

print(f"Found {len(failures)} failures out of 1000 samples ({len(failures)/10:.1f}% error rate)")
print(f"False Positives (predicted hateful, actually not): {len(false_positives)}")
print(f"False Negatives (predicted not hateful, actually hateful): {len(false_negatives)}")

# Show False Positive examples (model thinks it's hate speech but it's not)
print(f"\n" + "-"*60)
print("FALSE POSITIVES (Model says HATE SPEECH, but label is NOT HATEFUL)")
print("Note: This includes cases where model predicts 'Hate Speech' but true label is 0")
print("-"*60)

for i, fp in enumerate(false_positives[:10]):  # Show first 10
    print(f"\nFalse Positive #{i+1}:")
    print(f"Text: {fp['text'][:200]}{'...' if len(fp['text']) > 200 else ''}")
    print(f"True Label: Not Hateful (0)")
    print(f"Model Prediction: {original_label_names[fp['predicted_original']]} -> Hateful (1)")
    print(f"Confidence: {fp['confidence']:.3f}")
    print(f"Scores - Neither: {fp['all_scores']['neither']:.3f}, "
          f"Offensive: {fp['all_scores']['offensive']:.3f}, "
          f"Hate: {fp['all_scores']['hate']:.3f}")

# Show False Negative examples (model doesn't predict hate speech but it should)
print(f"\n" + "-"*60)
print("FALSE NEGATIVES (Model says NOT HATE SPEECH, but label is HATEFUL)")
print("Note: This includes 'Neither' and 'Offensive' predictions when true label is 1")
print("-"*60)

for i, fn in enumerate(false_negatives[:10]):  # Show first 10
    print(f"\nFalse Negative #{i+1}:")
    print(f"Text: {fn['text'][:200]}{'...' if len(fn['text']) > 200 else ''}")
    print(f"True Label: Hateful (1)")
    print(f"Model Prediction: {original_label_names[fn['predicted_original']]} -> Not Hateful (0)")
    print(f"Confidence: {fn['confidence']:.3f}")
    print(f"Scores - Neither: {fn['all_scores']['neither']:.3f}, "
          f"Offensive: {fn['all_scores']['offensive']:.3f}, "
          f"Hate: {fn['all_scores']['hate']:.3f}")

# Additional analysis for this mapping
print(f"\n" + "-"*60)
print("MAPPING-SPECIFIC ANALYSIS")
print("-"*60)

# Get sample for analysis if not already defined
if 'sample_for_analysis' not in locals():
    sample_for_analysis = df.sample(n=min(1000, len(df)), random_state=42)

# Count how many predictions fall into each original category
original_predictions = []
for _, row in sample_for_analysis.iterrows():
    text = row[text_col]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_original = torch.argmax(outputs.logits, dim=-1).item()
        original_predictions.append(predicted_class_original)

pred_counts = pd.Series(original_predictions).value_counts().sort_index()
print("Original model predictions distribution:")
for pred_class, count in pred_counts.items():
    if pred_class in original_label_names:
        print(f"  {original_label_names[pred_class]} ({pred_class}): {count} ({count/len(original_predictions)*100:.1f}%)")

print(f"\nWith this mapping:")
print(f"  - Only 'Hate Speech' predictions become 1 (hateful)")
print(f"  - Both 'Neither' and 'Offensive' become 0 (not hateful)")
print(f"  - This is more conservative - fewer things flagged as hateful")

# Analyze patterns in failures
print(f"\n" + "-"*60)
print("FAILURE PATTERN ANALYSIS")
print("-"*60)

if false_positives:
    fp_confidence = [fp['confidence'] for fp in false_positives]
    print(f"False Positives - Average confidence: {sum(fp_confidence)/len(fp_confidence):.3f}")
    print(f"False Positives - Low confidence (<0.6): {sum(1 for c in fp_confidence if c < 0.6)}")

if false_negatives:
    fn_confidence = [fn['confidence'] for fn in false_negatives]
    print(f"False Negatives - Average confidence: {sum(fn_confidence)/len(fn_confidence):.3f}")
    print(f"False Negatives - Low confidence (<0.6): {sum(1 for c in fn_confidence if c < 0.6)}")

# Show confusion matrix
sample_for_analysis = df.sample(n=min(1000, len(df)), random_state=42)
predictions_analysis = []
true_labels = []

for _, row in sample_for_analysis.iterrows():
    text = row[text_col]
    true_label = row['binary_label']  # Use converted binary label
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_original = torch.argmax(outputs.logits, dim=-1).item()
        predicted_class_binary = map_to_binary(predicted_class_original)
        predictions_analysis.append(predicted_class_binary)
        true_labels.append(true_label)

pred_df = pd.DataFrame({
    'true_labels': true_labels,
    'predictions': predictions_analysis
})

print(f"\nConfusion Matrix (1000 samples):")
confusion_matrix = pd.crosstab(pred_df['true_labels'], pred_df['predictions'], 
                              rownames=['True'], colnames=['Predicted'], margins=True)
print(confusion_matrix)

# Calculate metrics
tp = confusion_matrix.loc[1, 1] if 1 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
tn = confusion_matrix.loc[0, 0] if 0 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
fp = confusion_matrix.loc[0, 1] if 0 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
fn = confusion_matrix.loc[1, 0] if 1 in confusion_matrix.index and 0 in confusion_matrix.columns else 0

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}")