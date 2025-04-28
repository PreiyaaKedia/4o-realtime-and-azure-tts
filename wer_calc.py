import json
from jiwer import wer
import string
import re

from scipy.io.wavfile import write

dataset_name = "SPRINGLab/IndicTTS_Telugu"
split = "train"

def remove_punctuation(text):  
    # Define Tamil-specific punctuation marks  
    tamil_punctuation = "।"  
      
    # Combine Tamil punctuation and standard punctuation  
    all_punctuation = string.punctuation 
      
    # Remove punctuations from the text  
    return ''.join(char for char in text if char not in all_punctuation) 

with open('wer_results_SPRINGLab_IndicTTS_Telugu_train_20_samples.json', 'r', encoding = 'utf-8') as f:
    data = json.load(f)

reference_texts = []
transcribed_texts = []
results = []

for sample in data["sample_results"]:
    sample_id = sample["sample_id"]
    ref_text = sample["reference"]
    transcribed_text = sample["hypothesis"]
    normalized_ref_text = remove_punctuation(ref_text)
    normalized_hypothesis_text = remove_punctuation(transcribed_text)
    sample_wer = wer(normalized_ref_text.strip(), normalized_hypothesis_text.strip())
    reference_texts.append(normalized_ref_text.strip())
    transcribed_texts.append(normalized_hypothesis_text.strip())    
    results.append({
                "sample_id": sample_id,
                "reference": normalized_ref_text.strip(),
                "hypothesis": normalized_hypothesis_text.strip(),
                "wer": sample_wer
            })
    
# Calculate overall WER
if reference_texts and transcribed_text:
    overall_wer = wer(reference_texts, transcribed_texts)
else:
    overall_wer = 1.0 

# Prepare final results
wer_results = {
    "dataset": dataset_name,
    "split": split,
    "samples_processed": len(results),
    "overall_wer": overall_wer,
    "sample_results": results
}

# Save results to a JSON file
output_file = f"wer_results_{dataset_name.replace('/', '_')}_{split}_{len(results)}_samples_normalized.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(wer_results, f, ensure_ascii=False, indent=2)
