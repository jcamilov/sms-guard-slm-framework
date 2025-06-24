import csv
from collections import Counter
from sklearn.metrics import f1_score
import os

# Nombre del archivo CSV de resultados (hardcoded)
CSV_FILENAME = os.path.join('experiment_results', 'small_test_dataset_experiment_results_20250624_095916.csv')
MODEL_NAME = 'gemma-3n-e4b-it'

# Etiquetas posibles
POS_LABEL = 'smishing'
NEG_LABEL = 'benign'

def compute_metrics(y_true, y_pred, positive_label="smishing", negative_label="benign"):
    # Conteos
    TP = sum((yt == positive_label and yp == positive_label) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == negative_label and yp == negative_label) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == negative_label and yp == positive_label) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == positive_label and yp == negative_label) for yt, yp in zip(y_true, y_pred))
    total = TP + TN + FP + FN

    # Métricas
    accuracy = (TP + TN) / total if total else 0
    tpr = TP / (TP + FN) if (TP + FN) else 0  # Sensibilidad
    tnr = TN / (TN + FP) if (TN + FP) else 0  # Especificidad
    fpr = FP / (FP + TN) if (FP + TN) else 0
    fnr = FN / (FN + TP) if (FN + TP) else 0
    # F1 Score
    fs = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    return {
        "Acc": accuracy,
        "FS": fs,
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr
    }

def main():
    # Dictionary to store data for each prompt
    prompt_data = {}
    
    with open(CSV_FILENAME, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Solo consideramos filas que no sean error
            if row['model_classification'] not in ['ERROR', 'unclassified']:
                prompt_name = row['prompt_name']
                if prompt_name not in prompt_data:
                    prompt_data[prompt_name] = {'y_true': [], 'y_pred': []}
                
                prompt_data[prompt_name]['y_true'].append(row['original_classification'].strip().lower())
                prompt_data[prompt_name]['y_pred'].append(row['model_classification'].strip().lower())
    
    # Debug: Print what we found
    print("Debug: Found prompts:", list(prompt_data.keys()))
    for prompt_name, data in prompt_data.items():
        print(f"Debug: {prompt_name} has {len(data['y_true'])} samples")
    
    if not prompt_data:
        print("No hay datos válidos para calcular métricas.")
        return
    
    # Calculate metrics for each prompt
    all_metrics = {}
    for prompt_name, data in prompt_data.items():
        if data['y_true']:  # Only if we have data for this prompt
            all_metrics[prompt_name] = compute_metrics(data['y_true'], data['y_pred'])
    
    # Display results in comparison table
    print("\n+-------------------+--------------+--------------+--------------+--------------+--------------+--------------+")
    print("|   Prompt          |    Acc       |    FS        |   TPR        |   TNR        |   FPR        |   FNR        |")
    print("+-------------------+--------------+--------------+--------------+--------------+--------------+--------------+")
    
    for prompt_name, metrics in all_metrics.items():
        print(f"| {prompt_name:<16}  | {metrics['Acc']*100:>10.2f}%  | {metrics['FS']*100:>10.2f}%  | {metrics['TPR']*100:>10.2f}%  | {metrics['TNR']*100:>10.2f}%  | {metrics['FPR']*100:>10.2f}%  | {metrics['FNR']*100:>10.2f}%  |")
    
    print("+-------------------+--------------+--------------+--------------+--------------+--------------+--------------+")
    
    # Show sample sizes for each prompt
    print("\nSample sizes:")
    for prompt_name, data in prompt_data.items():
        print(f"{prompt_name}: {len(data['y_true'])} samples")

if __name__ == "__main__":
    main() 