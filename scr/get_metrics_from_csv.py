import csv
from collections import Counter
from sklearn.metrics import f1_score

# Nombre del archivo CSV de resultados (hardcoded)
CSV_FILENAME = 'small_test_dataset_experiment_results_20250624_095916.csv'
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
    y_true = []
    y_pred = []
    with open(CSV_FILENAME, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Solo consideramos filas que no sean error
            if row['model_classification'] not in ['ERROR', 'unclassified']:
                y_true.append(row['original_classification'].strip().lower())
                y_pred.append(row['model_classification'].strip().lower())
    if not y_true:
        print("No hay datos válidos para calcular métricas.")
        return
    metrics = compute_metrics(y_true, y_pred)
    # Mostrar resultados en formato tabla
    print("\n+-------------------+--------------+--------------+--------------+--------------+--------------+--------------+")
    print("|   Model           |    Acc       |    FS        |   TPR        |   TNR        |   FPR        |   FNR        |")
    print("+-------------------+--------------+--------------+--------------+--------------+--------------+--------------+")
    print(f"| {MODEL_NAME:<16}  | {metrics['Acc']*100:>10.2f}%  | {metrics['FS']*100:>10.2f}%  | {metrics['TPR']*100:>10.2f}%  | {metrics['TNR']*100:>10.2f}%  | {metrics['FPR']*100:>10.2f}%  | {metrics['FNR']*100:>10.2f}%  |")
    print("+-------------------+--------------+--------------+--------------+--------------+--------------+--------------+")

if __name__ == "__main__":
    main() 