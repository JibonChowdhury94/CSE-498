import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class MetricsCalculator:
    def __init__(self, label_columns):
        self.label_columns = label_columns
#         self.report_name = "metrics_report.csv"
        
    def compute_metrics(self, predictions, true_labels):
        metrics_per_class = {
            label: {
                "accuracy": None,
                "f1_score": None,
                "specificity": None,
                "sensitivity": None
            } for label in self.label_columns
        }

        accuracies = []
        f1_scores = []
        specificities = []
        sensitivities = []
        classification_reports = []

        for label_idx, label in enumerate(self.label_columns):
            pred = [p[0][label_idx] for p in predictions]
            true = [t[0][label_idx] for t in true_labels]

            acc = accuracy_score(true, pred)
            f1 = f1_score(true, pred)
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

            metrics_per_class[label]["accuracy"] = acc
            metrics_per_class[label]["f1_score"] = f1
            metrics_per_class[label]["specificity"] = specificity
            metrics_per_class[label]["sensitivity"] = sensitivity

            accuracies.append(acc)
            f1_scores.append(f1)
            specificities.append(specificity)
            sensitivities.append(sensitivity)

            # Classification report for each label
            classification_reports.append((label, classification_report(true, pred, zero_division=0, digits=3)))

        aggregated_metrics = {
            "accuracy": sum(accuracies) / len(accuracies),
            "f1_score": sum(f1_scores) / len(f1_scores),
            "specificity": sum(specificities) / len(specificities),
            "sensitivity": sum(sensitivities) / len(sensitivities)
        }
        
        # Reshape predictions and true labels and get multi-label classification report
        reshaped_predictions = np.squeeze(predictions, axis=1)
        reshaped_true_labels = np.squeeze(true_labels, axis=1)
        multi_label_report = classification_report(reshaped_true_labels, reshaped_predictions, target_names=self.label_columns, zero_division=1, digits=3)

        return aggregated_metrics, metrics_per_class, classification_reports, multi_label_report
    
    def display_metrics(self, aggregated_metrics, metrics_per_class, classification_reports, multi_label_report):
        print("Aggregated Metrics:")
        print(aggregated_metrics)
        
        print("\nClass-wise Metrics:")
        for label, metrics in metrics_per_class.items():
            print(f"\n{label}:")
            print(metrics)

        print("\nClassification Reports Per Class:")
        for label, report in classification_reports:
            print(f"\n{label}:")
            print(report)
            
        print("\nMulti-label Classification Reports:")
        print(multi_label_report)