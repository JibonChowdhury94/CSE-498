import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc

class PlotVisualizer:
    def __init__(self, label_columns, save_path):
        self.label_columns = label_columns
        self.save_path = save_path
        self.confusion_matrix_plot_name = "confusion_matrix.png"
        self.roc_curve_plot_name = "roc_curve.png"
        self.precision_recall_curve_plot_name = "precision_recall_curve.png"
        
        sns.set_style("whitegrid")  # Use Seaborn's whitegrid style
        self.color_palette = sns.color_palette("husl", len(self.label_columns))

    def plot_multilabel_confusion_matrix(self, true_labels, predictions):

        # Convert the list of lists into numpy arrays and reshape to remove unnecessary dimension
        true_labels = np.vstack(true_labels).reshape(-1, 3)
        predictions = np.vstack(predictions).reshape(-1, 3)

        matrices = multilabel_confusion_matrix(true_labels, predictions)

    
        def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
            df_cm = pd.DataFrame(
                confusion_matrix, index=class_names, columns=class_names
            )
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes)
            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
            axes.set_ylabel('True label')
            axes.set_xlabel('Predicted label')
            axes.set_title("Class: " + class_label)

        # Determine grid size for the subplots
        n_classes = len(self.label_columns)
        ncols = 2
        nrows = n_classes // ncols + (n_classes % ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    
        for matrix, label, ax in zip(matrices, self.label_columns, axes.ravel()):
            print_confusion_matrix(matrix, ax, label, ["No", "Yes"])

        # If there are any remaining subplots, hide them
        for ax in axes.ravel()[n_classes:]:
            ax.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, self.confusion_matrix_plot_name), dpi=300)
        plt.show()
        
    def plot_roc_curve(self, true_labels, probabilities):
        true_labels = np.vstack(true_labels).reshape(-1, len(self.label_columns))
        probabilities = np.vstack(probabilities).reshape(-1, len(self.label_columns))

        # Setting up the figure and axis
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')  # Draw the diagonal line

        # Compute and plot ROC curve for each class
        for i, label in enumerate(self.label_columns):
            fpr, tpr, _ = roc_curve(true_labels[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})', color=self.color_palette[i], linewidth=2.5)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.save_path, self.roc_curve_plot_name), dpi=300)
        plt.show()

    def plot_precision_recall_curve(self, true_labels, probabilities):
        true_labels = np.vstack(true_labels).reshape(-1, len(self.label_columns))
        probabilities = np.vstack(probabilities).reshape(-1, len(self.label_columns))

        # Setting up the figure and axis
        plt.figure(figsize=(10, 8))

        # Compute and plot Precision-Recall curve for each class
        for i, label in enumerate(self.label_columns):
            precision, recall, _ = precision_recall_curve(true_labels[:, i], probabilities[:, i])
            avg_precision = average_precision_score(true_labels[:, i], probabilities[:, i])
            plt.plot(recall, precision, label=f'{label} (Avg. Precision = {avg_precision:.2f})', color=self.color_palette[i], linewidth=2.5)

        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.title('Precision-Recall Curve', fontsize=18)
        plt.legend(loc='best', fontsize=12)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, self.precision_recall_curve_plot_name), dpi=300)
        plt.show()