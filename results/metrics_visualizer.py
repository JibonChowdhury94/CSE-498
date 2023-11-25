import matplotlib.pyplot as plt
import seaborn as sns
import os

class MetricsVisualizer:
    def __init__(self, train_losses, val_losses, train_accuracies, val_accuracies, save_path):
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.save_path = save_path
        self.plot_name = "training_validation_plot.png"
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def plot_metrics(self):
        # Use Seaborn's sophisticated visualization styles
        sns.set_style("whitegrid")
        color_palette = sns.color_palette("husl", 2)  # using the HUSL color space
        
        num_folds = len(self.train_losses)
        
        # Create a figure for plotting with larger font sizes
        fig, axs = plt.subplots(num_folds, 2, figsize=(16, 6 * num_folds))
        plt.rcParams.update({'font.size': 16})

        # Loop through each fold
        for i in range(num_folds):
            # Loss subplot
            axs[i, 0].plot(self.train_losses[i], label='Training Loss', color=color_palette[0], linewidth=2.5)
            axs[i, 0].plot(self.val_losses[i], label='Validation Loss', color=color_palette[1], linewidth=2.5)
            axs[i, 0].set_title(f'Fold {i + 1} Loss', fontsize=18, pad=20)
            axs[i, 0].set_xlabel('Epoch', fontsize=16, labelpad=10)
            axs[i, 0].set_ylabel('Loss', fontsize=16, labelpad=10)
            axs[i, 0].legend(loc='upper right', fancybox=True, shadow=True)

            # Accuracy subplot
            axs[i, 1].plot(self.train_accuracies[i], label='Training Accuracy', color=color_palette[0], linewidth=2.5)
            axs[i, 1].plot(self.val_accuracies[i], label='Validation Accuracy', color=color_palette[1], linewidth=2.5)
            axs[i, 1].set_title(f'Fold {i + 1} Accuracy', fontsize=18, pad=20)
            axs[i, 1].set_xlabel('Epoch', fontsize=16, labelpad=10)
            axs[i, 1].set_ylabel('Accuracy', fontsize=16, labelpad=10)
            axs[i, 1].legend(loc='lower right', fancybox=True, shadow=True)

        # Adjust the layout and show the plots
        plt.tight_layout(h_pad=5)
        fig.savefig(os.path.join(self.save_path, self.plot_name), dpi=300)
        plt.show()
