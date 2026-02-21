import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
OUTPUT_DIR = 'output_data'
MODELS = ['EfficientNet', 'ResNet50', 'InceptionV3']

def plot_history_from_csv():
    history_dir = os.path.join(OUTPUT_DIR, 'history')
    if not os.path.exists(history_dir):
        print(f"Error: History directory not found at {history_dir}. Run train.py first.")
        return

    num_models = len(MODELS)
    # Changed to 4 columns to include Recall
    fig, axes = plt.subplots(num_models, 4, figsize=(24, 5 * num_models))

    if num_models == 1:
        # Ensure axes is 2D array even for a single model
        axes = axes.reshape(1, 4)

    print("Loading and plotting training histories...")

    for i, model_name in enumerate(MODELS):
        history_path = os.path.join(history_dir, f'{model_name}_history.csv')
        params_path = os.path.join(history_dir, f'{model_name}_params.csv')

        if not os.path.exists(history_path):
            print(f"Warning: History file for {model_name} not found.")
            continue

        history_df = pd.read_csv(history_path)

        # Attempt to load params for title info
        epochs_trained = "Unknown"
        if os.path.exists(params_path):
            try:
                params_df = pd.read_csv(params_path)
                if 'Epochs_Trained' in params_df.columns:
                    epochs_trained = params_df['Epochs_Trained'].iloc[0]
            except:
                pass

        # --- 1. LOSS PLOT ---
        ax1 = axes[i, 0]
        if 'loss' in history_df.columns:
            ax1.plot(history_df['loss'], label='Train Loss', color='darkorange')
        if 'val_loss' in history_df.columns:
            ax1.plot(history_df['val_loss'], label='Val Loss', color='darkblue')

        ax1.set_title(f'{model_name} - Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- 2. ACCURACY PLOT ---
        ax2 = axes[i, 1]
        if 'accuracy' in history_df.columns and not history_df['accuracy'].isnull().all():
            ax2.plot(history_df['accuracy'], label='Train Accuracy', color='darkorange')
        if 'val_accuracy' in history_df.columns:
            ax2.plot(history_df['val_accuracy'], label='Val Accuracy', color='darkblue')

        ax2.set_title(f'{model_name} - Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        # --- 3. PRECISION PLOT ---
        ax3 = axes[i, 2]
        if 'precision' in history_df.columns and not history_df['precision'].isnull().all():
            ax3.plot(history_df['precision'], label='Train Precision', color='darkorange')
        if 'val_precision' in history_df.columns:
            ax3.plot(history_df['val_precision'], label='Val Precision', color='darkblue')

        ax3.set_title(f'{model_name} - Precision', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)

        # --- 4. RECALL PLOT ---
        ax4 = axes[i, 3]
        if 'recall' in history_df.columns and not history_df['recall'].isnull().all():
            ax4.plot(history_df['recall'], label='Train Recall', color='darkorange')
        if 'val_recall' in history_df.columns:
            ax4.plot(history_df['val_recall'], label='Val Recall', color='darkblue')

        ax4.set_title(f'{model_name} - Recall', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_save_path = os.path.join(OUTPUT_DIR, 'ensemble_training_history.png')
    plt.savefig(plot_save_path)
    print(f"\nTraining history plots saved to {plot_save_path}")
    plt.show()

if __name__ == '__main__':
    plot_history_from_csv()
