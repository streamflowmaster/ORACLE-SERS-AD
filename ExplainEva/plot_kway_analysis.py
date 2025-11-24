import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_results(input_dir='results/interpret'):
    """Load results from CSV files."""
    results_file = os.path.join(input_dir, 'top_5_kway_interaction_analysis.csv')
    summary_file = os.path.join(input_dir, 'top_5_kway_interaction_analysis_summary.csv')

    try:
        results_df = pd.read_csv(results_file)
        summary_df = pd.read_csv(summary_file)
    except FileNotFoundError as e:
        logging.error(f"Error loading file: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        return None, None

    return results_df, summary_df

def plot_box_plots(results_df, output_dir='results/interpret'):
    """Create box plots for each metric across k-way files."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['jaccard_similarity', 'precision', 'recall', 'f1_score', 'graph_similarity']
    metric_labels = {
        'jaccard_similarity': 'Jaccard Similarity',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'graph_similarity': 'Graph Similarity'
    }

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='kway_file', y=metric, data=results_df)
        plt.title(f'{metric_labels[metric]} Distribution Across Experiments')
        plt.xlabel('K-Way File')
        plt.ylabel(metric_labels[metric])
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved box plot for {metric} to {output_path}")

def plot_line_plots(results_df, output_dir='results/interpret'):
    """Create line plots for each metric across experiments."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['jaccard_similarity', 'precision', 'recall', 'f1_score', 'graph_similarity']
    metric_labels = {
        'jaccard_similarity': 'Jaccard Similarity',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'graph_similarity': 'Graph Similarity'
    }

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for kway_file in results_df['kway_file'].unique():
            subset = results_df[results_df['kway_file'] == kway_file]
            plt.plot(subset['experiment'], subset[metric], marker='o', label=kway_file)
        plt.title(f'{metric_labels[metric]} Across Experiments')
        plt.xlabel('Experiment')
        plt.ylabel(metric_labels[metric])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'lineplot_{metric}.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved line plot for {metric} to {output_path}")

def plot_summary_bar(summary_df, output_dir='results/interpret'):
    """Create bar plots for summary statistics."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = summary_df['metric'].tolist()
    kway_files = ['kway_pairs_1', 'kway_pairs_2']
    metric_labels = {
        'jaccard_similarity': 'Jaccard Similarity',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'graph_similarity': 'Graph Similarity'
    }

    # Plot mean values
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    means_1 = [summary_df[f'mean_{kway_files[0]}'].iloc[i] for i in range(len(metrics))]
    means_2 = [summary_df[f'mean_{kway_files[1]}'].iloc[i] for i in range(len(metrics))]

    plt.bar(x - width/2, means_1, width, label='kway_pairs_1.csv')
    plt.bar(x + width/2, means_2, width, label='kway_pairs_2.csv')
    plt.xlabel('Metric')
    plt.ylabel('Mean Value')
    plt.title('Mean Metric Values Across K-Way Files')
    plt.xticks(x, [metric_labels[m] for m in metrics], rotation=45)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'summary_mean_barplot.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved mean bar plot to {output_path}")

def visualize_kway_results(input_dir='results/interpret', output_dir='results/interpret'):
    """Visualize k-way interaction analysis results."""
    results_df, summary_df = load_results(input_dir)
    if results_df is None or summary_df is None:
        logging.error("Failed to load results. Visualization aborted.")
        return

    # Generate plots
    plot_box_plots(results_df, output_dir)
    logging.info("Visualization complete.")

if __name__ == "__main__":
    visualize_kway_results()