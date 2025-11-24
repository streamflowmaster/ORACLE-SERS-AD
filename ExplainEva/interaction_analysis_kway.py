import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import logging
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
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_gt_variable_interactions(settings_file):
    """Load ground truth interactions from settings.json and decompose higher-order interactions into pairwise interactions."""
    try:
        with open(settings_file, 'r') as f:
            json_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading settings file {settings_file}: {e}")
        return []

    variables_with_more_than_one = []
    for latent_function in json_data.get('latent_functions', []):
        for term in latent_function.get('terms', []):
            if 'variables' in term and len(term['variables']) >= 2:
                # Generate all pairwise combinations for the variables in the term
                for pair in combinations(term['variables'], 2):
                    if pair[0] != pair[1]:
                        variables_with_more_than_one.append(sorted(pair))

    # Remove duplicates and ensure pairs have exactly 2 variables
    seen = set()
    processed_list = []
    for sublist in variables_with_more_than_one:
        tuple_sublist = tuple(sublist)
        if tuple_sublist not in seen and len(sublist) == 2:
            processed_list.append(sublist)
            seen.add(tuple_sublist)

    return processed_list


def load_con_or_discs(settings_file):
    """Load discrete and continuous variables from settings.json."""
    try:
        with open(settings_file, 'r') as f:
            json_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading settings file {settings_file}: {e}")
        return [], []

    discrete_vars = []
    continuous_vars = []
    for var, dist in json_data.get("variable_distributions", {}).items():
        if dist.get('type') == 'discrete':
            discrete_vars.append(var)
        else:
            continuous_vars.append(var)

    return discrete_vars, continuous_vars


def load_model_interactions(pairs_file, top_k=5, var_mapping=None):
    """Load model-predicted pairwise interactions from k-way CSV file and apply mapping."""
    if not os.path.exists(pairs_file):
        logging.warning(f"Pairs file not found: {pairs_file}")
        return []

    try:
        df = pd.read_csv(pairs_file)
        if df.empty:
            logging.warning(f"Empty pairs file: {pairs_file}")
            return []

        # Extract pairwise interactions from k-way interactions
        pairwise_interactions = []
        for _, row in df.iterrows():
            # Get all features (values in columns except 'KWay_Interaction')
            features = [row[col] for col in df.columns if col != 'KWay_Interaction' and pd.notna(row[col])]
            score = row['KWay_Interaction']
            logging.debug(f"Row features: {features}, Score: {score}")
            # Generate all pairwise combinations
            for pair in combinations(features, 2):
                # Remove '_bin' suffix and apply mapping
                mapped_pair = [
                    var_mapping.get(feature.replace('_bin', ''), feature.replace('_bin', ''))
                    for feature in pair
                ]
                pairwise_interactions.append((sorted(mapped_pair), score))

        # Sort by score and select top_k unique pairs
        pairwise_interactions.sort(key=lambda x: x[1], reverse=True)
        model_pairs = [pair for pair, _ in pairwise_interactions]
        unique_pairs = []
        seen = set()
        for pair in model_pairs:
            pair_tuple = tuple(pair)
            if pair_tuple not in seen:
                unique_pairs.append(pair)
                seen.add(pair_tuple)
                if len(unique_pairs) >= top_k:
                    break

        logging.info(f"Loaded model pairs from {os.path.basename(pairs_file)}: {unique_pairs}")
        return unique_pairs
    except Exception as e:
        logging.error(f"Error loading pairs file {pairs_file}: {e}")
        return []


def compute_jaccard_similarity(predicted, ground_truth):
    """Compute Jaccard similarity between predicted and ground truth interactions."""
    if not predicted or not ground_truth:
        return 0.0

    # Convert lists to sets of tuples for comparison
    pred_set = set(tuple(pair) for pair in predicted)
    gt_set = set(tuple(pair) for pair in ground_truth)

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union if union > 0 else 0.0


def compute_precision_recall_f1(predicted, ground_truth):
    """Compute precision, recall, and F1 score for predicted interactions."""
    if not predicted and not ground_truth:
        return 0.0, 0.0, 0.0

    # Create all possible pairs
    all_features = set()
    for pair in predicted + ground_truth:
        all_features.update(pair)

    all_pairs = [sorted([f1, f2]) for f1 in all_features for f2 in all_features if f1 < f2]
    if not all_pairs:
        logging.warning("No valid pairs to evaluate.")
        return 0.0, 0.0, 0.0

    # Create binary labels
    pred_set = set(tuple(pair) for pair in predicted)
    gt_set = set(tuple(pair) for pair in ground_truth)

    y_true = [1 if tuple(pair) in gt_set else 0 for pair in all_pairs]
    y_pred = [1 if tuple(pair) in pred_set else 0 for pair in all_pairs]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1


def compute_graph_similarity(predicted, ground_truth):
    """Compute graph similarity using normalized Graph Edit Distance."""
    # Create graphs
    G_pred = nx.Graph()
    G_gt = nx.Graph()

    # Add edges from interactions
    for pair in predicted:
        G_pred.add_edge(pair[0], pair[1])
    for pair in ground_truth:
        G_gt.add_edge(pair[0], pair[1])

    # Compute Graph Edit Distance with timeout for large graphs
    try:
        # Use a timeout to avoid excessive computation
        ged = nx.graph_edit_distance(G_pred, G_gt, timeout=10)  # Timeout after 10 seconds
        if ged is None:
            logging.warning("GED computation timed out. Using edge difference as fallback.")
            edge_diff = len(set(G_pred.edges) ^ set(G_gt.edges))
            node_diff = len(set(G_pred.nodes) ^ set(G_gt.nodes))
            ged = edge_diff + node_diff  # Fallback approximation
    except Exception as e:
        logging.warning(f"Error computing GED: {e}. Using edge difference as fallback.")
        edge_diff = len(set(G_pred.edges) ^ set(G_gt.edges))
        node_diff = len(set(G_pred.nodes) ^ set(G_gt.nodes))
        ged = edge_diff + node_diff

    # Normalize GED
    max_nodes = max(len(G_pred.nodes), len(G_gt.nodes))
    max_edges = max(len(G_pred.edges), len(G_gt.edges))
    max_ged = max_nodes + max_edges  # Simplified upper bound
    if max_ged == 0:
        return 1.0 if len(predicted) == 0 and len(ground_truth) == 0 else 0.0

    normalized_ged = ged / max_ged if max_ged > 0 else 0.0
    similarity = 1.0 - min(normalized_ged, 1.0)

    return similarity


def analyze_kway_interactions(num_experiments, dataset_dir='datasets', output_dir='results/interpret', top_k=5):
    """Analyze k-way algorithm interactions across experiments with graph similarity."""
    results = []
    kway_files = ['k2_kway_interaction_class_0.csv', 'k3_kway_interaction_class_0.csv']

    for i in tqdm(range(1, num_experiments + 1), desc="Analyzing experiments"):
        exp_dir = os.path.join(output_dir, f'experiment_{i}', 'kway')
        settings_file = os.path.join(dataset_dir, f'ds_{i:03d}_settings.json')

        if not os.path.exists(settings_file):
            logging.warning(f"Skipping experiment {i}: Missing settings file {settings_file}")
            continue

        logging.info(f"Analyzing experiment {i}/{num_experiments}...")

        # Load ground truth and variable mappings
        ground_truth = load_gt_variable_interactions(settings_file)
        discrete_vars, continuous_vars = load_con_or_discs(settings_file)
        var_mapping = {f'cont_{j + 1}': var for j, var in enumerate(continuous_vars)}
        var_mapping.update({f'disc_{j + 1}': var for j, var in enumerate(discrete_vars)})

        # Analyze each k-way CSV file
        for kway_file in kway_files:
            pairs_file = os.path.join(exp_dir, kway_file)
            model_pairs = load_model_interactions(pairs_file, top_k=top_k, var_mapping=var_mapping)

            # Compute metrics
            jaccard = compute_jaccard_similarity(model_pairs, ground_truth)
            precision, recall, f1 = compute_precision_recall_f1(model_pairs, ground_truth)
            graph_sim = compute_graph_similarity(model_pairs, ground_truth)

            # Store results
            results.append({
                'experiment': i,
                'kway_file': kway_file,
                'jaccard_similarity': jaccard,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'graph_similarity': graph_sim,
                'num_predicted_pairs': len(model_pairs),
                'num_ground_truth_pairs': len(ground_truth)
            })

            logging.info(f"  {kway_file} Results:")
            logging.info(f"    Jaccard Similarity: {jaccard:.4f}")
            logging.info(f"    Precision: {precision:.4f}")
            logging.info(f"    Recall: {recall:.4f}")
            logging.info(f"    F1 Score: {f1:.4f}")
            logging.info(f"    Graph Similarity: {graph_sim:.4f}")
            logging.info(f"    Predicted Pairs: {model_pairs}")
            logging.info(f"    Ground Truth Pairs: {ground_truth}")

    # Summarize results
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    if not results_df.empty:
        metrics = ['jaccard_similarity', 'precision', 'recall', 'f1_score', 'graph_similarity']
        summary_data = {'metric': metrics}
        for kway_file in kway_files:
            file_key = kway_file.replace('.csv', '')
            for metric in metrics:
                data = results_df[results_df['kway_file'] == kway_file][metric]
                summary_data.update({
                    f'mean_{file_key}': data.mean(),
                    f'std_{file_key}': data.std(),
                    f'min_{file_key}': data.min(),
                    f'max_{file_key}': data.max()
                })
        summary_df = pd.DataFrame([summary_data])

        logging.info("\nSummary Statistics:")
        logging.info(summary_df.to_string())

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, f'top_{top_k}_kway_interaction_analysis.csv'), index=False)
        summary_df.to_csv(os.path.join(output_dir, f'top_{top_k}_kway_interaction_analysis_summary.csv'), index=False)
    else:
        logging.warning("No results to summarize.")

    return results_df, summary_df


if __name__ == "__main__":
    results_df, summary_df = analyze_kway_interactions(num_experiments=100,top_k=3)
    results_df, summary_df = analyze_kway_interactions(num_experiments=100,top_k=5)
    results_df, summary_df = analyze_kway_interactions(num_experiments=100,top_k=10)
