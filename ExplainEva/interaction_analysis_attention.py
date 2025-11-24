import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import logging
import networkx as nx
from itertools import combinations

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
    """Load model-predicted interactions from pairs CSV file and apply mapping."""
    if not os.path.exists(pairs_file):
        logging.warning(f"Pairs file not found: {pairs_file}")
        return []

    try:
        df = pd.read_csv(pairs_file)
        if df.empty:
            logging.warning(f"Empty pairs file: {pairs_file}")
            return []

        # Select top_k pairs and convert to list of lists
        model_pairs = [[row['feature_1'], row['feature_2']] for _, row in df.head(top_k).iterrows()]
        # Apply mapping
        if var_mapping:
            model_pairs = [[var_mapping.get(item, item) for item in pair] for pair in model_pairs]
            model_pairs = [sorted(pair) for pair in model_pairs]

        logging.info(f"Loaded model pairs from {os.path.basename(pairs_file)}: {model_pairs}")
        return model_pairs
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

    # Compute Graph Edit Distance with timeout
    try:
        ged = nx.graph_edit_distance(G_pred, G_gt, timeout=10)  # Timeout after 10 seconds
        if ged is None:
            logging.warning("GED computation timed out. Using edge difference as fallback.")
            edge_diff = len(set(G_pred.edges) ^ set(G_gt.edges))
            node_diff = len(set(G_pred.nodes) ^ set(G_gt.nodes))
            ged = edge_diff + node_diff
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


def analyze_interactions(num_experiments, dataset_dir='datasets', output_dir='results/interpret', top_k=5):
    """Analyze consistency and accuracy of model interactions across experiments."""
    results = []
    attention_types = ['mean', 'flow', 'rollout']
    pairs_files = {
        'mean': 'top_attention_mean_pairs.csv',
        'flow': 'top_flow_pairs.csv',
        'rollout': 'top_rollout_pairs.csv'
    }

    for i in tqdm(range(1, num_experiments + 1), desc="Analyzing experiments"):
        exp_dir = os.path.join(output_dir, f'experiment_{i}')
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

        # Analyze each attention mechanism
        for attn_type in attention_types:
            pairs_file = os.path.join(exp_dir, pairs_files[attn_type])
            model_pairs = load_model_interactions(pairs_file, top_k=top_k, var_mapping=var_mapping)

            # Compute metrics
            jaccard = compute_jaccard_similarity(model_pairs, ground_truth)
            precision, recall, f1 = compute_precision_recall_f1(model_pairs, ground_truth)
            graph_sim = compute_graph_similarity(model_pairs, ground_truth)

            # Store results
            results.append({
                'experiment': i,
                'attention_type': attn_type,
                'jaccard_similarity': jaccard,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'graph_similarity': graph_sim,
                'num_predicted_pairs': len(model_pairs),
                'num_ground_truth_pairs': len(ground_truth)
            })

            logging.info(f"  {attn_type} Results:")
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
        for attn_type in attention_types:
            for metric in metrics:
                data = results_df[results_df['attention_type'] == attn_type][metric]
                summary_data.update({
                    f'mean_{attn_type}': data.mean(),
                    f'std_{attn_type}': data.std(),
                    f'min_{attn_type}': data.min(),
                    f'max_{attn_type}': data.max()
                })
        summary_df = pd.DataFrame([summary_data])

        logging.info("\nSummary Statistics:")
        logging.info(summary_df.to_string())

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'interaction_analysis.csv'), index=False)
        summary_df.to_csv(os.path.join(output_dir, 'interaction_analysis_summary.csv'), index=False)
    else:
        logging.warning("No results to summarize.")

    return results_df, summary_df


if __name__ == "__main__":
    results_df, summary_df = analyze_interactions(num_experiments=100)