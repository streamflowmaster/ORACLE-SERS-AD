import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
import os
import json
from transformer_model import Transformer
from train_transformer import SyntheticDataset

def entropy(Y):
    """Calculate entropy H(Y) for a discrete variable Y."""
    probs = np.bincount(Y) / len(Y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def conditional_entropy(Y, X):
    """Calculate conditional entropy H(Y|X) for discrete Y and X."""
    H_Y_given_X = 0
    unique_X, counts_X = np.unique(X, return_counts=True, axis=0)
    for x, count in zip(unique_X, counts_X):
        prob_X = count / len(X)
        Y_given_X = Y[np.all(X == x, axis=1)]
        if len(Y_given_X) > 0:
            H_Y_given_X += prob_X * entropy(Y_given_X)
    return H_Y_given_X

def k_way_interaction(Y, X_list):
    """Calculate k-way interaction: sum(H(Y|X_i)) - k*H(Y|X_1,...,X_k)."""
    k = len(X_list)
    sum_H_Y_given_Xi = 0
    for X_i in X_list:
        sum_H_Y_given_Xi += conditional_entropy(Y, X_i.reshape(-1, 1))
    X_combined = np.column_stack(X_list)
    H_Y_given_all = conditional_entropy(Y, X_combined)
    return sum_H_Y_given_Xi - k * H_Y_given_all

def standard_interaction_information(Y, X_list):
    """Calculate standard k-way interaction information II(Y; X_1, ..., X_k)."""
    k = len(X_list)
    total = 0
    for r in range(k + 1):
        for subset in combinations(range(k), r):
            if len(subset) == 0:
                total += entropy(Y)
            else:
                X_subset = np.column_stack([X_list[i] for i in subset])
                total += (-1) ** (len(subset) + 1) * conditional_entropy(Y, X_subset)
    return total

def compute_kway_interaction(k_values, df, all_features, continuous_cols, output_dir, class_idx=0):
    """Compute k-way interaction for specified k values using logits for a specific class."""
    disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    for col in continuous_cols:
        if col in df.columns:
            df[f'{col}_bin'] = disc.fit_transform(df[[col]]).astype(int)

    logit_col = f'logit_class_{class_idx}'
    if logit_col not in df.columns:
        raise ValueError(f"Column {logit_col} not found in DataFrame")
    df['ad_logits_bin'] = (df[logit_col] > df[logit_col].median()).astype(int)

    os.makedirs(output_dir, exist_ok=True)

    for k in k_values:
        print(f"\nComputing k-way interactions for k={k} (class {class_idx})...")
        kway_results = []

        feature_cols = [f'{col}_bin' if col in continuous_cols else col for col in all_features]
        total_combinations = len(list(combinations(feature_cols, k)))
        print(f"Total combinations for k={k}: {total_combinations}")

        for combo in tqdm(combinations(feature_cols, k), total=total_combinations,
                          desc=f"Processing k={k} combinations"):
            X_list = [df[col].values for col in combo]
            Y = df['ad_logits_bin'].values
            assert all(len(X) == len(Y) for X in X_list), \
                f"Inconsistent sample sizes: Y={len(Y)}, X_list={[len(X) for X in X_list]}"
            kway = k_way_interaction(Y, X_list)
            result_dict = {f'X{i + 1}': col for i, col in enumerate(combo)}
            result_dict['KWay_Interaction'] = kway
            kway_results.append(result_dict)

        kway_df = pd.DataFrame(kway_results).sort_values(by='KWay_Interaction', ascending=False)
        print(f"\nk={k} K-way Interaction Top 10 (class {class_idx}):\n", kway_df.head(10))
        kway_df.to_csv(os.path.join(output_dir, f'k{k}_kway_interaction_class_{class_idx}.csv'), index=False)

def attention_rollout(model, attention_weights, num_layers, num_heads):
    """Compute Attention Rollout for Transformer model, averaging over batch dimension."""
    batch_size, seq_len = attention_weights[0].shape[0], attention_weights[0].shape[-1]
    rollout = torch.eye(seq_len, device=attention_weights[0].device).unsqueeze(0).repeat(num_heads, 1, 1)

    for layer in range(num_layers):
        attn = attention_weights[layer]  # [batch_size, num_heads, seq_len, seq_len]
        if attn.dim() == 4:
            attn = attn.mean(dim=0)  # Average across batch_size: [num_heads, seq_len, seq_len]
        else:
            raise ValueError(f"Expected 4D attention weights, got {attn.dim()}D")
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        rollout = torch.bmm(attn, rollout)
    return rollout

def attention_flow(model, attention_weights, num_layers, num_heads):
    """Compute Attention Flow for Transformer model, averaging over batch dimension."""
    batch_size, seq_len = attention_weights[0].shape[0], attention_weights[0].shape[-1]
    flow = torch.ones(num_heads, seq_len, device=attention_weights[0].device) / seq_len

    for layer in range(num_layers):
        attn = attention_weights[layer]  # [batch_size, num_heads, seq_len, seq_len]
        if attn.dim() == 4:
            attn = attn.mean(dim=0)  # Average across batch_size: [num_heads, seq_len, seq_len]
        else:
            raise ValueError(f"Expected 4D attention weights, got {attn.dim()}D")
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        flow = torch.bmm(flow.unsqueeze(1), attn).squeeze(1)
    return flow

def attention_mean(attention_weights, num_layers, num_heads):
    """Compute mean attention by averaging raw attention weights over batch and layer dimensions."""
    batch_size, seq_len = attention_weights[0].shape[0], attention_weights[0].shape[-1]
    mean_scores = []

    for layer in range(num_layers):
        attn = attention_weights[layer]  # [batch_size, num_heads, seq_len, seq_len]
        if attn.dim() == 4:
            attn = attn.mean(dim=0)  # Average across batch_size: [num_heads, seq_len, seq_len]
        else:
            raise ValueError(f"Expected 4D attention weights, got {attn.dim()}D")
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)  # Normalize
        mean_scores.append(attn)

    # Average across layers to get a single [num_heads, seq_len, seq_len] matrix
    mean_scores = torch.stack(mean_scores, dim=0).mean(dim=0)
    return mean_scores

def find_top_attention_pairs(attention_matrix, feature_names, top_k=5):
    """Find the top-k feature pairs with highest attention scores for each head."""
    num_heads, seq_len, _ = attention_matrix.shape
    top_pairs = []

    for head in range(num_heads):
        attn = attention_matrix[head].clone()  # [seq_len, seq_len]
        attn.fill_diagonal_(0)  # 忽略自注意力
        flat_attn = attn.flatten()
        # 选择最大的 top_k 个值
        top_indices = flat_attn.argsort(descending=True)[:top_k]
        scores = flat_attn[top_indices]
        for idx, score in zip(top_indices, scores):
            i, j = divmod(int(idx), seq_len)
            top_pairs.append({
                'head': head,
                'feature_1': feature_names[i],
                'feature_2': feature_names[j],
                'attention_score': float(score)
            })

    # 按注意力分数排序并返回前 top_k 个（全局）
    top_pairs = sorted(top_pairs, key=lambda x: x['attention_score'], reverse=True)[:top_k]
    return top_pairs

def interpret_transformer(data_file, settings_file, model_path, output_dir, k_values=[2, 3], class_idx=0,
                         device="cuda" if torch.cuda.is_available() else "cpu", top_k_pairs=5):
    """Perform interpretability analysis for Transformer model."""
    # Load dataset
    dataset = SyntheticDataset(data_file, settings_file)
    cont_dim, discrete_dims, num_classes = dataset.get_dims()
    print(f"Dataset dims: cont_dim={cont_dim}, discrete_dims={discrete_dims}, num_classes={num_classes}")

    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = Transformer(cont_dim, discrete_dims, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare feature names
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    all_features = (
        [var for var in settings["variable_distributions"] if
         var != "Z" and settings["variable_distributions"][var]["type"] in ["continuous", "normal", "uniform"]] +
        [var for var in settings["variable_distributions"] if
         var != "Z" and settings["variable_distributions"][var]["type"] == "discrete"]
    )
    continuous_cols = [
        var for var in settings["variable_distributions"] if
        var != "Z" and settings["variable_distributions"][var]["type"] in ["continuous", "normal", "uniform"]
    ]

    # Feature names for attention analysis
    feature_names = [f'cont_{i + 1}' for i in range(cont_dim)]
    feature_names += [f'disc_{i + 1}' for i in range(len(discrete_dims))]

    # Collect test data, logits, and attention weights
    all_x_cont = []
    all_x_disc = []
    all_targets = []
    all_logits = []
    all_attention_weights = []

    with torch.no_grad():
        for batch_idx, (x_cont, x_disc, y) in enumerate(tqdm(test_loader, desc="Processing test batches")):
            x_cont, x_disc, y = x_cont.to(device), x_disc.to(device), y.to(device)
            outputs = model(x_cont, x_disc)
            weights = model.get_attention_weights()
            batch_weights = []
            num_layers = len(model.transformer.layers)
            for layer_idx in range(num_layers):
                if layer_idx < len(weights):
                    attn = torch.tensor(weights[layer_idx], device=device)
                    print(f"Batch {batch_idx}, Layer {layer_idx}: Attention weights shape: {attn.shape}")
                    batch_weights.append(attn)
                else:
                    print(f"Batch {batch_idx}, Layer {layer_idx}: No attention weights available")
            all_x_cont.append(x_cont.cpu().numpy())
            all_x_disc.append(x_disc.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())
            all_attention_weights.append(batch_weights)

    print(f"Total batches processed: {len(all_attention_weights)}")
    print(f"Attention weights captured per batch: {[len(batch) for batch in all_attention_weights]}")

    # Prepare test data DataFrame
    x_cont = np.concatenate(all_x_cont, axis=0) if all_x_cont else np.zeros((len(test_dataset), 0))
    x_disc = np.concatenate(all_x_disc, axis=0) if all_x_disc else np.zeros((len(test_dataset), 0))
    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    test_df = pd.DataFrame({
        **{col: dataset.data_df.iloc[test_dataset.indices][col].values for col in all_features},
        'target': targets,
        **{f'logit_class_{i}': logits[:, i] for i in range(num_classes)}
    })

    # Save test data and logits
    os.makedirs(output_dir, exist_ok=True)
    test_df.to_csv(os.path.join(output_dir, 'test_data_logits.csv'), index=False)

    # Compute k-way interactions for specified class
    kway_output_dir = os.path.join(output_dir, 'kway')
    compute_kway_interaction(k_values, test_df, all_features, continuous_cols, kway_output_dir, class_idx=class_idx)

    # Compute Attention Rollout and Attention Flow if weights are available
    if all_attention_weights and any(len(batch) > 0 for batch in all_attention_weights):
        rollout_scores = []
        flow_scores = []
        mean_scores = []
        num_layers = len(model.transformer.layers)
        num_heads = model.transformer.layers[0].self_attn.num_heads

        for batch_idx, batch_attn in enumerate(all_attention_weights):
            if len(batch_attn) != num_layers:
                print(f"Warning: Incomplete attention weights for batch {batch_idx}, got {len(batch_attn)} layers, expected {num_layers}")
                continue
            rollout = attention_rollout(model, batch_attn, num_layers, num_heads)
            flow = attention_flow(model, batch_attn, num_layers, num_heads)
            mean = attention_mean(batch_attn, num_layers, num_heads)
            rollout_scores.append(rollout.cpu().numpy())
            flow_scores.append(flow.cpu().numpy())
            mean_scores.append(mean.cpu().numpy())


        if rollout_scores and flow_scores:
            rollout_scores = np.stack(rollout_scores, axis=0).mean(axis=0)  # [num_heads, seq_len, seq_len]
            flow_scores = np.stack(flow_scores, axis=0).mean(axis=0)  # [num_heads, seq_len]
            mean_scores = np.stack(mean_scores, axis=0).mean(axis=0)

            # Debug: Print statistics of attention matrices
            print(f"Rollout scores shape: {rollout_scores.shape}, max: {rollout_scores.max()}, min: {rollout_scores.min()}, mean: {rollout_scores.mean()}, std: {rollout_scores.std()}")
            print(f"Flow scores shape: {flow_scores.shape}, max: {flow_scores.max()}, min: {flow_scores.min()}, mean: {flow_scores.mean()}, std: {flow_scores.std()}")
            print(f'Mean scores = mean: {mean_scores.mean()}, std: {mean_scores.std()}')
            # Save full attention matrices
            np.save(os.path.join(output_dir, 'attention_rollout.npy'), rollout_scores)
            np.save(os.path.join(output_dir, 'attention_flow.npy'), flow_scores)
            np.save(os.path.join(output_dir, 'attention_mean.npy'), mean_scores)

            # Find top attention pairs
            top_rollout_pairs = find_top_attention_pairs(torch.tensor(rollout_scores), feature_names, top_k=top_k_pairs)
            top_flow_pairs = find_top_attention_pairs(torch.tensor(flow_scores).unsqueeze(-1), feature_names, top_k=top_k_pairs)  # Adjust flow_scores to [num_heads, seq_len, seq_len]
            top_mean_pairs = find_top_attention_pairs(torch.tensor(mean_scores), feature_names, top_k=top_k_pairs)

            # Save top attention pairs
            rollout_pairs_df = pd.DataFrame(top_rollout_pairs)
            flow_pairs_df = pd.DataFrame(top_flow_pairs)
            mean_pairs_df = pd.DataFrame(top_mean_pairs)

            mean_pairs_df.to_csv(os.path.join(output_dir, 'top_attention_mean_pairs.csv'), index=False)
            rollout_pairs_df.to_csv(os.path.join(output_dir, 'top_rollout_pairs.csv'), index=False)
            flow_pairs_df.to_csv(os.path.join(output_dir, 'top_flow_pairs.csv'), index=False)

            print(f"\nTop {top_k_pairs} Rollout Attention Pairs:\n", rollout_pairs_df)
            print(f"\nTop {top_k_pairs} Flow Attention Pairs:\n", flow_pairs_df)
            print(f"\nTop {top_k_pairs} Mean Attention Pairs:\n", mean_pairs_df)

        else:
            print("Warning: No rollout or flow scores computed due to missing attention weights.")
    else:
        print(
            f"Warning: No attention weights captured. Skipping Attention Rollout and Flow. Input dims: cont_dim={cont_dim}, discrete_dims={discrete_dims}")

def interpret_experiments(num_experiments, dataset_dir='datasets', checkpoint_dir='checkpoints',
                         output_dir='results/interpret', k_values=[2, 3], class_idx=0, top_k_pairs=5):
    """Perform interpretability analysis for all experiments."""
    for i in range(num_experiments):
        data_file = os.path.join(dataset_dir, f'ds_{i + 1:03d}_data.csv')
        settings_file = os.path.join(dataset_dir, f'ds_{i + 1:03d}_settings.json')
        model_path = os.path.join(checkpoint_dir, f'model_ds_{i + 1:03d}.pth')

        if not (os.path.exists(data_file) and os.path.exists(settings_file) and os.path.exists(model_path)):
            print(f"Files missing for experiment {i + 1}, skipping...")
            continue

        print(f"\nInterpreting experiment {i + 1}/{num_experiments} for class {class_idx}")
        exp_output_dir = os.path.join(output_dir, f'experiment_{i + 1}')
        try:
            interpret_transformer(data_file, settings_file, model_path, exp_output_dir, k_values=k_values,
                                 class_idx=class_idx, top_k_pairs=top_k_pairs)
        except Exception as e:
            print(f"Interpretability analysis failed for ds_{i + 1:03d}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python interpret_transformer.py <data_file> <settings_file> <model_path>")
    else:
        interpret_transformer(sys.argv[1], sys.argv[2], sys.argv[3])