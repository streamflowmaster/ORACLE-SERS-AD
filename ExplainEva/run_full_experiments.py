import os
import json
import pandas as pd
from batch_generate_multiclass_data_with_multi_interactions import dataset_configs, generate_dataset
from train_transformer import train_model, SyntheticDataset
from interpret_transformer import interpret_transformer
from datetime import datetime
import torch


def run_full_experiments(num_experiments=4, dataset_dir='datasets', checkpoint_dir='checkpoints',
                         output_dir='results/interpret', summary_dir='results/summary',
                         k_values=[2, 3], class_idx=0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """
    Run full experiments: dataset generation, model training, interpretability analysis, and summarize results.

    Args:
        num_experiments: Number of experiments to run
        dataset_dir: Directory to save datasets
        checkpoint_dir: Directory to save model checkpoints
        output_dir: Directory to save interpretability results
        summary_dir: Directory to save summary report
        k_values: List of k values for k-way interaction
        class_idx: Index of the class to analyze logits for
        device: Device for model training and inference
    """
    os.makedirs(summary_dir, exist_ok=True)
    summary_results = []

    for i in range(num_experiments):
        print(f"\n=== Experiment {i + 1}/{num_experiments} ===")

        # Step 1: Generate dataset
        print(f"Generating dataset ds_{i + 1:03d}...")
        config = dataset_configs[i % len(dataset_configs)]
        generate_dataset(config, i + 1)
        data_file = os.path.join(dataset_dir, f'ds_{i + 1:03d}_data.csv')
        settings_file = os.path.join(dataset_dir, f'ds_{i + 1:03d}_settings.json')

        if not (os.path.exists(data_file) and os.path.exists(settings_file)):
            print(f"Dataset ds_{i + 1:03d} generation failed, skipping...")
            continue

        # Load dataset statistics
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        dataset = SyntheticDataset(data_file, settings_file)
        class_dist = pd.read_csv(data_file)['Y'].value_counts(normalize=True).to_dict()

        # Step 2: Train model
        print(f"Training Transformer model for ds_{i + 1:03d}...")
        best_val_acc = train_model(data_file, settings_file, epochs=100, batch_size=256, lr=0.001, device=device)
        model_path = os.path.join(checkpoint_dir, f'model_ds_{i + 1:03d}.pth')

        if not os.path.exists(model_path):
            print(f"Model training for ds_{i + 1:03d} failed, skipping...")
            continue

        # Step 3: Run interpretability analysis
        print(f"Running interpretability analysis for ds_{i + 1:03d}...")
        exp_output_dir = os.path.join(output_dir, f'experiment_{i + 1}')
        interpret_transformer(data_file, settings_file, model_path, exp_output_dir,
                              k_values=k_values, class_idx=class_idx, device=device)

        # Step 4: Collect results
        result = {
            'experiment_id': f'ds_{i + 1:03d}',
            'n_samples': settings['n_samples'],
            'n_variables': settings['n_variables'],
            'n_classes': settings['n_classes'],
            'discrete_ratio': config['discrete_ratio'],
            'decision_ratio': config['decision_ratio'],
            'class_distribution': class_dist,
            'best_val_accuracy': best_val_acc
        }

        # Load k-way interaction results
        kway_results = {}
        for k in k_values:
            kway_file = os.path.join(exp_output_dir, 'kway', f'k{k}_kway_interaction_class_{class_idx}.csv')
            if os.path.exists(kway_file):
                kway_df = pd.read_csv(kway_file).head(3)  # Top 3 interactions
                kway_results[f'k{k}_top_interactions'] = [
                    {col: row[col] for col in row.index if col.startswith('X')} |
                    {'KWay_Interaction': row['KWay_Interaction']}
                    for _, row in kway_df.iterrows()
                ]

        # Load attention results
        rollout_file = os.path.join(exp_output_dir, 'attention_rollout.csv')
        flow_file = os.path.join(exp_output_dir, 'attention_flow.csv')
        if os.path.exists(rollout_file):
            rollout_df = pd.read_csv(rollout_file)
            result['attention_rollout_means'] = rollout_df.mean().to_dict()
        if os.path.exists(flow_file):
            flow_df = pd.read_csv(flow_file)
            result['attention_flow_means'] = flow_df.mean().to_dict()

        summary_results.append(result)

        print(f"Experiment {i + 1} completed: Best Val Acc = {best_val_acc:.4f}")

    # Step 5: Save summary report
    summary_df = pd.DataFrame([{
        'Experiment': r['experiment_id'],
        'N_Samples': r['n_samples'],
        'N_Variables': r['n_variables'],
        'N_Classes': r['n_classes'],
        'Discrete_Ratio': r['discrete_ratio'],
        'Decision_Ratio': r['decision_ratio'],
        **{f'Class_{k}_Dist': v for k, v in r['class_distribution'].items()},
        'Best_Val_Accuracy': r['best_val_accuracy'],
        **{f'K{k}_Top1': str(r.get(f'k{k}_top_interactions', [{}])[0]) for k in k_values},
        **{f'Rollout_{k}': v for k, v in r.get('attention_rollout_means', {}).items()},
        **{f'Flow_{k}': v for k, v in r.get('attention_flow_means', {}).items()}
    } for r in summary_results])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_df.to_csv(os.path.join(summary_dir, f'experiment_summary_{timestamp}.csv'), index=False)
    with open(os.path.join(summary_dir, f'experiment_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_results, f, indent=4, default=str)

    print(f"\nSummary report saved to {summary_dir}/experiment_summary_{timestamp}.csv and .json")


if __name__ == "__main__":
    run_full_experiments(
        num_experiments=100,
        dataset_dir='datasets',
        checkpoint_dir='checkpoints',
        output_dir='results/interpret',
        summary_dir='results/summary',
        k_values=[2, 3],
        class_idx=0
    )