import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 文件路径
attention_file = 'results/interpret/interaction_analysis.csv'
kway_file = 'results/interpret/top_5_kway_interaction_analysis.csv'
json_file = 'results/summary/experiment_summary_20250711_031908.json'

# 读取 CSV 文件
try:
    attention_data = pd.read_csv(attention_file)
    kway_data = pd.read_csv(kway_file)
except FileNotFoundError as e:
    print(f"Error: CSV 文件未找到 - {e}")
    exit(1)

# 读取 JSON 文件
try:
    with open(json_file, 'r') as f:
        config_data = json.load(f)
    config_df = pd.DataFrame(config_data)
except FileNotFoundError as e:
    print(f"Error: JSON 文件未找到 - {e}")
    exit(1)

# 验证 CSV 数据列
expected_columns = ['experiment', 'recall']
if not all(col in attention_data.columns for col in expected_columns) or \
        not all(col in kway_data.columns for col in expected_columns):
    print("Error: CSV 文件缺少 'experiment' 或 'recall' 列")
    exit(1)

# 验证 JSON 数据列
expected_json_columns = ['experiment_id', 'n_samples', 'n_variables', 'n_classes', 'best_val_accuracy']
if not all(col in config_df.columns for col in expected_json_columns):
    print("Error: JSON 文件缺少必要列")
    exit(1)


# 动态映射 experiment_id 到 experiment
def extract_experiment_number(experiment_id):
    match = re.match(r'ds_(\d+)', experiment_id)
    if match:
        return str(int(match.group(1)))  # 去除前导零，例如 '001' -> '1'
    return experiment_id


config_df['experiment'] = config_df['experiment_id'].apply(extract_experiment_number)
config_df['experiment'] = config_df['experiment'].astype(str)
attention_data['experiment'] = attention_data['experiment'].astype(str)
kway_data['experiment'] = kway_data['experiment'].astype(str)

# 合并 Attention 和 K-way 数据
attention_data['method'] = attention_data['attention_type']
kway_data['method'] = kway_data['kway_file']
combined_data = pd.concat([attention_data[['experiment', 'recall', 'method']],
                           kway_data[['experiment', 'recall', 'method']]],
                          ignore_index=True)

# 合并 JSON 配置
merged_data = pd.merge(combined_data,
                       config_df[['experiment', 'n_samples', 'n_variables', 'n_classes', 'best_val_accuracy']],
                       on='experiment', how='inner')

# 检查未匹配的 experiment
missing_experiments = set(combined_data['experiment']) - set(config_df['experiment'])
if missing_experiments:
    print(f"警告：以下 experiment 在 JSON 配置中未找到匹配项：{missing_experiments}")
    print("这些实验的数据将被排除在折线图之外")

# 检查合并后数据是否为空
if merged_data.empty:
    print("Error: 合并后数据为空，可能由于 experiment_id 和 experiment 无法匹配")
    print("请检查 JSON 文件中的 experiment_id 和 CSV 文件中的 experiment 列")
    exit(1)

# 调试：打印 merged_data 的摘要
print("\nMerged Data Summary:")
print(merged_data[
          ['experiment', 'method', 'recall', 'n_samples', 'n_variables', 'n_classes', 'best_val_accuracy']].head())
print("\nUnique best_val_accuracy values:", merged_data['best_val_accuracy'].unique())

# 为 best_val_accuracy 分箱
unique_best_val = merged_data['best_val_accuracy'].nunique()
if unique_best_val > 1:  # 需要至少 2 个唯一值以进行分箱
    bins = min(5, unique_best_val)  # 动态调整分箱数
    try:
        merged_data['best_val_accuracy_bin'] = pd.cut(merged_data['best_val_accuracy'],
                                                      bins=bins,
                                                      labels=range(bins),  # 使用整数标签 0, 1, 2, ...
                                                      include_lowest=True,
                                                      precision=3)
        # 转换为整数类型
        merged_data['best_val_accuracy_bin'] = merged_data['best_val_accuracy_bin'].astype(int)
    except ValueError as e:
        print(f"Error in binning best_val_accuracy: {e}")
        print("Falling back to using raw best_val_accuracy values")
        merged_data['best_val_accuracy_bin'] = merged_data['best_val_accuracy'].rank(method='dense').astype(int) - 1
else:
    print("Warning: Only one unique best_val_accuracy value. Using rank-based labels.")
    merged_data['best_val_accuracy_bin'] = 0  # Single value gets label 0

# 调试：打印 best_val_accuracy_bin 的分布
print("\nBest Validation Accuracy Bin Distribution:")
print(merged_data['best_val_accuracy_bin'].value_counts().sort_index())

# 配置字段列表
config_fields = ['n_samples', 'n_variables', 'n_classes', 'best_val_accuracy']

# 为每个配置字段绘制折线图
for field in config_fields:
    plt.figure(figsize=(10, 6))

    if field == 'best_val_accuracy':
        # 使用分箱的整数标签
        grouped_data = merged_data.groupby(['best_val_accuracy_bin', 'method'], observed=True)[
            'recall'].mean().reset_index()
        x_field = 'best_val_accuracy_bin'
        x_label = 'Best Validation Accuracy (Bin Index)'
    else:
        # 对离散字段，直接分组并排序
        grouped_data = merged_data.groupby([field, 'method'], observed=True)['recall'].mean().reset_index()
        grouped_data[field] = grouped_data[field].astype(float)  # 转换为数值以便排序
        grouped_data = grouped_data.sort_values(field)
        x_field = field
        x_label = field.replace("_", " ").title()

    # 调试：打印 grouped_data
    print(f"\nGrouped Data for {field}:")
    print(grouped_data[[x_field, 'method', 'recall']])

    # 使用 seaborn 绘制折线图
    sns.lineplot(x=x_field, y='recall', hue='method', marker='o', data=grouped_data, palette='Set2',
                 errorbar=('sd', 1),  # 使用标准差作为误差条)
                 )

    plt.title(f'Recall Trend by {x_label} and Method')
    plt.xlabel(x_label)
    plt.ylabel('Average Recall')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    output_file = f'recall_trend_by_{field}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表：{output_file}")
print(merged_data)
# 为每个配置字段绘制折线图
# 为每个配置字段绘制箱线图
for field in config_fields:
    plt.figure(figsize=(12, 6))

    if field == 'best_val_accuracy':
        # 使用分箱的整数标签
        x_field = 'best_val_accuracy_bin'
        x_label = 'Best Validation Accuracy (Bin Index)'
        plot_data = merged_data
    else:
        x_field = field
        x_label = field.replace("_", " ").title()
        plot_data = merged_data
        # 对离散字段排序
        plot_data[field] = plot_data[field].astype(float)  # 转换为数值以便排序
        plot_data = plot_data.sort_values(field)

    # 使用 seaborn 绘制箱线图
    sns.boxplot(x=x_field, y='recall', hue='method', data=plot_data, palette='Set2')

    plt.title(f'Recall Distribution by {x_label} and Method')
    plt.xlabel(x_label)
    plt.ylabel('Recall')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    output_file = f'recall_boxplot_by_{field}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表：{output_file}")