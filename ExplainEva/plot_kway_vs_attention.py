import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 文件路径
attention_file = 'results/interpret/interaction_analysis.csv'
kway_file = 'results/interpret/top_5_kway_interaction_analysis.csv'
output_dir = 'results/plots'  # 输出目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 读取 CSV 文件
try:
    attention_data = pd.read_csv(attention_file)
    kway_data = pd.read_csv(kway_file)
except FileNotFoundError as e:
    print(f"Error: 文件未找到 - {e}")
    exit(1)

# 验证数据列
expected_columns = ['experiment', 'jaccard_similarity', 'precision', 'recall', 'f1_score', 'graph_similarity']
if not all(col in attention_data.columns for col in expected_columns) or \
        not all(col in kway_data.columns for col in expected_columns):
    print("Error: CSV 文件缺少必要列")
    exit(1)

# 合并数据并添加方法标识
attention_data['method'] = attention_data['attention_type']
kway_data['method'] = kway_data['kway_file']
combined_data = pd.concat([attention_data, kway_data], ignore_index=True)

# 检查合并后数据是否为空
if combined_data.empty:
    print("Error: 合并后数据为空，请检查输入文件内容")
    exit(1)

# 调试：打印 combined_data 的摘要
print("\nCombined Data Summary:")
print(combined_data[['experiment', 'method'] + expected_columns[1:]].head())
print("\nUnique methods:", combined_data['method'].unique())

# 保存源数据到 CSV
source_data_file = os.path.join(output_dir, 'boxplot_source_data.csv')
combined_data[['experiment', 'method'] + expected_columns[1:]].to_csv(source_data_file, index=False)
print(f"已保存源数据：{source_data_file}")

# 指标列表
metrics = ['jaccard_similarity', 'precision', 'recall', 'f1_score', 'graph_similarity']

# 为每个指标绘制箱线图
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='method', y=metric, data=combined_data, palette='Set2')
    plt.title(f'Boxplot of {metric.replace("_", " ").title()} by Method')
    plt.xlabel('Method')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    output_file = os.path.join(output_dir, f'boxplot_{metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表：{output_file}")