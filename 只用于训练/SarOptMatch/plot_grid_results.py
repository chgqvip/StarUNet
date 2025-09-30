import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 加载 CSV 数据 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "grid_results.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ 未找到 grid_results.csv 文件，检查路径: {csv_path}")

df = pd.read_csv(csv_path, header=None, names=["dataset_name", "alpha", "threshold", "score"])
df["alpha"] = df["alpha"].astype(float)
df["threshold"] = df["threshold"].astype(float)
df["score"] = df["score"].astype(float)

# === 提取基础数据集名 ===
last_full_name = df["dataset_name"].iloc[-1]
base_name = last_full_name.split('_a')[0]
last_dataset = base_name  # ✅ 补上这个定义

# === 过滤出该数据集所有组合 ===
filtered_df = df[df["dataset_name"].str.startswith(base_name)]

# === 判断是否所有组合都完成 ===
expected_count = filtered_df["alpha"].nunique() * filtered_df["threshold"].nunique()
if len(filtered_df) < expected_count:
    print(f"⚠️ 数据集 {base_name} 尚未完成所有组合（{len(filtered_df)} / {expected_count}），跳过绘图。")
    exit()

# === 透视成 α × τ 格式 ===
pivot = filtered_df.pivot_table(index="threshold", columns="alpha", values="score", aggfunc="max")


# === 绘制热力图 ===
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "Score"})
plt.title(f"Grid Search Heatmap\nDataset: {last_dataset}", fontsize=14)
plt.xlabel("Alpha (α)")
plt.ylabel("Threshold (τ)")
plt.tight_layout()

# === 保存图像 ===
out_path = os.path.join(project_root, f"{last_dataset}_heatmap.png")
plt.savefig(out_path)
print(f"✅ 热力图已保存: {out_path}")

plt.close()  # ✅ 不显示图像窗口，防止阻塞
