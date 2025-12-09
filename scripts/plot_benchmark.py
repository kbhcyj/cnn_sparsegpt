import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    # Load CSVs
    base_dir = "cnn_sparsegpt/experiments/results"
    df_under = pd.read_csv(os.path.join(base_dir, "benchmark_results.csv"))
    df_full = pd.read_csv(os.path.join(base_dir, "benchmark_results_full.csv"))
    
    # Add Training Status column
    df_under["Training"] = "Under-trained"
    df_full["Training"] = "Fully-trained"
    
    # Merge
    df = pd.concat([df_under, df_full], ignore_index=True)
    
    # Method 명칭 통일 (obs -> sparsegpt)
    df["Method"] = df["Method"].replace({"obs": "sparsegpt"})
    
    # Convert numeric columns
    numeric_cols = ["Baseline Acc", "Pruned Acc", "Accuracy Drop"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    return df

def set_style():
    # 논문 스타일 설정 (Times New Roman, High DPI)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    
    # 배경 설정
    sns.set_style("whitegrid")

def plot_benchmark(df):
    set_style()
    
    # Create plots directory
    output_dir = "cnn_sparsegpt/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fully-trained 상태에서의 성능 비교 (Bar Chart)
    plt.figure(figsize=(10, 6), dpi=300)
    df_full = df[df["Training"] == "Fully-trained"]
    
    ax = sns.barplot(data=df_full, x="Model", y="Pruned Acc", hue="Method", palette="viridis", edgecolor="black")
    
    # Baseline 표시
    for i, model in enumerate(df_full["Model"].unique()):
        baseline = df_full[df_full["Model"] == model]["Baseline Acc"].values[0]
        ax.hlines(y=baseline, xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='--', linewidth=2, label='Baseline' if i == 0 else "")
        ax.text(i, baseline + 2, f"Base: {baseline:.1f}%", color='red', ha='center', va='bottom', fontweight='bold')
        
    plt.title("Pruning Accuracy Comparison (Fully Trained Model)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(title="Method", loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_fully_trained_acc.png"), dpi=300)
    print(f"Saved {os.path.join(output_dir, '1_fully_trained_acc.png')}")
    
    # 2. Under-trained vs Fully-trained: SparseGPT 성능 변화
    plt.figure(figsize=(10, 6), dpi=300)
    df_obs = df[df["Method"] == "sparsegpt"]
    
    sns.barplot(data=df_obs, x="Model", y="Pruned Acc", hue="Training", palette="Blues", edgecolor="black")
    
    # 값 표시
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().text(p.get_x() + p.get_width()/2, p.get_height() + 1, 
                           f"{p.get_height():.1f}%", ha='center', va='bottom', fontsize=10)

    plt.title("Effect of Training Convergence on SparseGPT Performance")
    plt.ylabel("Pruned Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(title="Training Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_training_effect_sparsegpt.png"), dpi=300)
    print(f"Saved {os.path.join(output_dir, '2_training_effect_sparsegpt.png')}")

    # 3. Accuracy Drop Comparison (Fully Trained)
    plt.figure(figsize=(10, 6), dpi=300)
    
    sns.barplot(data=df_full, x="Model", y="Accuracy Drop", hue="Method", palette="Reds", edgecolor="black")
    
    plt.title("Accuracy Drop (Lower is Better)")
    plt.ylabel("Accuracy Drop (%p)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_accuracy_drop.png"), dpi=300)
    print(f"Saved {os.path.join(output_dir, '3_accuracy_drop.png')}")

if __name__ == "__main__":
    try:
        df = load_data()
        plot_benchmark(df)
        print("\nAll plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
