#!/usr/bin/env python3
"""
Generate Comprehensive Visualizations for 4-Model Comparison
Creates publication-quality plots for the research report
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Define paths
BASE_DIR = "/home/rebbouh/data_segm"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")

# Color scheme
COLORS = {
    'Model_1_Original_Paper': '#e74c3c',      # Red - suspicious
    'Model_2_Your_Reproduction': '#f39c12',    # Orange - reproduction
    'Model_3_Your_v7': '#3498db',              # Blue - your work v7
    'Model_4_Your_v8': '#2ecc71'               # Green - your work v8 (best of yours)
}

MODEL_LABELS = {
    'Model_1_Original_Paper': 'Paper Model\n(Suspicious)',
    'Model_2_Your_Reproduction': 'Reproduction\nAttempt',
    'Model_3_Your_v7': 'Your v7\n(800px, 100ep)',
    'Model_4_Your_v8': 'Your v8\n(640px, 120ep)'
}

def load_data():
    """Load all metrics"""
    combined_path = os.path.join(RESULTS_DIR, 'all_models_combined.csv')
    
    if not os.path.exists(combined_path):
        print(f"❌ Error: {combined_path} not found")
        print("   Please run script 04_combine_and_analyze.py first")
        sys.exit(1)
    
    df = pd.read_csv(combined_path)
    print(f"✓ Loaded data from: {combined_path}")
    return df

def plot_1_metrics_comparison_bars(df):
    """Bar chart comparing key metrics across all models"""
    print("\nGenerating Plot 1: Metrics Comparison Bars...")
    
    metrics = [
        'Precision(Mask)',
        'Recall(Mask)',
        'mAP50(Mask)',
        'mAP50-95(Mask)',
        'F1-Score(Mask)'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        models = df['Model'].tolist()
        values = df[metric].tolist()
        colors_list = [COLORS[m] for m in models]
        
        bars = ax.bar(range(len(models)), values, color=colors_list, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric.replace('(Mask)', '').replace('(Box)', ''))
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Performance Metrics Comparison - All 4 Models', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, '01_metrics_comparison_bars.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close()

def plot_2_performance_gap_visualization(df):
    """Visualize the performance gap relative to original paper"""
    print("\nGenerating Plot 2: Performance Gap Visualization...")
    
    baseline = df[df['Model'] == 'Model_1_Original_Paper'].iloc[0]
    
    metrics = ['mAP50(Mask)', 'F1-Score(Mask)', 'Precision(Mask)', 'Recall(Mask)']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(metrics))
    width = 0.2
    
    for idx, (_, row) in enumerate(df.iterrows()):
        model = row['Model']
        
        # Calculate percentage of baseline
        percentages = [
            (row[metric] / baseline[metric] * 100) if baseline[metric] != 0 else 0
            for metric in metrics
        ]
        
        offset = (idx - 1.5) * width
        bars = ax.bar(x_pos + offset, percentages, width, 
                     label=MODEL_LABELS[model],
                     color=COLORS[model], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.0f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, 
               label='Baseline (Original Paper = 100%)', alpha=0.7)
    ax.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, 
               label='50% of Baseline', alpha=0.5)
    
    ax.set_ylabel('Performance (% of Original Paper)', fontsize=12, fontweight='bold')
    ax.set_title('Reproducibility Gap: Models vs Original Paper Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('(Mask)', '') for m in metrics])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 120)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '02_performance_gap_vs_baseline.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close()

def plot_3_radar_chart(df):
    """Radar chart showing model performance profiles"""
    print("\nGenerating Plot 3: Radar Chart...")
    
    from math import pi
    
    metrics = ['Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 
               'mAP50-95(Mask)', 'F1-Score(Mask)']
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for _, row in df.iterrows():
        model = row['Model']
        values = [row[m] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=MODEL_LABELS[model],
               color=COLORS[model])
        ax.fill(angles, values, alpha=0.15, color=COLORS[model])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('(Mask)', '') for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    plt.title('Model Performance Profile - Radar Chart', 
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '03_radar_chart.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close()

def plot_4_heatmap(df):
    """Heatmap showing all metrics for all models"""
    print("\nGenerating Plot 4: Metrics Heatmap...")
    
    metrics = [
        'Precision(Box)', 'Recall(Box)', 'mAP50(Box)', 'mAP50-95(Box)',
        'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 'mAP50-95(Mask)',
        'F1-Score(Mask)', 'Dice_Coefficient'
    ]
    
    # Create data for heatmap
    heatmap_data = []
    model_names = []
    
    for _, row in df.iterrows():
        model_names.append(MODEL_LABELS[row['Model']].replace('\n', ' '))
        heatmap_data.append([row[m] if m in row else 0 for m in metrics])
    
    heatmap_array = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.replace('(Mask)', '(M)').replace('(Box)', '(B)') 
                        for m in metrics], rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{heatmap_array[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('All Metrics Heatmap - 4 Models', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Metric Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '04_metrics_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close()

def plot_5_v7_vs_v8_detailed(df):
    """Detailed comparison of v7 vs v8"""
    print("\nGenerating Plot 5: v7 vs v8 Detailed Comparison...")
    
    v7 = df[df['Model'] == 'Model_3_Your_v7'].iloc[0]
    v8 = df[df['Model'] == 'Model_4_Your_v8'].iloc[0]
    
    metrics = [
        'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 
        'mAP50-95(Mask)', 'F1-Score(Mask)', 'Speed_Inf'
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Side-by-side bars
    x = np.arange(len(metrics))
    width = 0.35
    
    v7_values = [v7[m] if m in v7 else 0 for m in metrics]
    v8_values = [v8[m] if m in v8 else 0 for m in metrics]
    
    bars1 = ax1.bar(x - width/2, v7_values, width, label='v7 (800px, 100ep)',
                    color=COLORS['Model_3_Your_v7'], alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, v8_values, width, label='v8 (640px, 120ep)',
                    color=COLORS['Model_4_Your_v8'], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Metric Value')
    ax1.set_title('v7 vs v8: Metric Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('(Mask)', '') for m in metrics], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Percentage difference
    differences = [(v8_values[i] - v7_values[i]) / v7_values[i] * 100 
                   if v7_values[i] != 0 else 0 
                   for i in range(len(metrics))]
    
    colors_diff = ['green' if d > 0 else 'red' for d in differences]
    
    bars = ax2.barh(range(len(metrics)), differences, color=colors_diff, alpha=0.6, edgecolor='black')
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        width_bar = bar.get_width()
        ax2.text(width_bar + (2 if width_bar > 0 else -2), bar.get_y() + bar.get_height()/2.,
                f'{diff:+.1f}%',
                ha='left' if width_bar > 0 else 'right', 
                va='center', fontsize=9, fontweight='bold')
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Percentage Change (%)')
    ax2.set_title('v8 vs v7: Percentage Improvement', fontweight='bold')
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels([m.replace('(Mask)', '') for m in metrics])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Your Models: v7 vs v8 Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, '05_v7_vs_v8_detailed.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close()

def plot_6_suspicious_gap(df):
    """Highlight the suspicious performance gap"""
    print("\nGenerating Plot 6: Suspicious Performance Gap...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = df['Model'].tolist()
    map50_values = df['mAP50(Mask)'].tolist()
    f1_values = df['F1-Score(Mask)'].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, map50_values, width, label='mAP50(Mask)',
                   color=[COLORS[m] for m in models], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_values, width, label='F1-Score(Mask)',
                   color=[COLORS[m] for m in models], alpha=0.5, edgecolor='black', hatch='//')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight the gap
    baseline_map50 = map50_values[0]
    baseline_f1 = f1_values[0]
    
    # Draw gap arrows
    for i in range(1, len(models)):
        # mAP50 gap
        gap_start = map50_values[i]
        gap_end = baseline_map50
        mid_point = (gap_start + gap_end) / 2
        
        ax.annotate('', xy=(i - width/2, gap_end), xytext=(i - width/2, gap_start),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        
        gap_pct = ((gap_end - gap_start) / gap_end * 100)
        ax.text(i - width/2 - 0.15, mid_point, f'{gap_pct:.0f}% gap',
               rotation=90, va='center', fontsize=8, color='red', fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('CRITICAL FINDING: Reproducibility Gap\n(Original Paper vs All Other Models)', 
                 fontsize=14, fontweight='bold', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add warning box
    props = dict(boxstyle='round', facecolor='yellow', alpha=0.3, edgecolor='red', linewidth=2)
    textstr = '⚠️ WARNING: 35-40% Performance Gap\nReproducibility Failure'
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='center', horizontalalignment='right',
           bbox=props, fontweight='bold', color='red')
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '06_suspicious_performance_gap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close()

def main():
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Create plots directory if not exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate all plots
    plot_1_metrics_comparison_bars(df)
    plot_2_performance_gap_visualization(df)
    plot_3_radar_chart(df)
    plot_4_heatmap(df)
    plot_5_v7_vs_v8_detailed(df)
    plot_6_suspicious_gap(df)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*80)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("\nGenerated Plots:")
    print("  1. 01_metrics_comparison_bars.png - All metrics across 4 models")
    print("  2. 02_performance_gap_vs_baseline.png - Gap relative to original paper")
    print("  3. 03_radar_chart.png - Performance profile comparison")
    print("  4. 04_metrics_heatmap.png - All metrics heatmap")
    print("  5. 05_v7_vs_v8_detailed.png - Your v7 vs v8 analysis")
    print("  6. 06_suspicious_performance_gap.png - Highlighted reproducibility gap")
    print("\nNext: Run comprehensive report generation script")
    print("="*80)

if __name__ == "__main__":
    main()