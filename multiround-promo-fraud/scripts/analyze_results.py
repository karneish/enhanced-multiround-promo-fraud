"""
Results Analysis Script for Adaptive Multi-Model Detection Layer Research

Analyzes experiment results and generates publication-quality plots.

Usage:
    python analyze_results.py -r ../result/adaptive_experiment/<timestamp>/
    python analyze_results.py -r ../result/adaptive_experiment/<timestamp>/ --format pdf
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plots will not be generated.")


VARIANT_COLORS = {
    'baseline_xgb': '#2c3e50',
    'individual_xgb': '#34495e',
    'individual_rf': '#3498db',
    'individual_et': '#1abc9c',
    'individual_hgb': '#e74c3c',
    'individual_lr': '#9b59b6',
    'adaptive_full': '#f39c12',
    'adaptive_no_stability': '#e67e22',
    'adaptive_no_historical': '#d35400',
    'adaptive_f1_only': '#c0392b',
    'adaptive_equal_avg': '#8e44ad',
}

VARIANT_LABELS = {
    'baseline_xgb': 'TPNE + XGBoost (Baseline)',
    'individual_xgb': 'TPNE + XGBoost (single)',
    'individual_rf': 'TPNE + Random Forest',
    'individual_et': 'TPNE + Extra Trees',
    'individual_hgb': 'TPNE + HistGradientBoost',
    'individual_lr': 'TPNE + Logistic Regression',
    'adaptive_full': 'TPNE + Adaptive (Proposed)',
    'adaptive_no_stability': 'Adaptive w/o Stability',
    'adaptive_no_historical': 'Adaptive w/o Historical',
    'adaptive_f1_only': 'Adaptive F1-Only',
    'adaptive_equal_avg': 'Simple Average Ensemble',
}


def extract_round_num(round_str):
    if pd.isna(round_str):
        return 0
    s = str(round_str)
    if s.startswith('round_'):
        try:
            return int(s.split('_')[1])
        except (IndexError, ValueError):
            return 0
    try:
        return int(s)
    except ValueError:
        return 0


def load_results(result_dir):
    all_path = os.path.join(result_dir, 'all_results.csv')
    summary_path = os.path.join(result_dir, 'summary.csv')

    all_df = pd.read_csv(all_path) if os.path.exists(all_path) else pd.DataFrame()
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()

    if not summary_df.empty and 'round' in summary_df.columns:
        summary_df['round_num'] = summary_df['round'].apply(extract_round_num)
        summary_df = summary_df.sort_values(['variant', 'round_num'])

    if not all_df.empty and 'round' in all_df.columns:
        all_df['round_num'] = all_df['round'].apply(extract_round_num)

    return all_df, summary_df


def print_summary_table(summary_df):
    if summary_df.empty:
        print("No summary data available.")
        return

    print('\n' + '='*90)
    print('RESULTS SUMMARY TABLE')
    print('='*90)

    for category in ['baseline', 'individual', 'proposed', 'ablation']:
        cat_df = summary_df[summary_df['category'] == category]
        if cat_df.empty:
            continue
        print(f'\n--- {category.upper()} ---')
        for _, row in cat_df.iterrows():
            variant = row.get('variant', '')
            label = VARIANT_LABELS.get(variant, variant)
            f1 = row.get('f1', 0)
            recall = row.get('recall', 0)
            precision = row.get('precision', 0)
            auc = row.get('auc', 0)
            rnd = row.get('round', '')
            print(f'  {label:<45} {str(rnd):<12} F1={f1:.4f}  Rec={recall:.4f}  Prec={precision:.4f}  AUC={auc:.4f}')


def plot_metric_over_rounds(summary_df, metric_col, metric_name, output_dir, fmt='png'):
    if summary_df.empty or not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    variants = summary_df['variant'].unique()

    for variant in variants:
        vdf = summary_df[summary_df['variant'] == variant].copy()
        if metric_col not in vdf.columns:
            continue
        vdf = vdf.sort_values('round_num')
        x = vdf['round_num'].values
        y = pd.to_numeric(vdf[metric_col], errors='coerce').values

        color = VARIANT_COLORS.get(variant, None)
        label = VARIANT_LABELS.get(variant, variant)
        is_proposed = 'adaptive_full' in variant
        is_baseline = 'baseline' in variant

        marker = 'o' if is_proposed else ('s' if is_baseline else '^')
        linewidth = 2.5 if is_proposed else 1.5
        markersize = 8 if is_proposed else 5
        zorder = 10 if is_proposed else 5

        ax.plot(x, y, marker=marker, color=color, label=label,
                linewidth=linewidth, markersize=markersize, alpha=0.9, zorder=zorder)

    ax.set_xlabel('Adversarial Round', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Across Adversarial Rounds', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{metric_col}_over_rounds.{fmt}')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_weight_evolution(summary_df, output_dir, fmt='png'):
    if summary_df.empty or not HAS_MATPLOTLIB:
        return

    weight_cols = [c for c in summary_df.columns if c.startswith('weight_')]
    if not weight_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    adaptive_df = summary_df[summary_df['variant'] == 'adaptive_full'].copy()
    if adaptive_df.empty:
        return

    adaptive_df = adaptive_df.sort_values('round_num')
    x = adaptive_df['round_num'].values

    model_names = ['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for name, color in zip(model_names, colors):
        col = f'weight_{name}'
        if col in adaptive_df.columns:
            y = pd.to_numeric(adaptive_df[col], errors='coerce').values
            ax.plot(x, y, marker='o', color=color, label=name, linewidth=2, markersize=6)

    ax.set_xlabel('Adversarial Round', fontsize=12)
    ax.set_ylabel('Model Weight', fontsize=12)
    ax.set_title('Adaptive Detector Weight Evolution Across Rounds', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'weight_evolution.{fmt}')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_individual_model_f1(summary_df, output_dir, fmt='png'):
    if summary_df.empty or not HAS_MATPLOTLIB:
        return

    f1_cols = [c for c in summary_df.columns if c.startswith('individual_f1_')]
    if not f1_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    adaptive_df = summary_df[summary_df['variant'] == 'adaptive_full'].copy()
    if adaptive_df.empty:
        return

    adaptive_df = adaptive_df.sort_values('round_num')
    x = adaptive_df['round_num'].values

    model_names = ['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for name, color in zip(model_names, colors):
        col = f'individual_f1_{name}'
        if col in adaptive_df.columns:
            y = pd.to_numeric(adaptive_df[col], errors='coerce').values
            ax.plot(x, y, marker='s', color=color, label=name, linewidth=1.5, markersize=5, alpha=0.8)

    ax.set_xlabel('Adversarial Round', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title('Individual Model F1 Within Adaptive System', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'individual_model_f1.{fmt}')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_comparison_bar(summary_df, output_dir, fmt='png'):
    if summary_df.empty or not HAS_MATPLOTLIB:
        return

    if 'round_num' not in summary_df.columns:
        return

    max_round = summary_df['round_num'].max()
    last_df = summary_df[summary_df['round_num'] == max_round].copy()

    if last_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [('f1', 'Macro F1'), ('recall', 'Recall'), ('auc', 'AUC')]

    for ax, (col, title) in zip(axes, metrics):
        if col not in last_df.columns:
            continue

        last_df_sorted = last_df.sort_values(col, ascending=True)
        variants = last_df_sorted['variant'].values
        values = pd.to_numeric(last_df_sorted[col], errors='coerce').values
        colors = [VARIANT_COLORS.get(v, '#666666') for v in variants]
        labels = [VARIANT_LABELS.get(v, v) for v in variants]

        bars = ax.barh(range(len(variants)), values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(title, fontsize=12)
        ax.set_title(f'{title} at Final Round (R{max_round})', fontsize=13, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'final_round_comparison.{fmt}')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_performance_degradation(summary_df, output_dir, fmt='png'):
    if summary_df.empty or not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = summary_df['variant'].unique()

    for variant in variants:
        vdf = summary_df[summary_df['variant'] == variant].copy()
        if 'f1' not in vdf.columns or vdf.empty:
            continue

        vdf = vdf.sort_values('round_num')
        f1_values = pd.to_numeric(vdf['f1'], errors='coerce').values

        if len(f1_values) > 0 and f1_values[0] > 0:
            degradation = ((f1_values - f1_values[0]) / f1_values[0]) * 100
            x = vdf['round_num'].values

            color = VARIANT_COLORS.get(variant, None)
            label = VARIANT_LABELS.get(variant, variant)
            linewidth = 2.5 if 'adaptive_full' in variant else 1.5

            ax.plot(x, degradation, marker='o', color=color, label=label,
                   linewidth=linewidth, alpha=0.9)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Adversarial Round', fontsize=12)
    ax.set_ylabel('F1 Change from Round 0 (%)', fontsize=12)
    ax.set_title('Performance Degradation Across Adversarial Rounds', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'performance_degradation.{fmt}')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_final_round_table(summary_df, output_dir, fmt='png'):
    if summary_df.empty or not HAS_MATPLOTLIB:
        return

    max_round = summary_df['round_num'].max()
    last_df = summary_df[summary_df['round_num'] == max_round].copy()
    if last_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.5 * len(last_df)))
    ax.axis('off')

    display_cols = ['variant', 'f1', 'recall', 'precision', 'auc']
    col_labels = ['Variant', 'F1', 'Recall', 'Precision', 'AUC']

    table_data = []
    for _, row in last_df.iterrows():
        variant = row.get('variant', '')
        label = VARIANT_LABELS.get(variant, variant)
        table_data.append([
            label,
            f"{row.get('f1', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('precision', 0):.4f}",
            f"{row.get('auc', 0):.4f}",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=['#f0f0f0'] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for i, (variant, _) in enumerate(zip(last_df['variant'].values, table_data)):
        color = VARIANT_COLORS.get(variant, '#ffffff')
        table[i + 1, 0].set_facecolor(color + '30')

    plt.title(f'Final Round (R{max_round}) Performance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'final_round_table.{fmt}')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def main():
    parser = argparse.ArgumentParser(description='Analyze adaptive detector experiment results')
    parser.add_argument('-r', '--result-dir', type=str, required=True,
                        help='Path to experiment result directory')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output plot format')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    args = parser.parse_args()

    print(f'Loading results from: {args.result_dir}')
    all_df, summary_df = load_results(args.result_dir)

    if all_df.empty and summary_df.empty:
        print('No results found. Check the result directory path.')
        return

    print(f'Loaded {len(all_df)} total rows, {len(summary_df)} summary rows')

    print_summary_table(summary_df)

    if not args.no_plots and not summary_df.empty:
        print('\nGenerating plots...')
        plot_metric_over_rounds(summary_df, 'f1', 'Macro F1 Score', args.result_dir, args.format)
        plot_metric_over_rounds(summary_df, 'recall', 'Recall', args.result_dir, args.format)
        plot_metric_over_rounds(summary_df, 'precision', 'Precision', args.result_dir, args.format)
        plot_metric_over_rounds(summary_df, 'auc', 'AUC', args.result_dir, args.format)
        plot_weight_evolution(summary_df, args.result_dir, args.format)
        plot_individual_model_f1(summary_df, args.result_dir, args.format)
        plot_comparison_bar(summary_df, args.result_dir, args.format)
        plot_performance_degradation(summary_df, args.result_dir, args.format)
        plot_final_round_table(summary_df, args.result_dir, args.format)
        print('All plots generated.')

    print('\nAnalysis complete.')


if __name__ == '__main__':
    main()
