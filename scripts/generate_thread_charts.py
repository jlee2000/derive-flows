#!/usr/bin/env python3
"""Generate charts for Twitter thread on unusual options flow research."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "thread"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Statistics from tradeability_analysis.csv
RAW_SIGNALS = 31
INDEPENDENT_WINDOWS = 11
OVERLAP_PCT = 64.5
CI_LOW = -2.62
CI_HIGH = 11.43

# Statistics from volatility_results.csv (mean absolute returns)
VOL_DATA = {
    'BTC': {'unusual': 1.12, 'baseline': 1.23, 'p': 0.703, 'n': 12},
    'ETH': {'unusual': 2.13, 'baseline': 1.80, 'p': 0.166, 'n': 60},
    'HYPE': {'unusual': 9.55, 'baseline': 3.84, 'p': 0.0003, 'n': 7},
}


def generate_clustering_breakdown():
    """Generate signal clustering breakdown visual for Tweet 2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})

    # Left panel: Stacked bar showing raw vs independent signals
    ax1 = axes[0]

    # Bar data
    categories = ['Raw Signals', 'Independent\nWindows']
    values = [RAW_SIGNALS, INDEPENDENT_WINDOWS]
    overlap_count = RAW_SIGNALS - INDEPENDENT_WINDOWS

    # Create horizontal bars
    bars = ax1.barh(categories, values, color=['#4A90D9', '#2E7D32'], height=0.5, edgecolor='white', linewidth=2)

    # Add overlap portion to raw signals bar
    ax1.barh(['Raw Signals'], [overlap_count], left=[INDEPENDENT_WINDOWS],
             color='#FF6B6B', height=0.5, edgecolor='white', linewidth=2, alpha=0.8)

    # Add value labels
    ax1.text(RAW_SIGNALS + 0.5, 0, f'{RAW_SIGNALS}', va='center', ha='left', fontsize=14, fontweight='bold')
    ax1.text(INDEPENDENT_WINDOWS + 0.5, 1, f'{INDEPENDENT_WINDOWS}', va='center', ha='left', fontsize=14, fontweight='bold')

    # Add overlap annotation
    ax1.annotate(f'{overlap_count} overlapping\n(64.5%)',
                xy=(INDEPENDENT_WINDOWS + overlap_count/2, 0),
                xytext=(25, -1.2),
                fontsize=11,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5))

    ax1.set_xlim(0, 40)
    ax1.set_xlabel('Number of Signals', fontsize=12)
    ax1.set_title('Signal Clustering Impact\n(ETH Calls, 72h window)', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2E7D32', edgecolor='white', label='Independent signals'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='white', alpha=0.8, label='Overlapping (clustered)')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Right panel: Bootstrap confidence interval
    ax2 = axes[1]

    # CI visualization
    ci_center = (CI_LOW + CI_HIGH) / 2
    ci_width = CI_HIGH - CI_LOW

    # Draw the CI bar
    ax2.barh([0], [ci_width], left=[CI_LOW], color='#E0E0E0', height=0.3, edgecolor='#666666', linewidth=1.5)

    # Mark the zero line
    ax2.axvline(x=0, color='#D32F2F', linestyle='--', linewidth=2, label='Zero (no edge)')

    # Mark the CI bounds
    ax2.plot([CI_LOW], [0], 'o', markersize=12, color='#1976D2', zorder=5)
    ax2.plot([CI_HIGH], [0], 'o', markersize=12, color='#1976D2', zorder=5)

    # Add corrected mean point
    corrected_mean = 1.60  # From tradeability_analysis.csv
    ax2.plot([corrected_mean], [0], 's', markersize=10, color='#2E7D32', zorder=6, label=f'Corrected mean: +{corrected_mean:.1f}%')

    # Labels
    ax2.text(CI_LOW, -0.25, f'{CI_LOW:.1f}%', ha='center', va='top', fontsize=11, fontweight='bold', color='#1976D2')
    ax2.text(CI_HIGH, -0.25, f'+{CI_HIGH:.1f}%', ha='center', va='top', fontsize=11, fontweight='bold', color='#1976D2')
    ax2.text(0, 0.25, 'Zero', ha='center', va='bottom', fontsize=10, color='#D32F2F')

    ax2.set_xlim(-6, 15)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_xlabel('Return (%)', fontsize=12)
    ax2.set_title('Bootstrap 95% Confidence Interval\n(Includes Zero = Not Tradeable)', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.legend(loc='upper right', fontsize=10)

    # Add key takeaway
    fig.text(0.5, 0.02,
             "2/3 of signals are echoes of earlier trades. After correction, can't reject zero edge.",
             ha='center', fontsize=12, style='italic', color='#555555')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    output_path = OUTPUT_DIR / "clustering_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Generated: {output_path}")
    return output_path


def generate_vol_prediction():
    """Generate volatility prediction bar chart for Tweet 2."""
    fig, ax = plt.subplots(figsize=(10, 6))

    assets = list(VOL_DATA.keys())
    x = np.arange(len(assets))
    width = 0.35

    unusual_vals = [VOL_DATA[a]['unusual'] for a in assets]
    baseline_vals = [VOL_DATA[a]['baseline'] for a in assets]

    # Create bars
    bars1 = ax.bar(x - width/2, unusual_vals, width, label='After Unusual Flow',
                   color='#4A90D9', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, baseline_vals, width, label='Random Baseline',
                   color='#CCCCCC', edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars1, unusual_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, color='#666666')

    # Add p-value annotations
    for i, asset in enumerate(assets):
        p = VOL_DATA[asset]['p']
        n = VOL_DATA[asset]['n']
        if p < 0.001:
            p_text = f'p<0.001***'
            color = '#2E7D32'
        elif p < 0.05:
            p_text = f'p={p:.3f}*'
            color = '#2E7D32'
        else:
            p_text = f'p={p:.2f}'
            color = '#999999'

        # Position p-value between the two bars, above them
        max_height = max(VOL_DATA[asset]['unusual'], VOL_DATA[asset]['baseline'])
        ax.text(i, max_height + 1.5, p_text, ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold' if p < 0.05 else 'normal')
        ax.text(i, max_height + 2.5, f'(n={n})', ha='center', va='bottom',
                fontsize=9, color='#666666')

    ax.set_ylabel('Mean Absolute Return (24h)', fontsize=12)
    ax.set_title('Volatility Prediction: Unusual Flow vs Baseline\n24h Forward Price Movement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(assets, fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add takeaway
    fig.text(0.5, 0.02,
             "HYPE unusual flow predicts 2.5x larger price moves (p<0.001). Doesn't tell direction, but something is about to move.",
             ha='center', fontsize=11, style='italic', color='#555555')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    output_path = OUTPUT_DIR / "vol_prediction_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Generated: {output_path}")
    return output_path


def main():
    """Generate all thread charts."""
    print("Generating Twitter thread charts...")
    print()

    # Generate clustering breakdown
    clustering_path = generate_clustering_breakdown()

    # Generate volatility prediction chart
    vol_path = generate_vol_prediction()

    print()
    print("Charts generated successfully!")
    print()
    print("Files created:")
    print(f"  - {clustering_path}")
    print(f"  - {vol_path}")
    print()
    print("Existing charts to include:")
    print("  - outputs/unusual_flow_eth.png (Tweet 1)")
    print("  - outputs/eth_signal_consistency.png (Tweet 1)")


if __name__ == "__main__":
    main()
