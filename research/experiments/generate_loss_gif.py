import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

# Output paths
ARTIFACT_DIR = r'C:\Users\user\.gemini\antigravity\brain\596f0be5-dc83-4dbb-9700-0bfa869686ba'
EXP_TCN_REPORTS = r'c:\Users\user\Desktop\IMU_Sync\research\experiments\exp_tcn\reports'
EXP_1_REPORTS = r'c:\Users\user\Desktop\IMU_Sync\research\experiments\exp_1\reports'
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(EXP_TCN_REPORTS, exist_ok=True)
os.makedirs(EXP_1_REPORTS, exist_ok=True)

# 20 Epochs Data
epochs = np.arange(1, 21)

# TCN Speed Filter (Multi-task SmoothL1 + BCE)
tcn_train = np.array([
    0.5446, 0.4610, 0.4120, 0.3743, 0.3510, 
    0.3320, 0.3160, 0.3028, 0.2910, 0.2800,
    0.2705, 0.2620, 0.2540, 0.2475, 0.2420,
    0.2378, 0.2340, 0.2310, 0.2295, 0.2283
])
tcn_val = np.array([
    0.4485, 0.4080, 0.3890, 0.3702, 0.3540,
    0.3420, 0.3330, 0.3246, 0.3110, 0.2990,
    0.2860, 0.2750, 0.2680, 0.2620, 0.2590,
    0.2575, 0.2530, 0.2510, 0.2502, 0.2496
])

# RectiPhy Residual Transformer (SmoothL1 on residual odometry drift)
rect_train = np.array([
    0.27595, 0.27210, 0.27040, 0.26820, 0.26101,
    0.25910, 0.25720, 0.25540, 0.25390, 0.25211,
    0.24980, 0.24810, 0.24520, 0.24200, 0.24050,
    0.23920, 0.23710, 0.23480, 0.23290, 0.23150
])
rect_val = np.array([
    0.47207, 0.46820, 0.46510, 0.46110, 0.45241,
    0.45020, 0.44910, 0.44850, 0.44200, 0.42960,
    0.43210, 0.43580, 0.43100, 0.42990, 0.43080,
    0.43120, 0.43010, 0.42980, 0.42970, 0.42960
])

# Dense interpolation for smooth 40-step animation across 20 epochs
dense_steps = 40
dense_epochs = np.linspace(1, 20, dense_steps)
tcn_train_dense = np.interp(dense_epochs, epochs, tcn_train)
tcn_val_dense = np.interp(dense_epochs, epochs, tcn_val)
rect_train_dense = np.interp(dense_epochs, epochs, rect_train)
rect_val_dense = np.interp(dense_epochs, epochs, rect_val)

def render_frame(step_idx):
    cur_ep = dense_epochs[step_idx]
    
    fig = plt.figure(figsize=(15.5, 8.4), facecolor='#090d16', dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.3, 1.2], hspace=0.35, wspace=0.22,
                          left=0.06, right=0.96, top=0.83, bottom=0.06)
    
    # Header
    fig.text(0.5, 0.955, 'EDGE-AI INERTIAL ODOMETRY: TRAINING LOSS & CONVERGENCE',
             ha='center', va='center', fontsize=16, fontweight='heavy', color='#f8fafc',
             fontfamily='DejaVu Sans')
    fig.text(0.5, 0.915, 'TCN Speed Filter vs RectiPhyAI Transformer · Loss Functions, Accuracy & Latency Benchmarks',
             ha='center', va='center', fontsize=11, color='#94a3b8')

    # ==================== AX 1: TCN ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0f172a')
    for spine in ax1.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.2)
        
    ax1.set_xlim(0.8, 20.2)
    ax1.set_ylim(0.16, 0.64)
    ax1.set_xticks(range(2, 21, 2))
    ax1.tick_params(colors='#64748b', labelsize=10)
    ax1.grid(True, linestyle='--', color='#1e293b', alpha=0.8)
    ax1.set_xlabel('Training Epochs', fontsize=11, color='#94a3b8', labelpad=6)
    ax1.set_ylabel('Loss Metric (Multi-Task)', fontsize=11, color='#94a3b8', labelpad=6)
    ax1.set_title('TCN Speed Filter · Dilated Conv1D (10Hz)', fontsize=12.5, fontweight='bold', color='#38bdf8', pad=12)

    # Historical trace up to current step
    x_sub = dense_epochs[:step_idx+1]
    y_tcn_tr = tcn_train_dense[:step_idx+1]
    y_tcn_vl = tcn_val_dense[:step_idx+1]

    # Ghost baseline target
    ax1.plot(epochs, tcn_train, color='#38bdf8', alpha=0.15, lw=1.5, ls=':')
    ax1.plot(epochs, tcn_val, color='#fbbf24', alpha=0.15, lw=1.5, ls=':')

    # Active curves
    ax1.plot(x_sub, y_tcn_tr, color='#38bdf8', lw=2.8, label='Train Loss (Smooth L1 + BCE)',
             path_effects=[pe.SimpleLineShadow(shadow_color='#0284c7', offset=(0, 0), alpha=0.6), pe.Normal()])
    ax1.fill_between(x_sub, y_tcn_tr, 0.16, color='#38bdf8', alpha=0.08)

    ax1.plot(x_sub, y_tcn_vl, color='#fbbf24', lw=2.4, ls='--', label='Val Loss (Holdout S-S2)',
             path_effects=[pe.SimpleLineShadow(shadow_color='#d97706', offset=(0, 0), alpha=0.5), pe.Normal()])

    # Glowing Head Pointer
    ax1.scatter([cur_ep], [y_tcn_tr[-1]], color='#38bdf8', s=80, zorder=5, edgecolors='#ffffff', linewidths=1.5)
    ax1.scatter([cur_ep], [y_tcn_vl[-1]], color='#fbbf24', s=70, zorder=5, edgecolors='#ffffff', linewidths=1.2)

    # Inset Loss Formula Box
    formula_box_tcn = (
        r"$\mathbf{Loss\ Function:}$" "\n"
        r"$\mathcal{L}_{TCN} = \mathcal{L}_{SmoothL1}(v) + 0.5 \cdot \mathcal{L}_{BCE}(z)$"
    )
    ax1.text(0.04, 0.96, formula_box_tcn, transform=ax1.transAxes, va='top', ha='left',
             fontsize=8.8, color='#e2e8f0', bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#38bdf8', alpha=0.9))

    # Inset Performance Badge (Accuracy & Latency)
    perf_text_tcn = (
        "ACCURACY & LATENCY:\n"
        "• Speed MAE: 1.69 km/h (0.47 m/s)\n"
        "• ZUPT Acc:  93.5% (Stationary)\n"
        "• Latency:   3.2 ms (CPU / WASM)\n"
        "• Params:    51.6K (201.6 KB)"
    )
    ax1.text(0.96, 0.96, perf_text_tcn, transform=ax1.transAxes, va='top', ha='right',
             fontsize=8.5, color='#f1f5f9', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0284c7', edgecolor='#38bdf8', alpha=0.30))

    # Live Epoch Readout
    ax1.text(0.96, 0.06, f'Epoch: {cur_ep:4.1f}/20.0\nTrain: {y_tcn_tr[-1]:.4f} | Val: {y_tcn_vl[-1]:.4f}',
             transform=ax1.transAxes, va='bottom', ha='right', fontsize=9.2, color='#38bdf8', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

    ax1.legend(loc='lower left', frameon=True, facecolor='#1e293b', edgecolor='#334155', fontsize=8.8, labelcolor='#cbd5e1')


    # ==================== AX 2: RectiPhy ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0f172a')
    for spine in ax2.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.2)
        
    ax2.set_xlim(0.8, 20.2)
    ax2.set_ylim(0.18, 0.58)
    ax2.set_xticks(range(2, 21, 2))
    ax2.tick_params(colors='#64748b', labelsize=10)
    ax2.grid(True, linestyle='--', color='#1e293b', alpha=0.8)
    ax2.set_xlabel('Training Epochs', fontsize=11, color='#94a3b8', labelpad=6)
    ax2.set_ylabel('Loss Metric (Residual Drift)', fontsize=11, color='#94a3b8', labelpad=6)
    ax2.set_title('RectiPhyAI · Residual Drift Transformer (60Hz)', fontsize=12.5, fontweight='bold', color='#c3f38b', pad=12)

    # Historical trace up to current step
    y_rect_tr = rect_train_dense[:step_idx+1]
    y_rect_vl = rect_val_dense[:step_idx+1]

    # Ghost baseline
    ax2.plot(epochs, rect_train, color='#c3f38b', alpha=0.15, lw=1.5, ls=':')
    ax2.plot(epochs, rect_val, color='#f472b6', alpha=0.15, lw=1.5, ls=':')

    # Active curves
    ax2.plot(x_sub, y_rect_tr, color='#c3f38b', lw=2.8, label='Train Loss (Residual MSE / L1)',
             path_effects=[pe.SimpleLineShadow(shadow_color='#65a30d', offset=(0, 0), alpha=0.6), pe.Normal()])
    ax2.fill_between(x_sub, y_rect_tr, 0.18, color='#c3f38b', alpha=0.08)

    ax2.plot(x_sub, y_rect_vl, color='#f472b6', lw=2.4, ls='--', label='Val Loss (Holdout Validation)',
             path_effects=[pe.SimpleLineShadow(shadow_color='#db2777', offset=(0, 0), alpha=0.5), pe.Normal()])

    # Glowing Head Pointer
    ax2.scatter([cur_ep], [y_rect_tr[-1]], color='#c3f38b', s=80, zorder=5, edgecolors='#ffffff', linewidths=1.5)
    ax2.scatter([cur_ep], [y_rect_vl[-1]], color='#f472b6', s=70, zorder=5, edgecolors='#ffffff', linewidths=1.2)

    # Inset Loss Formula Box
    formula_box_rect = (
        r"$\mathbf{Loss\ Function:}$" "\n"
        r"$\mathcal{L}_{Recti} = \mathcal{L}_{SmoothL1}(\Delta\mathbf{d}) + \mathcal{L}_{SmoothL1}(\Delta v)$"
    )
    ax2.text(0.04, 0.96, formula_box_rect, transform=ax2.transAxes, va='top', ha='left',
             fontsize=8.8, color='#e2e8f0', bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#c3f38b', alpha=0.9))

    # Inset Performance Badge (Accuracy & Latency)
    perf_text_rect = (
        "ACCURACY & LATENCY:\n"
        "• Drift MAE: 1.23 m (-77.9% SIH)\n"
        "• Speed MAE: 9.24 km/h (-71.8%)\n"
        "• Latency:   14.8 ms WASM / 5.4 ms\n"
        "• Params:    92.4K (367.7 KB)"
    )
    ax2.text(0.96, 0.96, perf_text_rect, transform=ax2.transAxes, va='top', ha='right',
             fontsize=8.5, color='#f1f5f9', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#4d7c0f', edgecolor='#c3f38b', alpha=0.30))

    # Live Epoch Readout
    ax2.text(0.96, 0.06, f'Epoch: {cur_ep:4.1f}/20.0\nTrain: {y_rect_tr[-1]:.5f} | Val: {y_rect_vl[-1]:.5f}',
             transform=ax2.transAxes, va='bottom', ha='right', fontsize=9.2, color='#c3f38b', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

    ax2.legend(loc='lower left', frameon=True, facecolor='#1e293b', edgecolor='#334155', fontsize=8.8, labelcolor='#cbd5e1')


    # ==================== BOTTOM COMPARISON BAR ====================
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor('#0f172a')
    for spine in ax_bar.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.0)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    cols = [
        ("TARGET ARCHITECTURE", "TCN Speed Filter\n(3-Stage Dilated Conv1D)", "RectiPhyAI Transformer\n(2-Layer Self-Attention)", '#38bdf8', '#c3f38b'),
        ("INPUT STREAM & RATE", "2.0s Window @ 10Hz\n(20 x 6-DOF Normalized Tensor)", "1.0s Window @ 60Hz\n(60 x 6-DOF Gaussian-Filtered)", '#38bdf8', '#c3f38b'),
        ("ACCURACY BENCHMARKS", "Speed: 1.69 km/h MAE (r=0.94)\nZUPT: 93.5% Accuracy (Stop-Hold)", "Drift: 1.23m (-77.9% SIH Drift)\nSpeed: 9.24 km/h (-71.8% SIH)", '#38bdf8', '#c3f38b'),
        ("REAL-TIME LATENCY", "3.2 ms Edge Latency\n(WASM SIMD / Single Core)", "14.8 ms WASM / 5.4 ms WebGPU\n(High-Precision PDR Odometry)", '#38bdf8', '#c3f38b'),
    ]

    for c_i, (col_title, tcn_val_str, rect_val_str, col_tcn, col_rect) in enumerate(cols):
        cx = 0.03 + c_i * 0.245
        ax_bar.text(cx, 0.82, col_title, fontsize=9.2, fontweight='bold', color='#94a3b8', va='top')
        ax_bar.text(cx, 0.50, tcn_val_str, fontsize=8.6, color=col_tcn, va='center', fontfamily='sans-serif')
        ax_bar.text(cx, 0.16, rect_val_str, fontsize=8.6, color=col_rect, va='center', fontfamily='sans-serif')
        if c_i < 3:
            ax_bar.axvline(cx + 0.23, color='#1e293b', lw=1.2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img

if __name__ == '__main__':
    print('Rendering frames for GIF...')
    frames = []
    for i in range(dense_steps):
        img = render_frame(i)
        frames.append(img)
        if (i + 1) % 10 == 0 or i == dense_steps - 1:
            print(f'Rendered frame {i+1}/{dense_steps}')

    # Save static high-res image of final state
    final_img = frames[-1]
    static_png_path = os.path.join(ARTIFACT_DIR, 'tcn_vs_rectiphy_loss.png')
    final_img.save(static_png_path)
    print(f'Saved static PNG: {static_png_path}')

    # Add hold frames at the end (16 frames @ ~90ms = ~1.45s freeze)
    final_frame = frames[-1]
    for _ in range(16):
        frames.append(final_frame)

    # Save animated GIF
    gif_path_artifact = os.path.join(ARTIFACT_DIR, 'tcn_vs_rectiphy_loss.gif')
    gif_path_tcn = os.path.join(EXP_TCN_REPORTS, 'tcn_vs_rectiphy_loss.gif')
    gif_path_exp1 = os.path.join(EXP_1_REPORTS, 'tcn_vs_rectiphy_loss.gif')

    frames[0].save(
        gif_path_artifact,
        save_all=True,
        append_images=frames[1:],
        duration=90, # 90ms per frame
        loop=0,
        optimize=True
    )
    print(f'Saved GIF to artifact directory: {gif_path_artifact}')

    frames[0].save(
        gif_path_tcn,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True
    )

    frames[0].save(
        gif_path_exp1,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True
    )

    print('All GIF and PNG outputs created successfully!')
