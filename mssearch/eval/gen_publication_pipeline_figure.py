"""
========================================================================================
PUBLICATION-GRADE METHOD PIPELINE FIGURE GENERATOR (CVSearch Fig 2 / HEE Fig 2 Style)
========================================================================================
Generates a modular 3-panel figure:
  - (a) Top: Workflow & 3-Tier Cascade (Root Canvas with Red Callout Zoom, Decision Diamonds,
             Early Exit Gates, Visual Evidence Pool, Final Answer)
  - (b) Bottom-Left: Stages 3 & 4 (SLIC / SGAP Partitioning, Saliency-Plateau Tree,
                     Micro-Fovea Token Density Boost, Cosine Feature Grouping)
  - (c) Bottom-Right: Stage 6 (Query-Adaptive Dual-Scale Context [Global + Crop],
                      Prompt-Directed Reading, Collision-Free Consensus Exit)

Accepts an assets directory containing real intermediate image crops saved from Kaggle,
or automatically synthesizes high-contrast mock assets if images are missing.
========================================================================================
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.gridspec as gridspec

# Standard publication font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.edgecolor'] = '#2B2D42'
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette: Ocean Dusk & Conference Standard
C_TEAL_DARK   = "#1B3B48"  # Primary containers
C_TEAL_MED    = "#2A9D8F"  # Stage 1 & 6 / Success
C_TEAL_LIGHT  = "#E8F5F3"  # Background fills
C_ORANGE_DARK = "#D35400"  # Target & Fovea callouts
C_ORANGE_MED  = "#E67E22"  # Stage 3 Partitioning
C_ORANGE_LIGHT= "#FDF2E9"  # Partition fill
C_GOLD        = "#E9C46A"  # Feature cache & SAM3
C_GOLD_LIGHT  = "#FEF9E7"  # Feature fill
C_CORAL_DARK  = "#C0392B"  # Pruned / Negative
C_CORAL_LIGHT = "#FDEDEC"  # Negative fill
C_BLUE_DARK   = "#2980B9"  # MLLM / Reading
C_BLUE_LIGHT  = "#EBF5FB"  # Reader fill
C_GRAY_DARK   = "#2C3E50"  # Text
C_GRAY_MED    = "#7F8C8D"  # Subtitles / Outlines
C_GRAY_LIGHT  = "#F8F9FA"  # Outer panel backgrounds
C_BORDER      = "#BDC3C7"  # Dashed box borders
C_WHITE       = "#FFFFFF"


def create_mock_canvas_assets(output_dir: str):
    """Generates realistic mock intermediate assets if real Kaggle images are not present."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Root Canvas (A stylish street/outdoor scene with a person wearing a jacket)
    w_root, h_root = 1200, 800
    img_root = Image.new("RGB", (w_root, h_root), color="#708090")
    draw_r = ImageDraw.Draw(img_root)
    # Sky gradient
    for y in range(400):
        r = int(135 + (200 - 135) * (y / 400))
        g = int(206 + (220 - 206) * (y / 400))
        b = int(235 + (240 - 235) * (y / 400))
        draw_r.line([(0, y), (w_root, y)], fill=(r, g, b))
    # Buildings in background
    draw_r.rectangle([100, 200, 350, 550], fill="#5B6770")
    draw_r.rectangle([400, 150, 750, 550], fill="#4A5568")
    draw_r.rectangle([800, 220, 1100, 550], fill="#606F7B")
    # Ground/Plaza
    draw_r.rectangle([0, 550, w_root, h_root], fill="#95A5A6")
    # Person standing on the right
    # Torso / Jacket
    draw_r.rectangle([880, 420, 1020, 680], fill="#1C2833")  # Dark jacket
    # Sleeve
    draw_r.rectangle([840, 440, 890, 620], fill="#273746")
    # Red logo on sleeve
    draw_r.ellipse([852, 490, 878, 520], fill="#E74C3C")
    # Text "epio MIX" above logo
    draw_r.rectangle([848, 475, 882, 485], fill="#FFFFFF")
    # Head
    draw_r.ellipse([920, 360, 980, 420], fill="#D35400")
    
    root_path = os.path.join(output_dir, "01_root_canvas.png")
    img_root.save(root_path)

    # 2. Target Callout (Zoomed crop on jacket sleeve with red logo and text)
    crop_box = (820, 450, 910, 540)
    img_crop = img_root.crop(crop_box).resize((300, 300), Image.Resampling.LANCZOS)
    draw_c = ImageDraw.Draw(img_crop)
    # Highlight text with bounding box
    draw_c.rectangle([40, 50, 260, 120], outline="#E74C3C", width=5)
    callout_path = os.path.join(output_dir, "02_target_callout.png")
    img_crop.save(callout_path)

    # 3. Stage 3 SLIC / SGAP Partitioning Visualization
    img_slic = img_root.copy().resize((600, 400), Image.Resampling.LANCZOS)
    draw_s = ImageDraw.Draw(img_slic)
    # Draw SLIC grid contours
    for x in range(0, 600, 75):
        draw_s.line([(x, 0), (x, 400)], fill="#F39C12", width=2)
    for y in range(0, 400, 50):
        draw_s.line([(0, y), (600, y)], fill="#F39C12", width=2)
    # Highlight the high-saliency patch containing target
    draw_s.rectangle([415, 200, 530, 350], outline="#27AE60", width=4)
    slic_path = os.path.join(output_dir, "03_stage3_partition_slic.png")
    img_slic.save(slic_path)

    # 4. Stage 4 Micro-Fovea Zoom (Tight 1.75x margin on sleeve)
    img_fovea = img_root.crop((835, 465, 895, 535)).resize((300, 300), Image.Resampling.LANCZOS)
    draw_f = ImageDraw.Draw(img_fovea)
    draw_f.rectangle([10, 10, 290, 290], outline="#2ECC71", width=6)
    fovea_path = os.path.join(output_dir, "04_stage4_micro_fovea.png")
    img_fovea.save(fovea_path)

    # 5. Stage 6 Global Context with Red Box
    img_ctx = img_root.copy().resize((400, 270), Image.Resampling.LANCZOS)
    draw_ctx = ImageDraw.Draw(img_ctx)
    draw_ctx.rectangle([280, 140, 340, 220], outline="#E74C3C", width=4)
    ctx_path = os.path.join(output_dir, "05_stage6_global_with_box.png")
    img_ctx.save(ctx_path)

    # 6. Stage 6 Focal High-Res Crop
    img_focal = img_fovea.copy()
    focal_path = os.path.join(output_dir, "06_stage6_focal_crop.png")
    img_focal.save(focal_path)

    print(f"[ASSETS] Mock visual assets generated in '{output_dir}'.")


def draw_diamond(ax, cx, cy, w, h, text="", fill_color="#FFFFFF", border_color="#2B2D42", text_color="#2B2D42", fontsize=9):
    """Draws a flowchart decision diamond."""
    pts = [
        [cx, cy + h / 2.0],
        [cx + w / 2.0, cy],
        [cx, cy - h / 2.0],
        [cx - w / 2.0, cy]
    ]
    poly = mpatches.Polygon(pts, closed=True, facecolor=fill_color, edgecolor=border_color, linewidth=1.5, zorder=5)
    ax.add_patch(poly)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=6)


def draw_card(ax, x, y, w, h, title="", subtitle="", fill_color="#FFFFFF", border_color="#BDC3C7", corner_radius=0.015, zorder=3):
    """Draws a rounded card with optional title."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                         facecolor=fill_color, edgecolor=border_color,
                         linewidth=1.2, zorder=zorder)
    ax.add_patch(box)
    if title:
        ax.text(x + w / 2.0, y + h - 0.025, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=C_TEAL_DARK, zorder=zorder + 1)
    if subtitle:
        ax.text(x + w / 2.0, y + h - 0.052, subtitle, ha="center", va="top",
                fontsize=7.5, color=C_GRAY_MED, zorder=zorder + 1)
    return box


def draw_icon_box(ax, cx, cy, w, h, label, sublabel="", tag="[VLM]", fill="#EBF5FB", border="#2980B9"):
    """Draws an agent/model entity box with a styled tag."""
    box = FancyBboxPatch((cx - w/2.0, cy - h/2.0), w, h,
                         boxstyle="round,pad=0,rounding_size=0.012",
                         facecolor=fill, edgecolor=border, linewidth=1.4, zorder=5)
    ax.add_patch(box)
    full_text = f"{tag} {label}" if tag else label
    ax.text(cx, cy + 0.016, full_text, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_TEAL_DARK, zorder=6)
    if sublabel:
        ax.text(cx, cy - 0.016, sublabel, ha="center", va="center",
                fontsize=7.0, color=C_GRAY_MED, zorder=6)


def build_pipeline_figure(assets_dir: str, output_pdf: str, output_png: str):
    """Assembles the full publication-quality Figure 2."""
    # Ensure assets exist
    required_files = [
        "01_root_canvas.png", "02_target_callout.png",
        "03_stage3_partition_slic.png", "04_stage4_micro_fovea.png",
        "05_stage6_global_with_box.png", "06_stage6_focal_crop.png"
    ]
    has_all = all(os.path.exists(os.path.join(assets_dir, f)) for f in required_files)
    if not has_all:
        print("[FIGURE BUILDER] Real assets missing in directory. Generating mock visual assets...")
        create_mock_canvas_assets(assets_dir)

    # Load image assets
    img_root    = Image.open(os.path.join(assets_dir, "01_root_canvas.png")).convert("RGB")
    img_callout = Image.open(os.path.join(assets_dir, "02_target_callout.png")).convert("RGB")
    img_slic    = Image.open(os.path.join(assets_dir, "03_stage3_partition_slic.png")).convert("RGB")
    img_fovea   = Image.open(os.path.join(assets_dir, "04_stage4_micro_fovea.png")).convert("RGB")
    img_ctx     = Image.open(os.path.join(assets_dir, "05_stage6_global_with_box.png")).convert("RGB")
    img_focal   = Image.open(os.path.join(assets_dir, "06_stage6_focal_crop.png")).convert("RGB")

    # Initialize Figure
    fig = plt.figure(figsize=(16.0, 9.2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # =========================================================================
    # OUTER CONTAINERS: 3 MODULAR PANELS (a), (b), (c)
    # =========================================================================
    # Panel (a): Top row full width
    draw_card(ax, 0.015, 0.515, 0.970, 0.465, fill_color="#FAFCFD", border_color="#95A5A6", corner_radius=0.012, zorder=1)
    ax.text(0.025, 0.958, "(a) End-to-End Workflow & 3-Tier Execution Cascade",
            fontsize=12, fontweight="bold", color=C_TEAL_DARK, zorder=4)

    # Panel (b): Bottom-Left (Stages 3 & 4)
    draw_card(ax, 0.015, 0.020, 0.585, 0.475, fill_color="#FAFCFD", border_color="#95A5A6", corner_radius=0.012, zorder=1)
    ax.text(0.025, 0.475, "(b) Saliency-Plateau Partitioning & Feature Grouping (Stages 3 & 4)",
            fontsize=11.5, fontweight="bold", color=C_TEAL_DARK, zorder=4)

    # Panel (c): Bottom-Right (Stage 6)
    draw_card(ax, 0.615, 0.020, 0.370, 0.475, fill_color="#FAFCFD", border_color="#95A5A6", corner_radius=0.012, zorder=1)
    ax.text(0.625, 0.475, "(c) Dual-Scale Reading & Consensus (Stage 6)",
            fontsize=11.5, fontweight="bold", color=C_TEAL_DARK, zorder=4)

    # =========================================================================
    # PANEL (a): TOP ROW WORKFLOW
    # =========================================================================
    # 1. Input Image Box with Question Banner
    x_in, y_in, w_in, h_in = 0.030, 0.545, 0.170, 0.370
    draw_card(ax, x_in, y_in, w_in, h_in, fill_color=C_WHITE, border_color=C_TEAL_MED, zorder=2)
    # Query Banner on top of input box
    q_box = FancyBboxPatch((x_in + 0.005, y_in + h_in - 0.062), w_in - 0.010, 0.055,
                           boxstyle="round,pad=0,rounding_size=0.006",
                           facecolor="#E8F8F5", edgecolor=C_TEAL_MED, linewidth=1.0, zorder=4)
    ax.add_patch(q_box)
    ax.text(x_in + w_in/2.0, y_in + h_in - 0.022, "Query $q$:",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_TEAL_DARK, zorder=5)
    ax.text(x_in + w_in/2.0, y_in + h_in - 0.046, "\"What is the text on the sleeve?\"",
            ha="center", va="center", fontsize=7.2, fontstyle="italic", color="#16A085", zorder=5)

    # Inset Root Image Thumbnail
    ax_root = fig.add_axes([x_in + 0.010, y_in + 0.065, 0.090, 0.225])
    ax_root.imshow(img_root)
    ax_root.axis("off")
    # Red circle on target region in root image
    circ_root = Circle((910, 490), radius=60, edgecolor="#E74C3C", facecolor="none", linewidth=2.0)
    ax_root.add_patch(circ_root)

    # Inset Callout Circle (Zoomed target)
    ax_callout = fig.add_axes([x_in + 0.105, y_in + 0.140, 0.060, 0.135])
    ax_callout.imshow(img_callout)
    ax_callout.axis("off")
    for spine in ax_callout.spines.values():
        spine.set_edgecolor("#E74C3C")
        spine.set_linewidth(2.0)

    # Target indicator text
    ax.text(x_in + w_in/2.0, y_in + 0.028, r"Ultra-High Resolution Canvas ($N \geq 1$)",
            ha="center", va="center", fontsize=7.2, color=C_GRAY_MED, zorder=4)

    # Leader dashed lines connecting root circle to callout
    con1 = mpatches.ConnectionPatch(xyA=(950, 460), xyB=(0, 150), coordsA="data", coordsB="data",
                                    axesA=ax_root, axesB=ax_callout, color="#E74C3C", linestyle="--", linewidth=1.2)
    fig.add_artist(con1)
    con2 = mpatches.ConnectionPatch(xyA=(950, 520), xyB=(0, 0), coordsA="data", coordsB="data",
                                    axesA=ax_root, axesB=ax_callout, color="#E74C3C", linestyle="--", linewidth=1.2)
    fig.add_artist(con2)

    # Arrow from Input -> Phase 1
    y_flow = 0.700
    ax.annotate("", xy=(0.235, y_flow), xytext=(0.203, y_flow),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL_DARK, lw=2.0, mutation_scale=15), zorder=5)

    # 2. Phase 1: Direct Global Assessment
    draw_icon_box(ax, 0.285, y_flow, 0.095, 0.065, "Global Assess", r"Root View ($\leq 1024\mathrm{px}$)", "[VLM]", fill="#F4F6F6", border="#7F8C8D")
    
    # Diamond 1: Sufficiency check
    draw_diamond(ax, 0.370, y_flow, 0.055, 0.060, "$c_q > 0.60$ ?", fill_color="#FFFFFF", border_color=C_TEAL_DARK, fontsize=7.5)

    # Arrow from Global Assess -> Diamond 1
    ax.annotate("", xy=(0.342, y_flow), xytext=(0.333, y_flow),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL_DARK, lw=1.8, mutation_scale=12), zorder=5)

    # Phase 1: YES Exit (Upward to Direct Answer)
    ax.annotate("", xy=(0.370, 0.815), xytext=(0.370, y_flow + 0.031),
                arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=2.0, mutation_scale=14), zorder=5)
    ax.text(0.380, 0.770, "Yes (0 Tools)", fontsize=7.2, fontweight="bold", color="#27AE60", zorder=6)
    draw_icon_box(ax, 0.370, 0.845, 0.082, 0.045, "Direct Answer", "1 MLLM Step", "[FAST]", fill="#E8F8F5", border="#27AE60")

    # Phase 1: NO Branch (Rightward to Phase 2 Fast Path)
    ax.annotate("", xy=(0.435, y_flow), xytext=(0.398, y_flow),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=2.0, mutation_scale=14), zorder=5)
    ax.text(0.408, y_flow + 0.012, "No", fontsize=7.5, fontweight="bold", color="#E74C3C", zorder=6)

    # 3. Phase 2: Tool-Based Fast Path (SAM3 Detection + Feature Cache)
    draw_icon_box(ax, 0.485, y_flow, 0.095, 0.065, "SAM3 Fast Path", r"Extracts $\mathbf{H}_e$ Cache", "[SAM3]", fill=C_GOLD_LIGHT, border=C_GOLD)
    
    # Diamond 2: Fast Path check
    draw_diamond(ax, 0.570, y_flow, 0.055, 0.060, "$c_e > 0.70$ ?", fill_color="#FFFFFF", border_color=C_TEAL_DARK, fontsize=7.5)

    # Arrow from SAM3 Fast Path -> Diamond 2
    ax.annotate("", xy=(0.542, y_flow), xytext=(0.533, y_flow),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL_DARK, lw=1.8, mutation_scale=12), zorder=5)

    # Phase 2: YES Exit (Upward to Fast Path Answer)
    ax.annotate("", xy=(0.570, 0.815), xytext=(0.570, y_flow + 0.031),
                arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=2.0, mutation_scale=14), zorder=5)
    ax.text(0.580, 0.770, "Yes (1 Step)", fontsize=7.2, fontweight="bold", color="#27AE60", zorder=6)
    draw_icon_box(ax, 0.570, 0.845, 0.082, 0.045, "Fast Answer", r"$C_{\mathrm{union}}$ Crop", "[EXIT]", fill="#E8F8F5", border="#27AE60")

    # Phase 2: NO Branch (Fallback to Deep Search)
    ax.annotate("", xy=(0.635, y_flow), xytext=(0.598, y_flow),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=2.0, mutation_scale=14), zorder=5)
    ax.text(0.608, y_flow + 0.012, "No", fontsize=7.5, fontweight="bold", color="#E74C3C", zorder=6)

    # 4. Tier 3 Deep Visual Search Container (Workflow summary)
    x_t3, y_t3, w_t3, h_t3 = 0.638, 0.565, 0.205, 0.295
    draw_card(ax, x_t3, y_t3, w_t3, h_t3, title="Tier 3: Multi-Scale Search",
              subtitle=r"Reuses Cached Features $\mathbf{H}_e$", fill_color="#F4F6F7", border_color=C_TEAL_MED, zorder=2)
    
    # Internal stages chip
    stg_chip = FancyBboxPatch((x_t3 + 0.012, y_t3 + 0.100), w_t3 - 0.024, 0.115,
                              boxstyle="round,pad=0,rounding_size=0.008",
                              facecolor=C_WHITE, edgecolor="#BDC3C7", linewidth=1.0, zorder=4)
    ax.add_patch(stg_chip)
    ax.text(x_t3 + w_t3/2.0, y_t3 + 0.180, r"Stage 3: Saliency Plateau ($\Delta S \leq 0.04$)",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_TEAL_DARK, zorder=5)
    ax.text(x_t3 + w_t3/2.0, y_t3 + 0.150, "Stage 4: Micro-Fovea Zoom & Latent Probing",
            ha="center", va="center", fontsize=7.0, color=C_ORANGE_DARK, zorder=5)
    ax.text(x_t3 + w_t3/2.0, y_t3 + 0.120, r"$\longrightarrow$ Candidate Pool $\mathcal{P} \leq 16$",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#16A085", zorder=5)

    # 5. Output: MLLM Reader & Final Consensus Answer
    ax.annotate("", xy=(0.865, y_flow), xytext=(x_t3 + w_t3, y_flow),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL_DARK, lw=2.0, mutation_scale=14), zorder=5)
    
    draw_icon_box(ax, 0.895, y_flow, 0.060, 0.065, "Reader", "Stage 6", "[MLLM]", fill=C_BLUE_LIGHT, border=C_BLUE_DARK)

    # Final Answer Speech Bubble
    ax.annotate("", xy=(0.932, y_flow), xytext=(0.925, y_flow),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL_DARK, lw=1.5, mutation_scale=10), zorder=5)
    ans_box = FancyBboxPatch((0.933, y_flow - 0.038), 0.045, 0.076,
                            boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor="#D5F5E3", edgecolor="#27AE60", linewidth=1.5, zorder=5)
    ax.add_patch(ans_box)
    ax.text(0.955, y_flow + 0.012, "\"epio MIX\"", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1E8449", zorder=6)
    ax.text(0.955, y_flow - 0.014, r"Option 'A' $\checkmark$", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#27AE60", zorder=6)

    # 6. Workflow Legend (Top-Right inside Panel a)
    x_leg, y_leg, w_leg, h_leg = 0.755, 0.895, 0.220, 0.072
    leg_box = FancyBboxPatch((x_leg, y_leg), w_leg, h_leg,
                             boxstyle="round,pad=0,rounding_size=0.006",
                             facecolor=C_WHITE, edgecolor="#BDC3C7", linewidth=0.8, zorder=3)
    ax.add_patch(leg_box)
    ax.text(x_leg + 0.010, y_leg + 0.054, "Legend:", fontsize=7.5, fontweight="bold", color=C_GRAY_DARK, zorder=4)
    # Legend items
    ax.text(x_leg + 0.015, y_leg + 0.032, r"$\checkmark$ Success / Kept", fontsize=6.8, color="#27AE60", zorder=4)
    ax.text(x_leg + 0.015, y_leg + 0.012, r"$\times$ Pruned / Excluded", fontsize=6.8, color="#C0392B", zorder=4)
    ax.text(x_leg + 0.085, y_leg + 0.032, "[SAM3] Detection", fontsize=6.8, color=C_ORANGE_DARK, zorder=4)
    ax.text(x_leg + 0.085, y_leg + 0.012, "[MLLM] Reader", fontsize=6.8, color=C_BLUE_DARK, zorder=4)
    ax.text(x_leg + 0.155, y_leg + 0.032, "[FAST] Early Exit", fontsize=6.8, color="#D35400", zorder=4)
    ax.text(x_leg + 0.155, y_leg + 0.012, r"$\mathbf{H}_e$ Cache", fontsize=6.8, color="#B7950B", zorder=4)

    # =========================================================================
    # PANEL (b): BOTTOM-LEFT (STAGES 3 & 4)
    # =========================================================================
    # Sub-Box 1: Stage 3 SLIC / SGAP & Spatial Entropy
    x_b1, y_b1, w_b1, h_b1 = 0.030, 0.040, 0.260, 0.410
    draw_card(ax, x_b1, y_b1, w_b1, h_b1, fill_color=C_WHITE, border_color="#BDC3C7", zorder=2)
    ax.text(x_b1 + w_b1/2.0, y_b1 + h_b1 - 0.018, "Stage 3: Entropy-Gated Partitioning",
            ha="center", va="top", fontsize=9.0, fontweight="bold", color=C_TEAL_DARK, zorder=3)
    ax.text(x_b1 + w_b1/2.0, y_b1 + h_b1 - 0.038, r"Reuses Cached Backbone Features $\mathbf{H}_e$",
            ha="center", va="top", fontsize=7.2, color=C_GRAY_MED, zorder=3)
    
    # Title badge above SLIC
    ax.text(x_b1 + w_b1/2.0, y_b1 + 0.345, r"SAM3-SGAP Partitioning ($k^* \in [8, 16]$)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_ORANGE_DARK, zorder=5)

    # Inset SLIC Partition Image
    ax_slic = fig.add_axes([x_b1 + 0.020, y_b1 + 0.145, 0.220, 0.180])
    ax_slic.imshow(img_slic)
    ax_slic.axis("off")

    # Partitioning rule pills
    r1 = FancyBboxPatch((x_b1 + 0.010, y_b1 + 0.075), w_b1 - 0.020, 0.055,
                        boxstyle="round,pad=0,rounding_size=0.005",
                        facecolor="#FDF2E9", edgecolor=C_ORANGE_MED, linewidth=0.8, zorder=3)
    ax.add_patch(r1)
    ax.text(x_b1 + w_b1/2.0, y_b1 + 0.108, r"Spatial Entropy: $H = 0.35\tilde{H} + 0.35\tilde{G} + 0.30\tilde{V}$",
            ha="center", va="center", fontsize=7.2, color=C_GRAY_DARK, zorder=4)
    ax.text(x_b1 + w_b1/2.0, y_b1 + 0.088, r"$H < 0.35 \rightarrow 2\times 2$ Quadrants; $H \geq 0.35 \rightarrow$ SAM3-SGAP",
            ha="center", va="center", fontsize=6.8, color="#BA4A00", zorder=4)

    # Dynamic Saliency-Plateau Self-Termination Badge
    r2 = FancyBboxPatch((x_b1 + 0.010, y_b1 + 0.012), w_b1 - 0.020, 0.055,
                        boxstyle="round,pad=0,rounding_size=0.005",
                        facecolor="#E8F8F5", edgecolor="#27AE60", linewidth=1.2, zorder=3)
    ax.add_patch(r2)
    ax.text(x_b1 + w_b1/2.0, y_b1 + 0.045, "Dynamic Saliency-Plateau Stopping:",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1E8449", zorder=4)
    ax.text(x_b1 + w_b1/2.0, y_b1 + 0.025, r"$\Delta S = S(c) - S(p) \leq 0.04 \Rightarrow$ Halt Zoom (Zero Fragmentation)",
            ha="center", va="center", fontsize=6.8, color="#196F3D", zorder=4)

    # Arrow from Stage 3 -> Stage 4
    ax.annotate("", xy=(0.320, 0.245), xytext=(0.293, 0.245),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL_DARK, lw=2.0, mutation_scale=14), zorder=5)

    # Sub-Box 2: Stage 4 Micro-Fovea Zoom & Latent Probing
    x_b2, y_b2, w_b2, h_b2 = 0.325, 0.040, 0.260, 0.410
    draw_card(ax, x_b2, y_b2, w_b2, h_b2, fill_color=C_WHITE, border_color="#BDC3C7", zorder=2)
    ax.text(x_b2 + w_b2/2.0, y_b2 + h_b2 - 0.018, "Stage 4: Micro-Fovea & Latent Probing",
            ha="center", va="top", fontsize=9.0, fontweight="bold", color=C_TEAL_DARK, zorder=3)
    ax.text(x_b2 + w_b2/2.0, y_b2 + h_b2 - 0.038, "Zero-Pass Feature Clustering & Dual Gating",
            ha="center", va="top", fontsize=7.2, color=C_GRAY_MED, zorder=3)

    # Title badge above Fovea
    ax.text(x_b2 + w_b2/2.0, y_b2 + 0.345, r"Micro-Fovea Zoom ($1.75\times$ Margin)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="#27AE60", zorder=5)

    # Inset Micro-Fovea Zoom Image
    ax_fovea = fig.add_axes([x_b2 + 0.045, y_b2 + 0.160, 0.170, 0.165])
    ax_fovea.imshow(img_fovea)
    ax_fovea.axis("off")

    # Token density comparison badge
    f_badge = FancyBboxPatch((x_b2 + 0.010, y_b2 + 0.088), w_b2 - 0.020, 0.060,
                             boxstyle="round,pad=0,rounding_size=0.006",
                             facecolor="#FEF9E7", edgecolor="#F1C40F", linewidth=1.0, zorder=3)
    ax.add_patch(f_badge)
    ax.text(x_b2 + w_b2/2.0, y_b2 + 0.128, "Visual Token Density Optimization:",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#7D6608", zorder=4)
    ax.text(x_b2 + w_b2/2.0, y_b2 + 0.104, r"Root View: $0.08\%$ $\rightarrow$ Micro-Fovea: $\sim 40\%$ Tokens",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#B7950B", zorder=4)

    # Latent Budget & Safeguard rules
    f_rules = FancyBboxPatch((x_b2 + 0.010, y_b2 + 0.012), w_b2 - 0.020, 0.068,
                             boxstyle="round,pad=0,rounding_size=0.006",
                             facecolor="#F4F6F6", edgecolor="#BDC3C7", linewidth=0.8, zorder=3)
    ax.add_patch(f_rules)
    ax.text(x_b2 + w_b2/2.0, y_b2 + 0.060, r"Cosine Grouping ($\cos \geq 0.80$) + Dual Safeguards",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=C_TEAL_DARK, zorder=4)
    ax.text(x_b2 + w_b2/2.0, y_b2 + 0.040, r"Question-Aware Latent Probing ($B_{\mathrm{latent}} \leq 8$)",
            ha="center", va="center", fontsize=6.8, color="#16A085", zorder=4)
    ax.text(x_b2 + w_b2/2.0, y_b2 + 0.022, r"Direct Pass-Through to Candidate Pool $\mathcal{P}$",
            ha="center", va="center", fontsize=6.8, color="#27AE60", zorder=4)

    # =========================================================================
    # PANEL (c): BOTTOM-RIGHT (STAGE 6)
    # =========================================================================
    x_c, y_c, w_c, h_c = 0.625, 0.040, 0.350, 0.410
    draw_card(ax, x_c, y_c, w_c, h_c, fill_color=C_WHITE, border_color=C_BLUE_DARK, zorder=2)
    ax.text(x_c + w_c/2.0, y_c + h_c - 0.018, "Stage 6: Query-Adaptive Dual-Scale Reading",
            ha="center", va="top", fontsize=9.0, fontweight="bold", color=C_TEAL_DARK, zorder=3)
    ax.text(x_c + w_c/2.0, y_c + h_c - 0.038, "Collision-Free Consensus & Answer Extraction",
            ha="center", va="top", fontsize=7.2, color=C_GRAY_MED, zorder=3)

    # 1. Relational intent trigger badge
    rel_box = FancyBboxPatch((x_c + 0.015, y_c + 0.315), w_c - 0.030, 0.040,
                             boxstyle="round,pad=0,rounding_size=0.005",
                             facecolor="#EBF5FB", edgecolor=C_BLUE_DARK, linewidth=1.0, zorder=3)
    ax.add_patch(rel_box)
    ax.text(x_c + w_c/2.0, y_c + 0.335, "Relational Trigger: is_relational = True (\"above\", \"on sleeve\")",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=C_BLUE_DARK, zorder=4)

    # 2. Dual-Scale Visual Prompting Pair
    # Left: Global with Box
    ax_ctx = fig.add_axes([x_c + 0.025, y_c + 0.135, 0.140, 0.155])
    ax_ctx.imshow(img_ctx)
    ax_ctx.axis("off")
    ax.text(x_c + 0.095, y_c + 0.120, "[1] Global Context (Red Box)",
            ha="center", va="center", fontsize=6.8, color=C_GRAY_MED, zorder=4)

    # Plus sign
    ax.text(x_c + 0.175, y_c + 0.210, "+", ha="center", va="center", fontsize=14, fontweight="bold", color=C_TEAL_DARK, zorder=4)

    # Right: Focal Crop
    ax_focal = fig.add_axes([x_c + 0.185, y_c + 0.135, 0.140, 0.155])
    ax_focal.imshow(img_focal)
    ax_focal.axis("off")
    ax.text(x_c + 0.255, y_c + 0.120, r"[2] Focal Crop ($1.25\times$ Padding)",
            ha="center", va="center", fontsize=6.8, color=C_GRAY_MED, zorder=4)

    # 3. Collision-Free Early Consensus & Deduplication
    cons_box = FancyBboxPatch((x_c + 0.015, y_c + 0.012), w_c - 0.030, 0.090,
                              boxstyle="round,pad=0,rounding_size=0.006",
                              facecolor="#E8F8F5", edgecolor="#27AE60", linewidth=1.2, zorder=3)
    ax.add_patch(cons_box)
    ax.text(x_c + w_c/2.0, y_c + 0.082, "Collision-Free Early Consensus Exit:",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1E8449", zorder=4)
    ax.text(x_c + w_c/2.0, y_c + 0.060, r"Extracts Final Answer: <concise>; $\mathrm{sim} \geq 0.75 \Rightarrow$ Early Exit",
            ha="center", va="center", fontsize=6.8, color="#196F3D", zorder=4)
    ax.text(x_c + w_c/2.0, y_c + 0.035, r'Deduplicated Prediction: "epio MIX" (Matches Option A $\checkmark$)',
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#27AE60", zorder=4)

    # Save outputs
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[FIGURE GENERATED] Successfully saved:\n  - Vector PDF: {output_pdf}\n  - High-Res PNG: {output_png}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "pipeline_assets", "sample_40")
    pdf_out = os.path.join(base_dir, "fig_pipeline_overview.pdf")
    png_out = os.path.join(base_dir, "fig_pipeline_overview.png")

    if len(sys.argv) > 1:
        assets_dir = sys.argv[1]
    if len(sys.argv) > 2:
        pdf_out = sys.argv[2]
    if len(sys.argv) > 3:
        png_out = sys.argv[3]

    build_pipeline_figure(assets_dir, pdf_out, png_out)
