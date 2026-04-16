from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from docparsingbench.visualize.data_aggregator import (
    AggregatedResult,
    ModelScores,
    aggregate,
)

BAR_PALETTE = [
    "#F4A261",
    "#C75A00",
    "#E63946",
    "#E9C46A",
    "#2A9D8F",
    "#264653",
    "#457B9D",
    "#1D3557",
    "#A8DADC",
    "#8338EC",
    "#3A0CA3",
]
FIGURE_FACE_COLOR = "#F8F9FA"
SPINE_COLOR = "#333333"
GRID_COLOR = "#CCCCCC"
FONT_SCALE = 1.2

TITLE_SIZE = 14 * FONT_SCALE
HEADER_SIZE = 24 * FONT_SCALE
AXIS_LABEL_SIZE = 12 * FONT_SCALE
TICK_SIZE = 10 * FONT_SCALE
LEGEND_SIZE = 12 * FONT_SCALE
VALUE_SIZE = 10 * FONT_SCALE
DEFAULT_Y_LIMITS: Tuple[float, float] = (30.0, 100.0)


def _configure_global_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["font.size"] = 10 * FONT_SCALE


def _build_model_styles(models: List[ModelScores]) -> List[Dict[str, str]]:
    styles: List[Dict[str, str]] = []
    for idx, _model in enumerate(models):
        facecolor = BAR_PALETTE[idx % len(BAR_PALETTE)]
        styles.append(
            {
                "facecolor": facecolor,
                "edgecolor": "black",
                "linewidth": 0.5,
            }
        )
    return styles


def _add_logo(fig: plt.Figure) -> None:
    logo_path = Path(__file__).resolve().parents[2] / "assets" / "LOGO.png"
    if not logo_path.exists():
        return
    try:
        logo_image = plt.imread(str(logo_path))
    except OSError:
        return
    logo_ax = fig.add_axes([0.02, 0.914, 0.052, 0.062], zorder=30)
    logo_ax.imshow(logo_image)
    logo_ax.set_axis_off()


def _draw_header(fig: plt.Figure) -> None:
    _add_logo(fig)
    fig.text(
        0.082,
        0.95,
        "DocParsingBench Summary",
        fontsize=HEADER_SIZE,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="center",
    )


def _apply_axis_style(ax: Axes) -> None:
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)


def _draw_bars(
    ax: Axes,
    models: List[ModelScores],
    metric: str,
    title: str,
    styles: List[Dict[str, str]],
    *,
    y_label: Optional[str],
    y_limits: Tuple[float, float],
    show_main_x_labels: bool,
) -> None:
    values = [getattr(model, metric) * 100 for model in models]
    x_positions = list(range(len(models)))

    bars = ax.bar(
        x_positions,
        values,
        width=0.7,
        color=[s["facecolor"] for s in styles],
        edgecolor=[s["edgecolor"] for s in styles],
        linewidth=[s["linewidth"] for s in styles],
        zorder=3,
    )

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", color="#111111", pad=18)
    ax.set_ylim(*y_limits)
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, color="#111111")
    else:
        ax.set_ylabel("")

    if show_main_x_labels:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [model.display_name for model in models],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=TICK_SIZE,
            color="#111111",
        )
        ax.tick_params(axis="x", length=0, pad=6)
    else:
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)

    ax.tick_params(axis="y", labelsize=TICK_SIZE, colors="#111111")

    y_min, y_max = y_limits
    for x, value in zip(x_positions, values):
        anchor_y = max(value, y_min)
        text_y = min(anchor_y + 0.8, y_max - 0.9)
        ax.text(
            x,
            text_y,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
            zorder=8,
        )


def _draw_legend_panel(ax: Axes, models: List[ModelScores], styles: List[Dict[str, str]]) -> None:
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles: List[Patch] = []
    labels: List[str] = []
    for model, style in zip(models, styles):
        handles.append(
            Patch(
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
            )
        )
        labels.append(model.display_name)

    legend = ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        edgecolor="#CCCCCC",
        fontsize=LEGEND_SIZE,
        markerscale=2.0,
        bbox_to_anchor=(0.44, 0.5),
        bbox_transform=ax.transAxes,
        handlelength=2.2,
        handletextpad=1.0,
        labelspacing=1.05,
        borderpad=1.3,
    )
    legend.get_frame().set_facecolor("white")


def generate_summary_chart(
    labels_path: Path,
    results_dir: Path,
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (20, 12),
    exclude_model_prefixes: Optional[List[str]] = None,
    y_limits: Tuple[float, float] = DEFAULT_Y_LIMITS,
) -> Path:
    """Generate a summary bar chart and save as PNG. Returns output path."""
    agg = aggregate(
        labels_path,
        results_dir,
        exclude_model_prefixes=exclude_model_prefixes,
    )
    return generate_summary_chart_from_agg(
        agg,
        output_path,
        dpi=dpi,
        figsize=figsize,
        y_limits=y_limits,
    )


def generate_summary_chart_from_agg(
    agg: AggregatedResult,
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (20, 12),
    y_limits: Tuple[float, float] = DEFAULT_Y_LIMITS,
) -> Path:
    """Generate chart from a pre-computed AggregatedResult."""
    _configure_global_style()

    models = agg.all  # already sorted by DPB descending
    styles = _build_model_styles(models)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=FIGURE_FACE_COLOR)
    gs = GridSpec(
        3,
        5,
        figure=fig,
        hspace=0.78,
        wspace=0.34,
        height_ratios=[1.0, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.15],
    )

    # rows 0-1, cols 0-3
    ax_main = fig.add_subplot(gs[0:2, 0:4])
    # rows 0-1, col 4
    ax_legend = fig.add_subplot(gs[0:2, 4])

    # Bottom row keeps 3 aligned charts with shared y-axis.
    bottom_gs = gs[2, 0:5].subgridspec(1, 3, wspace=0.2)
    ax_text = fig.add_subplot(bottom_gs[0, 0])
    ax_formula = fig.add_subplot(bottom_gs[0, 1], sharey=ax_text)
    ax_table = fig.add_subplot(bottom_gs[0, 2], sharey=ax_text)

    _apply_axis_style(ax_main)
    _apply_axis_style(ax_text)
    _apply_axis_style(ax_formula)
    _apply_axis_style(ax_table)

    _draw_bars(
        ax_main,
        models,
        "dpb",
        "DPB (Overall)",
        styles,
        y_label="Score (%)",
        y_limits=y_limits,
        show_main_x_labels=True,
    )

    _draw_bars(
        ax_text,
        models,
        "text",
        "Text",
        styles,
        y_label="Score (%)",
        y_limits=y_limits,
        show_main_x_labels=False,
    )
    _draw_bars(
        ax_formula,
        models,
        "formula",
        "Formula",
        styles,
        y_label=None,
        y_limits=y_limits,
        show_main_x_labels=False,
    )
    _draw_bars(
        ax_table,
        models,
        "table",
        "Table",
        styles,
        y_label=None,
        y_limits=y_limits,
        show_main_x_labels=False,
    )

    # Bottom three charts: keep y-axis numbers, remove x-axis labels.
    for axis in (ax_text, ax_formula, ax_table):
        axis.tick_params(axis="y", labelleft=True)
        axis.set_xticks([])

    _draw_legend_panel(ax_legend, models, styles)
    _draw_header(fig)

    fig.subplots_adjust(top=0.84, left=0.065, right=0.98, bottom=0.08)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)
    return output_path
