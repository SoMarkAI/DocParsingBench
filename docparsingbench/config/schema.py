from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class ParagraphConfig:
    alpha: float = 0.5
    match_metric: str = "NED"
    formula_placeholder: str = "[FORMULA]"


@dataclass
class FormulaConfig:
    metric: str = "CDM"
    match_metric: str = "NED"


@dataclass
class TableConfig:
    structure_only: bool = False
    match_metric: str = "TEDS"


@dataclass
class DPBConfig:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    preset: Optional[str] = None


@dataclass
class PerfConfig:
    enable: bool = False


@dataclass
class ProgressConfig:
    enable: bool = False


@dataclass
class SummaryChartConfig:
    enable: bool = True
    output_path: str = "summary_chart.png"
    y_min: float = 30.0
    y_max: float = 100.0


@dataclass
class Config:
    chromedriver_path: Optional[str] = None
    visualize: bool = False
    drop_img: bool = True
    skip_chemical: bool = True
    paragraph: ParagraphConfig = field(default_factory=ParagraphConfig)
    formula: FormulaConfig = field(default_factory=FormulaConfig)
    table: TableConfig = field(default_factory=TableConfig)
    DPB: DPBConfig = field(default_factory=DPBConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    summary_chart: SummaryChartConfig = field(default_factory=SummaryChartConfig)


def _dict_get(d: Dict[str, Any], key: str, default: Any):
    v = d.get(key, default)
    return v if v is not None else default


def _build_paragraph(d: Dict[str, Any]) -> ParagraphConfig:
    return ParagraphConfig(
        alpha=float(_dict_get(d, "alpha", 0.5)),
        match_metric=str(_dict_get(d, "match_metric", "NED")),
        formula_placeholder=str(_dict_get(d, "formula_placeholder", "[FORMULA]")),
    )


def _build_formula(d: Dict[str, Any]) -> FormulaConfig:
    return FormulaConfig(
        metric=str(_dict_get(d, "metric", "CDM")),
        match_metric=str(_dict_get(d, "match_metric", "NED")),
    )


def _build_table(d: Dict[str, Any]) -> TableConfig:
    return TableConfig(
        structure_only=bool(_dict_get(d, "structure_only", False)),
        match_metric=str(_dict_get(d, "match_metric", "TEDS")),
    )


def _build_dpb(d: Dict[str, Any]) -> DPBConfig:
    return DPBConfig(
        alpha=float(_dict_get(d, "alpha", 0.5)),
        beta=float(_dict_get(d, "beta", 0.3)),
        gamma=float(_dict_get(d, "gamma", 0.2)),
        preset=_dict_get(d, "preset", None),
    )


def _build_perf(d: Dict[str, Any]) -> PerfConfig:
    return PerfConfig(
        enable=bool(_dict_get(d, "enable", False)),
    )


def _build_progress(d: Dict[str, Any]) -> ProgressConfig:
    return ProgressConfig(
        enable=bool(_dict_get(d, "enable", False)),
    )


def _build_summary_chart(d: Dict[str, Any]) -> SummaryChartConfig:
    return SummaryChartConfig(
        enable=bool(_dict_get(d, "enable", True)),
        output_path=str(_dict_get(d, "output_path", "summary_chart.png")),
        y_min=float(_dict_get(d, "y_min", 30.0)),
        y_max=float(_dict_get(d, "y_max", 100.0)),
    )


def load_config(path: str) -> Config:
    """Load YAML config into a Config object.
    - Maps all user config fields 1:1
    - DPB supports preset: if set, preset weights are applied first, then overridden by explicit alpha/beta/gamma
    - chromedriver_path defaults to None when not configured, letting fastcdm use its own default
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    chrom = raw.get("chromedriver_path")
    visualize = bool(_dict_get(raw, "visualize", False))
    drop_img = bool(_dict_get(raw, "drop_img", True))
    skip_chemical = bool(_dict_get(raw, "skip_chemical", True))
    paragraph = _build_paragraph(raw.get("paragraph", {}))
    formula = _build_formula(raw.get("formula", {}))
    table = _build_table(raw.get("table", {}))

    raw_dpb = raw.get("DPB", {})
    dpb = _build_dpb(raw_dpb)

    # Apply preset default weights, then override
    preset_map = {
        "general": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
        "paper": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
        "financial": {"alpha": 0.6, "beta": 0.2, "gamma": 0.2},
    }
    if dpb.preset and dpb.preset in preset_map:
        defaults = preset_map[dpb.preset]
        alpha = defaults["alpha"]
        beta = defaults["beta"]
        gamma = defaults["gamma"]
        # Override: only when the user explicitly provides the field
        if "alpha" in raw_dpb:
            alpha = float(raw_dpb["alpha"])  # type: ignore
        if "beta" in raw_dpb:
            beta = float(raw_dpb["beta"])   # type: ignore
        if "gamma" in raw_dpb:
            gamma = float(raw_dpb["gamma"]) # type: ignore
        dpb = DPBConfig(alpha=alpha, beta=beta, gamma=gamma, preset=dpb.preset)

    perf = _build_perf(raw.get("perf", {}))
    progress = _build_progress(raw.get("progress", {}))
    summary_chart = _build_summary_chart(raw.get("summary_chart", {}))
    return Config(
        chromedriver_path=chrom,  # None means fastcdm uses its own default
        visualize=visualize,
        drop_img=drop_img,
        skip_chemical=skip_chemical,
        paragraph=paragraph,
        formula=formula,
        table=table,
        DPB=dpb,
        perf=perf,
        progress=progress,
        summary_chart=summary_chart,
    )
