"""Generate a standalone, shareable HTML leaderboard.

Produces a single self-contained ``.html`` file that renders the DPB
leaderboard without any server, using Google Fonts + html2canvas from
CDN. Open it in any browser to view; share the file directly.

Aesthetic direction: dark technical product showcase. The page preserves the same
client-side leaderboard behavior while using darker layers, gradient
surfaces, and more expressive orange-led hierarchy.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from docparsingbench.visualize.data_aggregator import (
    AggregatedResult,
    ModelScores,
    _compute_best,
    aggregate,
)

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
_LOGO_PATH = _ASSETS_DIR / "LOGO.png"


def _logo_data_url() -> str:
    if not _LOGO_PATH.exists():
        return ""
    b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _model_to_dict(m: ModelScores) -> dict:
    return {
        "model_name": m.model_name,
        "display_name": m.display_name,
        "dpb": m.dpb,
        "text": m.text,
        "formula": m.formula,
        "table": m.table,
    }


def _build_payload(agg: AggregatedResult) -> dict:
    views = {"all": [_model_to_dict(m) for m in agg.all]}
    bests = {"all": _compute_best(agg.all)}
    for ind in agg.industries:
        models = agg.by_industry.get(ind, [])
        views[ind] = [_model_to_dict(m) for m in models]
        bests[ind] = _compute_best(models)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "industries": ["all"] + list(agg.industries),
        "views": views,
        "bests": bests,
        "n_methods": len(agg.all),
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocParsingBench · Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700;800&display=swap" rel="stylesheet">
<script src="https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<style>
:root {
    --sm-primary: #FF8C00;
    --sm-primary-hover: #e67e00;
    --sm-primary-bg: rgba(255, 140, 0, 0.10);
    --sm-secondary: #FA5151;
    --sm-bg: #000000;
    --sm-surface-1: #101112;
    --sm-surface-2: #151517;
    --sm-surface-3: #1C1E1F;
    --sm-text-100: #FFFFFF;
    --sm-text-70: rgba(255, 255, 255, 0.70);
    --sm-text-50: rgba(255, 255, 255, 0.50);
    --sm-text-20: rgba(255, 255, 255, 0.20);
    --sm-border: rgba(255, 255, 255, 0.12);
    --sm-border-strong: rgba(255, 255, 255, 0.20);
    --sm-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.40);
    --sm-glow-primary: 0 0 20px rgba(255, 140, 0, 0.15);
    --metric-dpb: #FF8C00;
    --metric-text: #74B8FF;
    --metric-formula: #9BDE62;
    --metric-table: #FF7A4C;
}

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    min-height: 100%;
    background:
        radial-gradient(circle at top center, rgba(255, 140, 0, 0.18), transparent 24%),
        radial-gradient(circle at 85% 18%, rgba(255, 255, 255, 0.08), transparent 18%),
        linear-gradient(180deg, #040404 0%, #000000 62%, #050505 100%);
    color: var(--sm-text-100);
    font-family: "Inter", "Noto Sans SC", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
    background-size: 36px 36px;
    mask-image: radial-gradient(circle at top, rgba(0, 0, 0, 0.55), transparent 82%);
}

body {
    padding: 34px 20px 56px;
}

a {
    color: inherit;
    text-decoration: none;
}

button {
    font: inherit;
}

.stage {
    max-width: 1220px;
    margin: 0 auto;
}

.brand-shell {
    position: relative;
    overflow: hidden;
    padding: 1px;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.18), rgba(255, 140, 0, 0.28), rgba(255, 255, 255, 0.08));
    box-shadow: var(--sm-shadow-lg);
}

.brand-shell__inner {
    position: relative;
    overflow: hidden;
    padding: 24px;
    border-radius: 29px;
    background:
        radial-gradient(circle at top right, rgba(255, 140, 0, 0.12), transparent 28%),
        linear-gradient(180deg, rgba(16, 17, 18, 0.98), rgba(6, 6, 6, 0.98));
}

.brand-shell__inner::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), transparent 18%);
}

.hero {
    position: relative;
    display: block;
    margin-bottom: 18px;
}

.panel {
    position: relative;
    border-radius: 24px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.14), rgba(255, 140, 0, 0.18), rgba(255, 255, 255, 0.05));
}

.panel__inner {
    position: relative;
    height: 100%;
    border-radius: inherit;
    padding: 22px;
    background: linear-gradient(180deg, rgba(18, 18, 20, 0.96), rgba(10, 10, 11, 0.98));
}

.hero__lead .panel__inner {
    padding: 26px;
}

.hero {
    display: block;
}

.eyebrow {
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--sm-text-100);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.eyebrow::before,
.eyebrow::after {
    content: "";
    flex: 1 1 140px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 140, 0, 0.34), rgba(255, 255, 255, 0.16));
}

.eyebrow::after {
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.16), rgba(255, 140, 0, 0.34), transparent);
}

.eyebrow__label {
    display: inline-block;
    white-space: nowrap;
}

.hero__identity {
    display: flex;
    align-items: center;
    gap: 18px;
    margin: 18px 0 16px;
}

.hero__logo {
    width: 78px;
    height: 78px;
    object-fit: contain;
    flex: 0 0 auto;
    filter: drop-shadow(0 0 24px rgba(255, 140, 0, 0.22));
}

.hero__title {
    margin: 0;
    font-family: "Noto Sans SC", "Inter", sans-serif;
    font-size: clamp(40px, 5vw, 64px);
    line-height: 0.95;
    letter-spacing: -0.05em;
    font-weight: 800;
}

.hero__title span {
    color: var(--sm-primary);
}

.hero__meta-block {
    margin-top: 22px;
    overflow: visible;
}

.hero__section-title {
    display: block;
    margin-bottom: 14px;
    color: var(--sm-text-50);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero__links {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 10px;
    padding-top: 6px;
}

.hero-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 40px;
    padding: 0 14px;
    border-radius: 999px;
    color: var(--sm-text-70);
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.04);
    transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease, background 0.2s ease;
    white-space: nowrap;
    flex: 0 0 auto;
}

.hero-link:hover {
    transform: translateY(-1px);
    color: var(--sm-text-100);
    border-color: rgba(255, 140, 0, 0.30);
    background: rgba(255, 140, 0, 0.10);
}

.hero-link__icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    color: var(--sm-text-70);
    flex: 0 0 auto;
}

.hero-link__icon svg {
    width: 16px;
    height: 16px;
    display: block;
    fill: currentColor;
}

.hero-link:hover .hero-link__icon {
    color: var(--sm-text-100);
}

.control-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 18px;
    margin-bottom: 18px;
}

.control-row .panel__inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 16px 18px;
}

.toolbar__group {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}

.toolbar__label {
    color: var(--sm-text-50);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.feature-list {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-content: start;
}

.feature-item {
    position: relative;
    display: inline-flex;
    flex: 0 0 auto;
}

.feature-item__title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 38px;
    padding: 0 14px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.04);
    color: var(--sm-text-70);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    cursor: default;
    transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease, background 0.2s ease;
    white-space: nowrap;
    text-align: center;
}

.feature-item__dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--sm-primary);
    box-shadow: 0 0 10px rgba(255, 140, 0, 0.28);
    flex: 0 0 auto;
}

.feature-item__desc {
    position: absolute;
    top: calc(100% + 10px);
    left: 0;
    z-index: 2;
    width: 240px;
    padding: 10px 12px;
    border: 1px solid rgba(255, 140, 0, 0.20);
    border-radius: 12px;
    background: rgba(17, 17, 17, 0.96);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    color: var(--sm-text-70);
    font-size: 13px;
    line-height: 1.5;
    opacity: 0;
    visibility: hidden;
    transform: translateY(-4px);
    transition: opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;
    pointer-events: none;
}

.feature-item:hover .feature-item__title {
    transform: translateY(-1px);
    color: var(--sm-text-100);
    border-color: rgba(255, 140, 0, 0.30);
    background: rgba(255, 140, 0, 0.10);
}

.feature-item:hover .feature-item__desc {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

.chipstrip {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.chipstrip__sep {
    width: 1px;
    height: 18px;
    background: rgba(255, 255, 255, 0.10);
    margin: 0 2px;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 38px;
    padding: 0 15px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.05);
    color: var(--sm-text-70);
    cursor: pointer;
    transition: transform 0.2s ease, color 0.2s ease, border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}

.chip:hover {
    transform: translateY(-1px);
    color: var(--sm-text-100);
    border-color: rgba(255, 255, 255, 0.20);
}

.chip__dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.22);
    transition: background 0.2s ease, box-shadow 0.2s ease;
}

.chip--all {
    border-color: rgba(255, 140, 0, 0.26);
    background: rgba(255, 140, 0, 0.08);
}

.chip--active {
    color: var(--sm-text-100);
    border-color: rgba(255, 140, 0, 0.38);
    background: rgba(255, 140, 0, 0.16);
    box-shadow: 0 0 20px rgba(255, 140, 0, 0.12);
}

.chip--active .chip__dot,
.chip--all .chip__dot {
    background: var(--sm-primary);
    box-shadow: 0 0 14px rgba(255, 140, 0, 0.32);
}

.toolbar__save {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    min-height: 42px;
    padding: 0 18px;
    border: none;
    border-radius: 12px;
    background: var(--sm-primary);
    color: var(--sm-text-100);
    cursor: pointer;
    box-shadow: 0 10px 30px rgba(255, 140, 0, 0.24);
    transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}

.toolbar__save:hover {
    transform: translateY(-1px);
    background: var(--sm-primary-hover);
    box-shadow: 0 14px 34px rgba(255, 140, 0, 0.28);
}

.toolbar__save svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.table-wrap {
    position: relative;
}

.table-wrap .panel__inner {
    padding: 0;
    overflow: hidden;
}

.lb-table {
    width: 100%;
    border-collapse: collapse;
}

.lb-table col.col-rank { width: 76px; }
.lb-table col.col-method { width: 32%; }
.lb-table col.col-metric { width: auto; }

.lb-table thead th {
    padding: 18px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.02);
    color: var(--sm-text-50);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    text-align: left;
}

.lb-table thead th.th-metric {
    cursor: pointer;
    transition: color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}

.lb-table thead th.th-active,
.lb-table thead th.th-metric:hover {
    color: var(--sm-text-100);
}

.lb-table thead th.th-active {
    background: linear-gradient(180deg, rgba(255, 140, 0, 0.20), rgba(255, 140, 0, 0.10));
    box-shadow: inset 0 -1px 0 rgba(255, 140, 0, 0.38);
}

.lb-table thead th.th-metric:hover {
    background: rgba(255, 140, 0, 0.10);
}

.col-key {
    display: inline-flex;
    align-items: center;
    gap: 10px;
}

.col-mark {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--metric-dpb);
    box-shadow: 0 0 10px rgba(255, 140, 0, 0.16);
}

th[data-col="text"] .col-mark { background: var(--metric-text); box-shadow: none; }
th[data-col="formula"] .col-mark { background: var(--metric-formula); box-shadow: none; }
th[data-col="table"] .col-mark { background: var(--metric-table); box-shadow: none; }

.sort-arrow {
    width: 11px;
    height: 11px;
    display: inline-block;
    border-right: 2px solid currentColor;
    border-bottom: 2px solid currentColor;
    color: var(--sm-primary);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s ease, visibility 0.2s ease, transform 0.2s ease;
}

.sort-arrow.desc { transform: rotate(45deg) translateY(-1px); }
.sort-arrow.asc { transform: rotate(-135deg) translateY(-1px); }

.lb-table thead th.th-active .sort-arrow {
    opacity: 1;
    visibility: visible;
}

.lb-table tbody td,
.lb-table tfoot td {
    padding: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.data-row {
    transition: background 0.2s ease;
}

.data-row:hover {
    background: rgba(255, 255, 255, 0.03);
}

.r-top3 {
    background: linear-gradient(90deg, rgba(255, 140, 0, 0.05), transparent 52%);
}

.r-1 {
    background: linear-gradient(90deg, rgba(255, 140, 0, 0.10), rgba(255, 140, 0, 0.03) 44%, transparent 80%);
}

.td-rank {
    color: var(--sm-text-50);
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
    font-weight: 600;
}

.td-method {
    color: var(--sm-text-100);
    font-size: 15px;
    font-weight: 600;
}

.td-metric {
    transition: background 0.2s ease, box-shadow 0.2s ease;
}

.td-metric.td-active {
    background: linear-gradient(180deg, rgba(255, 140, 0, 0.12), rgba(255, 140, 0, 0.04));
    box-shadow: inset 1px 0 0 rgba(255, 140, 0, 0.14), inset -1px 0 0 rgba(255, 140, 0, 0.14);
}

.bar {
    display: grid;
    grid-template-columns: minmax(70px, 1fr) auto;
    align-items: center;
    gap: 12px;
}

.bar__rail {
    position: relative;
    height: 9px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
}

.bar__fill {
    position: absolute;
    inset: 0 auto 0 0;
    width: 0;
    border-radius: inherit;
    background: linear-gradient(90deg, #FF8C00, #FFB357);
    box-shadow: 0 0 14px rgba(255, 140, 0, 0.22);
    transition: width 0.5s cubic-bezier(0.2, 0.7, 0.2, 1);
}

.td-metric[data-col="text"] .bar__fill {
    background: linear-gradient(90deg, #5B9FFF, #9BD1FF);
    box-shadow: none;
}

.td-metric[data-col="formula"] .bar__fill {
    background: linear-gradient(90deg, #76C73A, #A9E86C);
    box-shadow: none;
}

.td-metric[data-col="table"] .bar__fill {
    background: linear-gradient(90deg, #FF6A3A, #FF9A76);
    box-shadow: none;
}

.bar__pct {
    color: var(--sm-text-70);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}

.best-row td {
    border-top: 1px solid rgba(255, 140, 0, 0.16);
    border-bottom: none;
    background: linear-gradient(90deg, rgba(255, 140, 0, 0.05), rgba(255, 140, 0, 0.02));
}

.best__label {
    display: block;
    margin-bottom: 6px;
    color: var(--sm-text-50);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.best__name {
    color: var(--sm-text-100);
    font-size: 13px;
    font-weight: 600;
}

.td-method--empty {
    color: transparent;
}

.footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding-top: 18px;
    color: var(--sm-text-50);
    font-size: 12px;
}

.legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.legend__item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.legend__swatch {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--metric-dpb);
}

.legend__item.text .legend__swatch { background: var(--metric-text); }
.legend__item.formula .legend__swatch { background: var(--metric-formula); }
.legend__item.table .legend__swatch { background: var(--metric-table); }

.cursor-tip {
    position: fixed;
    z-index: 10;
    pointer-events: none;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border: 1px solid rgba(255, 140, 0, 0.24);
    border-radius: 10px;
    background: rgba(17, 17, 17, 0.94);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    color: var(--sm-text-100);
    opacity: 0;
    transform: translate(10px, 14px);
    transition: opacity 0.15s ease;
}

.cursor-tip.show {
    opacity: 1;
}

.tip__label {
    color: var(--sm-text-50);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.tip__val {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 600;
}

body.saving .cursor-tip {
    opacity: 0 !important;
}

@media (max-width: 980px) {
    .control-row {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 820px) {
    body {
        padding: 14px 10px 30px;
    }

    .brand-shell {
        border-radius: 24px;
    }

    .brand-shell__inner {
        padding: 14px;
        border-radius: 23px;
    }

    .panel {
        border-radius: 18px;
    }

    .panel__inner,
    .hero__lead .panel__inner {
        padding: 18px;
    }

    .hero__identity {
        align-items: flex-start;
    }

    .eyebrow {
        gap: 10px;
    }

    .hero__logo {
        width: 60px;
        height: 60px;
    }

    .control-row .panel__inner,
    .footer {
        flex-direction: column;
        align-items: flex-start;
    }

    .table-wrap .panel__inner {
        overflow-x: auto;
    }

    .lb-table {
        min-width: 720px;
    }
}
</style>
</head>
<body>
<main class="stage">
<article id="lb-card" class="brand-shell">
    <div class="brand-shell__inner">
        <header class="hero">
            <section class="hero__lead panel">
                <div class="panel__inner">
                    <div class="eyebrow">
                        <span class="eyebrow__label">Leaderboard Table</span>
                    </div>

                    <div class="hero__identity">
                        __LOGO_HERO__
                        <div>
                            <h1 class="hero__title">Doc<span>Parsing</span>Bench</h1>
                        </div>
                    </div>

                    <div class="hero__meta-block">
                        <span class="hero__section-title">Project Resources · Repo & Datasets</span>
                        <div class="hero__links" aria-label="Project resources">
                            <a class="hero-link" href="https://github.com/SoMarkAI/docparsingbench" target="_blank" rel="noopener">
                                <span class="hero-link__icon"><svg viewBox="0 0 24 24"><path d="M12 .297C5.373.297 0 5.67 0 12.297c0 5.302 3.438 9.8 8.207 11.387.6.113.793-.26.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.757-1.333-1.757-1.089-.745.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.762-1.605-2.665-.306-5.467-1.334-5.467-5.933 0-1.311.468-2.381 1.236-3.221-.124-.303-.535-1.523.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.553 3.297-1.23 3.297-1.23.654 1.653.243 2.873.119 3.176.77.84 1.235 1.911 1.235 3.221 0 4.61-2.807 5.624-5.479 5.92.43.372.823 1.102.823 2.222v3.293c0 .32.192.694.8.576C20.566 22.094 24 17.6 24 12.297 24 5.67 18.627.297 12 .297"/></svg></span>
                                <span>GitHub ↗</span>
                            </a>
                            <a class="hero-link" href="https://huggingface.co/datasets/SoMarkAI/DocParsingBench" target="_blank" rel="noopener">
                                <span class="hero-link__icon" aria-hidden="true">🤗</span>
                                <span>Hugging Face ↗</span>
                            </a>
                            <a class="hero-link" href="https://modelscope.cn/datasets/SoMark/DocParsingBench" target="_blank" rel="noopener">
                                <span class="hero-link__icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4" fill="currentColor"/><circle cx="4" cy="12" r="2.2" fill="currentColor" opacity=".7"/><circle cx="20" cy="12" r="2.2" fill="currentColor" opacity=".7"/><circle cx="12" cy="4" r="2.2" fill="currentColor" opacity=".55"/><circle cx="12" cy="20" r="2.2" fill="currentColor" opacity=".55"/></svg></span>
                                <span>ModelScope ↗</span>
                            </a>
                        </div>
                    </div>

                    <div class="hero__meta-block">
                        <span class="hero__section-title">Interaction Guide</span>
                        <div class="feature-list" aria-label="Interaction guide">
                        <div class="feature-item">
                            <span class="feature-item__title"><span class="feature-item__dot"></span>Switch View</span>
                            <span class="feature-item__desc">Use the view chips to jump between all documents and each industry slice.</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-item__title"><span class="feature-item__dot"></span>Sort Columns</span>
                            <span class="feature-item__desc">Click any metric header to reorder methods and compare strengths from a new angle.</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-item__title"><span class="feature-item__dot"></span>Save PNG</span>
                            <span class="feature-item__desc">Export the current leaderboard state as a shareable image without leaving the page.</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-item__title"><span class="feature-item__dot"></span>Raw Values</span>
                            <span class="feature-item__desc">Hover any metric cell to reveal the precise score behind the percentage bar.</span>
                        </div>
                        </div>
                    </div>
                </div>
            </section>
        </header>

        <section class="control-row">
            <div class="panel">
                <div class="panel__inner">
                    <div class="toolbar__group">
                        <span class="toolbar__label">Views</span>
                        <div class="chipstrip" id="chipstrip"></div>
                    </div>
                </div>
            </div>
            <div class="panel">
                <div class="panel__inner">
                    <button class="toolbar__save" id="save-btn" aria-label="Save as image">
                        <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        Save current view
                    </button>
                </div>
            </div>
        </section>

        <section class="table-wrap panel">
            <div class="panel__inner">
                <table class="lb-table" id="lb-table">
                    <colgroup>
                        <col class="col-rank">
                        <col class="col-method">
                        <col class="col-metric">
                        <col class="col-metric">
                        <col class="col-metric">
                        <col class="col-metric">
                    </colgroup>
                    <thead>
                        <tr>
                            <th class="th-rank">No.</th>
                            <th class="th-method">Methods</th>
                            <th class="th-metric th-active" data-col="dpb">
                                <span class="col-key"><span class="col-mark"></span>DPB<span class="sort-arrow desc"></span></span>
                            </th>
                            <th class="th-metric" data-col="text">
                                <span class="col-key"><span class="col-mark"></span>Text<span class="sort-arrow desc"></span></span>
                            </th>
                            <th class="th-metric" data-col="formula">
                                <span class="col-key"><span class="col-mark"></span>Formula<span class="sort-arrow desc"></span></span>
                            </th>
                            <th class="th-metric" data-col="table">
                                <span class="col-key"><span class="col-mark"></span>Table<span class="sort-arrow desc"></span></span>
                            </th>
                        </tr>
                    </thead>
                    <tbody id="lb-tbody"></tbody>
                    <tfoot>
                        <tr class="best-row">
                            <td class="td-rank">★</td>
                            <td class="td-method td-method--empty">&nbsp;</td>
                            <td class="td-best" data-col="dpb"><span class="best__label">Best · DPB</span><span class="best__name">—</span></td>
                            <td class="td-best" data-col="text"><span class="best__label">Best · Text</span><span class="best__name">—</span></td>
                            <td class="td-best" data-col="formula"><span class="best__label">Best · Formula</span><span class="best__name">—</span></td>
                            <td class="td-best" data-col="table"><span class="best__label">Best · Table</span><span class="best__name">—</span></td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        </section>

        <footer class="footer">
            <div>Generated __GENERATED_AT__ · __N_METHODS__ methods evaluated</div>
            <div class="legend">
                <span class="legend__item"><span class="legend__swatch"></span>DPB</span>
                <span class="legend__item text"><span class="legend__swatch"></span>Text</span>
                <span class="legend__item formula"><span class="legend__swatch"></span>Formula</span>
                <span class="legend__item table"><span class="legend__swatch"></span>Table</span>
            </div>
        </footer>
    </div>
</article>

<div id="cursor-tip" class="cursor-tip" role="tooltip">
    <span class="tip__label">raw</span><span class="tip__val">0.0000</span>
</div>
</main>

<script type="application/json" id="lb-data">__DATA_JSON__</script>
<script>
(function() {
    const DATA = JSON.parse(document.getElementById('lb-data').textContent);
    const METRICS = ['dpb', 'text', 'formula', 'table'];
    const METRIC_LABELS = { dpb: 'DPB', text: 'Text', formula: 'Formula', table: 'Table' };

    const state = {
        view: 'all',
        sort: { col: 'dpb', asc: false },
    };

    const tbody = document.getElementById('lb-tbody');
    const table = document.getElementById('lb-table');
    const tip = document.getElementById('cursor-tip');
    const tipLabel = tip.querySelector('.tip__label');
    const tipVal = tip.querySelector('.tip__val');

    const chipstrip = document.getElementById('chipstrip');
    const makeChip = (ind, isAll) => {
        const btn = document.createElement('button');
        btn.className = 'chip' + (isAll ? ' chip--all' : '') + (ind === 'all' ? ' chip--active' : '');
        btn.dataset.view = ind;
        btn.innerHTML =
            '<span class="chip__dot"></span>' +
            (ind === 'all' ? 'All documents' : ind.charAt(0).toUpperCase() + ind.slice(1));
        btn.addEventListener('click', () => setView(ind));
        return btn;
    };
    DATA.industries.forEach((ind) => {
        if (ind === 'all') {
            chipstrip.appendChild(makeChip(ind, true));
            if (DATA.industries.length > 1) {
                const sep = document.createElement('span');
                sep.className = 'chipstrip__sep';
                chipstrip.appendChild(sep);
            }
        } else {
            chipstrip.appendChild(makeChip(ind, false));
        }
    });

    function sortRows(rows, col, asc) {
        return rows.slice().sort((a, b) => asc ? a[col] - b[col] : b[col] - a[col]);
    }

    const rowRegistry = Object.create(null);

    function createRow(modelName) {
        const tr = document.createElement('tr');
        tr.className = 'data-row';
        tr.dataset.model = modelName;

        const tdRank = document.createElement('td');
        tdRank.className = 'td-rank';
        tr.appendChild(tdRank);

        const tdMethod = document.createElement('td');
        tdMethod.className = 'td-method';
        tr.appendChild(tdMethod);

        METRICS.forEach(col => {
            const td = document.createElement('td');
            td.className = 'td-metric';
            td.dataset.col = col;
            td.dataset.label = METRIC_LABELS[col];
            td.innerHTML =
                '<div class="bar">' +
                  '<div class="bar__rail">' +
                    '<div class="bar__fill"></div>' +
                  '</div>' +
                  '<span class="bar__pct"></span>' +
                '</div>';
            tr.appendChild(td);
        });

        rowRegistry[modelName] = tr;
        return tr;
    }

    function renderBody() {
        const models = DATA.views[state.view] || [];
        const sorted = sortRows(models, state.sort.col, state.sort.asc);

        const wanted = new Set();
        const ordered = sorted.map((m, idx) => {
            const tr = rowRegistry[m.model_name] || createRow(m.model_name);
            wanted.add(m.model_name);

            const rank = idx + 1;
            tr.className =
                'data-row' +
                (rank <= 3 ? ' r-top3' : '') +
                (rank === 1 ? ' r-1' : '');

            tr.children[0].textContent = String(rank).padStart(2, '0');
            tr.children[1].textContent = m.display_name;

            METRICS.forEach((col, i) => {
                const td = tr.children[i + 2];
                const val = m[col];
                const width = Math.max(0, Math.min(100, val * 100));
                td.dataset.val = val;
                td.classList.toggle('td-active', col === state.sort.col);
                td.querySelector('.bar__fill').style.width = width.toFixed(1) + '%';
                td.querySelector('.bar__pct').textContent = (val * 100).toFixed(1) + '%';
            });

            return tr;
        });

        Object.keys(rowRegistry).forEach(name => {
            if (!wanted.has(name) && rowRegistry[name].parentNode === tbody) {
                tbody.removeChild(rowRegistry[name]);
            }
        });

        ordered.forEach(tr => tbody.appendChild(tr));

        const best = DATA.bests[state.view] || {};
        METRICS.forEach(col => {
            const cell = table.querySelector('tfoot td.td-best[data-col="' + col + '"] .best__name');
            if (cell) cell.textContent = best[col] || '—';
        });

        updateHeaderSort();
    }

    function updateHeaderSort() {
        const ths = table.querySelectorAll('thead th.th-metric');
        ths.forEach(th => {
            const col = th.dataset.col;
            th.classList.toggle('th-active', col === state.sort.col);
            const arrow = th.querySelector('.sort-arrow');
            if (arrow) {
                arrow.classList.remove('asc', 'desc');
                arrow.classList.add(state.sort.asc ? 'asc' : 'desc');
            }
        });
    }

    table.querySelectorAll('thead th.th-metric').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.col;
            if (state.sort.col === col) {
                state.sort.asc = !state.sort.asc;
            } else {
                state.sort.col = col;
                state.sort.asc = false;
            }
            renderBody();
        });
    });

    function setView(v) {
        state.view = v;
        chipstrip.querySelectorAll('.chip').forEach(c => {
            c.classList.toggle('chip--active', c.dataset.view === v);
        });
        renderBody();
    }

    function showTip(e) {
        const td = e.target.closest('td.td-metric');
        if (!td) { hideTip(); return; }
        const raw = parseFloat(td.dataset.val);
        if (Number.isNaN(raw)) { hideTip(); return; }
        tipLabel.textContent = td.dataset.label || 'raw';
        tipVal.textContent = raw.toFixed(4);
        tip.classList.add('show');
        tip.style.left = e.clientX + 'px';
        tip.style.top = e.clientY + 'px';
    }

    function hideTip() {
        tip.classList.remove('show');
    }

    document.addEventListener('mousemove', (e) => {
        if (e.target.closest('td.td-metric')) {
            showTip(e);
        } else {
            hideTip();
        }
    });
    document.addEventListener('mouseleave', hideTip);

    const saveBtn = document.getElementById('save-btn');
    saveBtn.addEventListener('click', async () => {
        if (!window.html2canvas) {
            alert('html2canvas failed to load (check your internet connection).');
            return;
        }
        const target = document.getElementById('lb-card');
        document.body.classList.add('saving');
        try {
            if (document.fonts && document.fonts.ready) await document.fonts.ready;
            const canvas = await html2canvas(target, {
                backgroundColor: null,
                scale: 2,
                useCORS: true,
                logging: false,
            });
            const link = document.createElement('a');
            const viewLabel = state.view === 'all' ? 'all' : state.view;
            const dateStr = new Date().toISOString().slice(0, 10);
            link.download = 'docparsingbench_leaderboard_' + viewLabel + '_' + dateStr + '.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        } catch (err) {
            console.error(err);
            alert('Save failed: ' + err.message);
        } finally {
            document.body.classList.remove('saving');
        }
    });

    renderBody();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_html(agg: AggregatedResult) -> str:
    """Return a full standalone HTML document for the leaderboard."""
    payload = _build_payload(agg)
    data_json = json.dumps(payload, ensure_ascii=False)

    logo = _logo_data_url()
    logo_hero = (
        f'<img src="{logo}" class="hero__logo" alt="DocParsingBench">'
        if logo
        else '<div class="hero__logo" aria-hidden="true"></div>'
    )

    generated_at = payload["generated_at"]
    issue = datetime.now().strftime("%y%m")
    n_methods = str(payload["n_methods"])

    return (
        _HTML_TEMPLATE
        .replace("__LOGO_HERO__", logo_hero)
        .replace("__GENERATED_AT__", generated_at)
        .replace("__ISSUE__", issue)
        .replace("__N_METHODS__", n_methods)
        .replace("__DATA_JSON__", data_json)
    )


def generate_leaderboard_html(
    labels_path: Path,
    results_dir: Path,
    output_path: Path,
    exclude_model_prefixes: Optional[List[str]] = None,
) -> Path:
    """Aggregate results and write a standalone leaderboard HTML file.

    Returns the resolved output path.
    """
    agg = aggregate(
        labels_path,
        results_dir,
        exclude_model_prefixes=exclude_model_prefixes,
    )
    html = build_html(agg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
