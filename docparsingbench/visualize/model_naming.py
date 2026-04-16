import re


ABBREV = {
    "ocr": "OCR",
    "vl": "VL",
    "glm": "GLM",
    "mineru": "MinerU",
    "deepseek": "DeepSeek",
    "qwen3": "Qwen3",
    "3b": "3B",
    "235b": "235B",
    "instruct": "Instruct",
    "pro": "Pro",
    "prod": "Prod",
}

_ABBREV_LOWER = {k.lower(): v for k, v in ABBREV.items()}

DISPLAY_NAME_ALIASES = {
    "paddleocrvl": "PaddleOCR-VL",
    "mineru25": "MinerU2.5",
    "opendoc01b": "OpenDoc-0.1B",
    "monkeyocrpro3b": "MonkeyOCR-pro-3B",
    "ocrverse": "OCRVerse",
    "dotsocr": "dots.ocr",
    "monkeyocr3b": "MonkeyOCR-3B",
    "deepseekocr": "Deepseek-OCR",
    "monkeyocrpro12b": "MonkeyOCR-pro-1.2B",
    "nanonetsocrs": "Nanonets-OCR-s",
    "mineru2vlm": "MinerU2-VLM",
    "olmocr": "olmOCR",
    "dolphin15": "Dolphin-1.5",
    "pointsreader": "POINTS-Reader",
    "mistralocr": "Mistral OCR",
    "ocrflux": "OCRFlux",
    "03b": "0.3B",
    "qwen3vl235ba22binstruct": "Qwen3-VL-235B-A22B-Instruct",
    "gemini25pro": "Gemini-2.5 Pro",
    "qwen25vl": "Qwen2.5-VL",
    "internvl35": "InternVL3.5",
    "internvl3": "InternVL3",
    "gpt4o": "GPT-4o",
    "ppstructurev3": "PP-StructureV3",
    "mineru2pipeline": "Mineru2-pipeline",
    "marker182": "Marker-1.8.2",
    "gotocr": "GOT OCR",
    "markerv162": "Marker v1.6.2",
    "mineruv1310": "MinerU v1.3.10",
    "mistralocrapi": "Mistral OCR API",
    "gpt4oanchored": "GPT-4o (Anchored)",
    "gpt4onoanchor": "GPT-4o (No Anchor)",
    "geminiflash2anchored": "Gemini Flash 2 (Anchored)",
    "geminiflash2noanchor": "Gemini Flash 2 (No Anchor)",
    "qwen2vlnoanchor": "Qwen 2 VL (No Anchor)",
    "qwen25vlnoanchor": "Qwen 2.5 VL (No Anchor)",
    # Existing common result names in this repo.
    "qwen3vl235binstruct": "Qwen3-VL-235B-Instruct",
    "glmocr": "GLM-OCR",
    "monkeyocrpro3bmd": "MonkeyOCR-pro-3B",
    "dotsocr15": "dots.ocr-1.5",
    "deepseekocr2": "Deepseek-OCR-2",
    "chandraocr2": "ChandraOCR-2",
    "hunyuanocr": "HunyuanOCR",
    "paddle15": "PaddleOCR-1.5",
}


def _canonical_model_key(raw: str) -> str:
    value = raw.strip().lower()
    value = re.sub(r"\.result\.json$", "", value)
    value = re.sub(r"_md$", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def _format_token(token: str) -> str:
    lower = token.lower()
    if lower in _ABBREV_LOWER:
        return _ABBREV_LOWER[lower]
    if re.fullmatch(r"\d+(\.\d+)?b", lower):
        return f"{lower[:-1]}B"
    if re.fullmatch(r"a\d+(\.\d+)?b", lower):
        return f"A{lower[1:-1]}B"
    if re.fullmatch(r"v\d+(\.\d+)*", lower):
        return lower
    return token.title()


def humanize_model_name(raw: str) -> str:
    """Convert raw model result stem to unified display name."""
    canonical_key = _canonical_model_key(raw)
    if canonical_key in DISPLAY_NAME_ALIASES:
        return DISPLAY_NAME_ALIASES[canonical_key]

    name = re.sub(r"_md$", "", raw)
    name = re.sub(r"_(\d+)_(\d+)", r"_\1.\2", name)
    tokens = [tok for tok in re.split(r"[_\-\.\s]+", name) if tok]
    result = []
    for tok in tokens:
        result.append(_format_token(tok))
    return " ".join(result)
