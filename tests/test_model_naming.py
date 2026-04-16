from docparsingbench.visualize.model_naming import humanize_model_name


def test_humanize_model_name():
    assert humanize_model_name("qwen3_vl_235b_instruct") == "Qwen3-VL-235B-Instruct"
    assert humanize_model_name("dots_ocr_1_5_md") == "dots.ocr-1.5"


def test_humanize_model_name_aliases():
    assert humanize_model_name("qwen3_vl_235b_a22b_instruct") == "Qwen3-VL-235B-A22B-Instruct"
    assert humanize_model_name("gemini_flash_2_no_anchor") == "Gemini Flash 2 (No Anchor)"
    assert humanize_model_name("qwen_2_5_vl_no_anchor") == "Qwen 2.5 VL (No Anchor)"
    assert humanize_model_name("mistral_ocr_api") == "Mistral OCR API"
