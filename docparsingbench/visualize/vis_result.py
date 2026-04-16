import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

import gradio as gr

from docparsingbench.labels import build_image_index


def _resolve_image_path(img_dir: Path, image_index: Dict[str, str], stem: str, img_name: str) -> Path:
    if img_name:
        candidate = img_dir / img_name
        if candidate.exists():
            return candidate

    fallback_name = image_index.get(stem, img_name or f"{stem}.png")
    return img_dir / fallback_name

def load_data(labels_path: Path, img_dir: Path, gt_dir: Path, pred_dir: Path, result_path: Path):
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    image_index = build_image_index(img_dir)
    
    # Parse Schema
    schema = labels_data.get("label_schema", [])
    industry_map = {}
    for ind in schema:
        ind_name = ind["industry"]
        subs = [sub["sub-industry"] for sub in ind["sub_industries"]]
        industry_map[ind_name] = subs

    # Load Result JSON
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    reports = result_data.get("reports", [])
    # Create a map from filename to metrics
    metrics_map = {}
    for r in reports:
        fname = r.get("file")
        summary = r.get("summary")
        if fname and summary:
            metrics_map[fname] = summary

    # Parse Data
    items = labels_data.get("data", [])
    files_data = []
    
    for item in items:
        fname = item["md"]
        # Assuming filename without extension is the key for matching
        stem = Path(fname).stem
        
        # Construct paths
        # Image might have different extension, use the one in json
        img_name = item.get("img", "")
        img_path = _resolve_image_path(img_dir, image_index, stem, img_name)
        
        gt_path = gt_dir / fname
        pred_path = pred_dir / fname
        
        # Get metrics from loaded result
        metrics = metrics_map.get(fname, {
            "dpb_score": 0.0, "text_score": 0.0, 
            "display_formula_score": 0.0, "table_score": 0.0
        })
        
        files_data.append({
            "filename": fname,
            "stem": stem,
            "img_path": img_path,
            "gt_path": gt_path,
            "pred_path": pred_path,
            "industry": item["industry"],
            "sub_industry": item["sub-industry"],
            "metrics": metrics
        })
        
    return industry_map, files_data

def color_score(score):
    val = score * 100
    color = "#ef4444" # red-500
    if val >= 90:
        color = "#22c55e" # green-500
    elif val >= 50:
        color = "#eab308" # yellow-500
    
    text = f"{val:.2f}"
    style = f"color: {color};"
    if val == 100 or val == 0:
        style += " font-weight: bold;"
    
    return f'<span style="{style}">{text}</span>'

def launch(labels_path_str, img_dir_str, gt_dir_str, pred_dir_str, result_path_str):
    labels_path = Path(labels_path_str)
    img_dir = Path(img_dir_str)
    gt_dir = Path(gt_dir_str)
    pred_dir = Path(pred_dir_str)
    result_path = Path(result_path_str)

    industry_map, files_data = load_data(labels_path, img_dir, gt_dir, pred_dir, result_path)

    # UI State helpers
    def get_files(industry, sub_industry):
        filtered = [f for f in files_data if f["industry"] == industry]
        if sub_industry:
            filtered = [f for f in filtered if f["sub_industry"] == sub_industry]
        return filtered

    def get_file_names(files):
        return [f["filename"] for f in files]
    
    # Custom CSS for layout width
    css = """
    .gradio-container {
        max-width: 85% !important;
    }
    """
    
    with gr.Blocks(title="DocParsingBench Visualization") as demo:
        # State
        filtered_files_state = gr.State([]) # List of file items
        
        # Row 1: Controls
        with gr.Row():
            # Column 1: Selection
            with gr.Column(scale=1):
                ind_choices = list(industry_map.keys())
                industry_dd = gr.Dropdown(choices=ind_choices, value=ind_choices[0] if ind_choices else None, label="Industry")
                
                sub_choices = industry_map.get(ind_choices[0], []) if ind_choices else []
                sub_industry_dd = gr.Dropdown(choices=sub_choices, value=sub_choices[0] if sub_choices else None, label="Sub-industry")
                
                file_dd = gr.Dropdown(choices=[], label="File")
            
            # Column 2: Stats
            with gr.Column(scale=1):
                with gr.Row():
                    # Using gr.Group() to provide container styling (borders, rounded corners) that adapts to theme
                    with gr.Column(scale=1, variant="panel", min_width=100):
                        ind_stats = gr.HTML()
                    with gr.Column(scale=1, variant="panel", min_width=100):
                        sub_stats = gr.HTML()
                    with gr.Column(scale=1, variant="panel", min_width=100):
                        page_stats = gr.HTML()
            
            # Column 3: Sort & Nav
            with gr.Column(scale=1):
                with gr.Row():
                    sort_metric = gr.Dropdown(choices=["DPE", "Text", "Formula", "Table"], value="DPE", label="Sort by")
                    sort_order = gr.Radio(choices=["Descending", "Ascending"], value="Descending", label="")
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")

        # Row 2: Display
        with gr.Row():
            # Image
            with gr.Column(scale=1):
                img_display = gr.Image(label="Original Image", type="filepath")
            
            # GT
            # variant="panel" gives it a background and border
            with gr.Column(scale=1, variant="panel"):
                gt_md_display = gr.Markdown(label="ground truth", latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ])
            
            # Pred
            with gr.Column(scale=1, variant="panel"):
                pred_md_display = gr.Markdown(label="predict", latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ])

        # Logic
        def update_subs(ind):
            subs = industry_map.get(ind, [])
            return gr.Dropdown(choices=subs, value=subs[0] if subs else None)

        def update_file_list(ind, sub, sort_key, order):
            files = get_files(ind, sub)
            
            # Sort
            key_map = {
                "DPE": "dpb_score",
                "Text": "text_score",
                "Formula": "display_formula_score",
                "Table": "table_score"
            }
            k = key_map.get(sort_key, "dpb_score")
            reverse = (order == "Descending")
            
            files.sort(key=lambda x: x["metrics"].get(k, 0), reverse=reverse)
            
            fnames = get_file_names(files)
            first_file = fnames[0] if fnames else None
            
            return files, gr.Dropdown(choices=fnames, value=first_file)

        def update_stats(ind, sub, current_file_item):
            # Calculate Industry Stats
            ind_files = [f for f in files_data if f["industry"] == ind]
            
            def calc_avg(flist):
                count = 0
                sums = {"dpb_score": 0, "text_score": 0, "display_formula_score": 0, "table_score": 0}
                for f in flist:
                    m = f["metrics"]
                    count += 1
                    sums["dpb_score"] += m["dpb_score"]
                    sums["text_score"] += m["text_score"]
                    sums["display_formula_score"] += m["display_formula_score"]
                    sums["table_score"] += m["table_score"]
                if count == 0: return None
                return {k: v/count for k,v in sums.items()}

            ind_avg = calc_avg(ind_files)
            
            # Sub Stats
            sub_files = [f for f in files_data if f["industry"] == ind and f["sub_industry"] == sub]
            sub_avg = calc_avg(sub_files)
            
            # Current Page Stats
            cur_metrics = current_file_item["metrics"] if current_file_item else None
            
            def fmt_card(title, m):
                content = ""
                if not m:
                    content = "No Data"
                else:
                    content = f"""
                    <ul>
                        <li><span>DPE:</span> {color_score(m['dpb_score'])}</li>
                        <li><span>Text:</span> {color_score(m['text_score'])}</li>
                        <li><span>Formula:</span> {color_score(m['display_formula_score'])}</li>
                        <li><span>Table:</span> {color_score(m['table_score'])}</li>
                    </ul>
                    """
                return f"""
                <div>
                    <h3>{title}</h3>
                    {content}
                </div>
                """

            return fmt_card(f"{ind} Metrics", ind_avg), \
                   fmt_card(f"{sub} Metrics", sub_avg), \
                   fmt_card("Current Page Metrics", cur_metrics)

        def render_file(file_name, files):
            if not file_name or not files:
                return None, "", "", "Wait...", "Wait...", "Wait..."
            
            # Find file object
            f = next((x for x in files if x["filename"] == file_name), None)
            if not f:
                return None, "Error", "Error", "", "", ""
            
            img_p = f["img_path"]
            gt_text = f["gt_path"].read_text(encoding="utf-8") if f["gt_path"].exists() else "File not found"
            pred_text = f["pred_path"].read_text(encoding="utf-8") if f["pred_path"].exists() else "File not found"
            
            # Update stats as well
            s1, s2, s3 = update_stats(f["industry"], f["sub_industry"], f)
            
            return str(img_p), gt_text, pred_text, s1, s2, s3

        def on_prev_next(current_name, files, direction):
            if not files or not current_name:
                return current_name
            
            try:
                # Find index of current_name in files list
                # files is a list of dicts
                idx = -1
                for i, x in enumerate(files):
                    if x["filename"] == current_name:
                        idx = i
                        break
                
                if idx == -1: return current_name
                
                new_idx = idx + direction
                if 0 <= new_idx < len(files):
                    return files[new_idx]["filename"]
                return current_name
            except ValueError:
                return current_name

        # Wire events
        
        # When industry changes
        industry_dd.change(update_subs, inputs=[industry_dd], outputs=[sub_industry_dd])
        
        # When sub-industry changes (or is updated by industry change), update file list
        # We need to chain this properly. 
        # Actually, industry_dd change triggers sub_industry_dd update.
        # sub_industry_dd change should trigger file list update.
        
        sub_industry_dd.change(
            update_file_list,
            inputs=[industry_dd, sub_industry_dd, sort_metric, sort_order],
            outputs=[filtered_files_state, file_dd]
        )
        
        # When sorting changes
        sort_metric.change(
            update_file_list,
            inputs=[industry_dd, sub_industry_dd, sort_metric, sort_order],
            outputs=[filtered_files_state, file_dd]
        )
        sort_order.change(
            update_file_list,
            inputs=[industry_dd, sub_industry_dd, sort_metric, sort_order],
            outputs=[filtered_files_state, file_dd]
        )

        # When file changes (or is initialized)
        file_dd.change(
            render_file,
            inputs=[file_dd, filtered_files_state],
            outputs=[img_display, gt_md_display, pred_md_display, ind_stats, sub_stats, page_stats]
        )
        
        # Buttons
        prev_btn.click(
            lambda name, files: on_prev_next(name, files, -1),
            inputs=[file_dd, filtered_files_state],
            outputs=[file_dd]
        )
        next_btn.click(
            lambda name, files: on_prev_next(name, files, 1),
            inputs=[file_dd, filtered_files_state],
            outputs=[file_dd]
        )

        # Initial Load
        demo.load(
            update_file_list, 
            inputs=[industry_dd, sub_industry_dd, sort_metric, sort_order],
            outputs=[filtered_files_state, file_dd]
        )

    demo.launch(share=False, server_name="127.0.0.1", css=css)

if __name__ == "__main__":
    pass
