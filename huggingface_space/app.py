import gradio as gr
import pandas as pd
import os

from inference import predict_sentiment, mock_compare_models
from preprocessing import preprocess, highlight_keywords
import charts

# ─────────────────────────────────────────────
# DESIGN SYSTEM — Unified Dark (#0F172A / #1E293B)
# Accent: Indigo #4F46E5 · Semantic: muted green/amber/red
# ─────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #0F172A !important;
    color: #CBD5E1 !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── HERO ── */
.hero {
    padding: 28px 32px 22px;
    border-bottom: 1px solid #1E293B;
}
.hero-logo {
    width: 40px; height: 40px; border-radius: 10px;
    background: #4F46E5; display: flex;
    align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.hero-title { font-size: 19px; font-weight: 700; color: #F1F5F9; margin: 0; letter-spacing: -0.3px; }
.hero-sub   { font-size: 13px; color: #475569; margin: 3px 0 0; }

/* ── TABS — Sticky Horizontal Bar ── */
.tab-nav {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    background: #0F172A !important;
    border-bottom: 1px solid #1E293B !important;
    padding: 0 16px !important;
    gap: 0 !important;
    scrollbar-width: none;
}
.tab-nav::-webkit-scrollbar { display: none; }
.tab-nav button {
    font-size: 13px !important; font-weight: 500 !important;
    color: #64748B !important; padding: 11px 16px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all .15s !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
.tab-nav button:hover { color: #94A3B8 !important; }
.tab-nav button.selected {
    color: #818CF8 !important; font-weight: 600 !important;
    border-bottom-color: #818CF8 !important;
    background: transparent !important;
}
/* Ẩn overflow menu của Gradio để tránh lỗi lặp text */
.tab-nav .overflow-menu, .tab-nav [class*="overflow"] { display: none !important; }

/* ── KPI GRID ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 20px 0 6px; }
.kpi {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 10px; padding: 16px 18px; transition: border-color .2s;
}
.kpi:hover { border-color: #4F46E5; }
.kpi-num { font-size: 26px; font-weight: 700; color: #F1F5F9; line-height: 1.1; }
.kpi-lbl { font-size: 11px; font-weight: 500; color: #475569;
           text-transform: uppercase; letter-spacing: 1.2px; margin-top: 5px; }
.kpi-num.indigo { color: #818CF8; }
.kpi-num.green  { color: #34D399; }
.kpi-num.amber  { color: #FBBF24; }

/* ── SECTION LABEL ── */
.sec {
    font-size: 11px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin: 20px 0 10px; display: flex; align-items: center; gap: 7px;
}
.sec::after { content: ''; flex: 1; height: 1px; background: #1E293B; }

/* ── PER-CLASS METRIC ── */
.pcm-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }
.pcm { background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 14px 16px; }
.pcm-title { font-size: 12px; font-weight: 600; margin-bottom: 10px; }
.pcm-row   { display: flex; justify-content: space-between; }
.pcm-item  { text-align: center; }
.pcm-val   { font-size: 16px; font-weight: 700; color: #F1F5F9; }
.pcm-key   { font-size: 10px; color: #475569; margin-top: 2px; }

/* ── CHIP ── */
.chip {
    display: inline-block; background: #1E293B; border: 1px solid #334155;
    color: #94A3B8; border-radius: 5px; padding: 3px 9px; font-size: 12px; font-weight: 500; margin: 2px;
}
.chip.indigo { background: rgba(79,70,229,.12); border-color: rgba(99,102,241,.3); color: #818CF8; }
.chip.green  { background: rgba(52,211,153,.08); border-color: rgba(52,211,153,.25); color: #34D399; }

/* ── BUTTON ── */
.btn-main {
    background: #4F46E5 !important; border: none !important;
    border-radius: 7px !important; color: #FFFFFF !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 10px 20px !important; transition: background .2s, transform .15s !important;
}
.btn-main:hover { background: #4338CA !important; transform: translateY(-1px) !important; }

/* ── INPUTS ── */
.gr-textbox textarea, .gr-textbox input {
    background: #1E293B !important; border: 1px solid #334155 !important;
    border-radius: 8px !important; color: #E2E8F0 !important;
    font-size: 14px !important; transition: border-color .15s !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.15) !important;
}

/* ── PANELS ── */
.gr-box, .gr-panel, .gr-block, .gr-group {
    background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 10px !important;
}

/* ── TABLE ── */
table { background: #1E293B !important; border-radius: 8px; overflow: hidden; }
th { background: #0F172A !important; color: #475569 !important;
     font-size: 11px !important; font-weight: 600 !important;
     text-transform: uppercase; letter-spacing: .8px; }
td { color: #CBD5E1 !important; border-color: #334155 !important; font-size: 13px !important; }

/* ── ACCORDION ── */
.gr-accordion { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 8px !important; }

/* ── LABELS ── */
label { color: #475569 !important; font-weight: 500 !important; font-size: 12px !important; }
.gr-markdown p, .gr-markdown li { color: #94A3B8 !important; font-size: 13px !important; }
.gr-markdown h3 { color: #F1F5F9 !important; font-weight: 700 !important; }

/* ── BATCH CARD ── */
.batch-card { background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 18px 22px; }
"""

theme = gr.themes.Base(
    primary_hue="indigo", secondary_hue="slate", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0F172A", body_background_fill_dark="#0F172A",
    block_background_fill="#1E293B", block_background_fill_dark="#1E293B",
    block_border_color="#334155", block_border_color_dark="#334155",
    block_title_text_color="#475569", block_title_text_color_dark="#475569",
    input_background_fill="#1E293B", input_background_fill_dark="#1E293B",
    input_border_color="#334155", input_border_color_dark="#334155",
    body_text_color="#CBD5E1", body_text_color_dark="#CBD5E1",
    button_primary_background_fill="#4F46E5", button_primary_background_fill_dark="#4F46E5",
    button_primary_background_fill_hover="#4338CA", button_primary_text_color="#FFFFFF",
)

# ─────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────
def process_single_prediction(text):
    preprocessed_text = preprocess(text)
    best_label, probs, _ = predict_sentiment(text)
    highlighted = highlight_keywords(text)
    gauge = charts.create_gauge_chart(probs, best_label)
    return best_label, gauge, text, preprocessed_text, highlighted


def process_batch_prediction(file_obj):
    if file_obj is None:
        return pd.DataFrame(), "<p style='color:#475569;padding:12px'>Upload file CSV để bắt đầu.</p>", None
    try:
        file_path = getattr(file_obj, "name", file_obj)
        df = pd.read_csv(file_path)
        col = df.columns[0] if ('review' not in df.columns and 'text' not in df.columns) \
              else ('review' if 'review' in df.columns else 'text')
        df = df.head(100)
        results, counts = [], {"Positive": 0, "Neutral": 0, "Negative": 0, "Unknown": 0}
        for text in df[col]:
            lbl, probs, _ = predict_sentiment(str(text))
            safe = lbl if lbl in counts else "Unknown"
            results.append({"Review": text, "Sentiment": lbl,
                            "Confidence": f"{max(probs.values()):.1%}" if probs else "—"})
            counts[safe] = counts.get(safe, 0) + 1
        res_df = pd.DataFrame(results)
        total = len(df)
        html = f"""
        <div class="batch-card">
          <p style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;
                    letter-spacing:1px;margin:0 0 14px">Batch Report &middot; {total} reviews</p>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
            <div style="text-align:center"><div style="font-size:24px;font-weight:700;color:#F1F5F9">{total}</div>
              <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-top:3px">Total</div></div>
            <div style="text-align:center"><div style="font-size:24px;font-weight:700;color:#34D399">{counts.get('Positive',0)}</div>
              <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-top:3px">Positive</div></div>
            <div style="text-align:center"><div style="font-size:24px;font-weight:700;color:#FBBF24">{counts.get('Neutral',0)}</div>
              <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-top:3px">Neutral</div></div>
            <div style="text-align:center"><div style="font-size:24px;font-weight:700;color:#F87171">{counts.get('Negative',0)}</div>
              <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-top:3px">Negative</div></div>
          </div>
        </div>"""
        out_path = "batch_results.csv"
        res_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        return res_df, html, out_path
    except Exception as e:
        return pd.DataFrame(), f"<p style='color:#F87171;padding:12px'>Lỗi: {e}</p>", None


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
with gr.Blocks(title="Vietnamese E-commerce Review Intelligence") as app:

    # HERO
    gr.HTML("""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:14px">
        <div class="hero-logo">&#x1F9E0;</div>
        <div>
          <p class="hero-title">Vietnamese E-commerce Review Intelligence</p>
          <p class="hero-sub">Fine-tuned PhoBERT-base-v2 &nbsp;&middot;&nbsp; K-Fold Ensemble &nbsp;&middot;&nbsp; 3-class Sentiment Analysis</p>
        </div>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════
        # TAB 1 · DASHBOARD
        # ══════════════════════════════════════
        with gr.Tab("Dashboard"):

            gr.HTML("""
            <div class="kpi-grid">
              <div class="kpi"><div class="kpi-num indigo">87.6%</div><div class="kpi-lbl">Accuracy</div></div>
              <div class="kpi"><div class="kpi-num green">82.3%</div><div class="kpi-lbl">Macro F1</div></div>
              <div class="kpi"><div class="kpi-num amber">10,751</div><div class="kpi-lbl">Total Samples</div></div>
              <div class="kpi"><div class="kpi-num">5-Fold</div><div class="kpi-lbl">Cross-Validation</div></div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Plot(charts.create_donut_chart())
                with gr.Column(scale=3):
                    gr.Plot(charts.create_stacked_bar())

            gr.HTML('<div class="sec">Per-class Metrics</div>')
            gr.HTML("""
            <div class="pcm-grid">
              <div class="pcm">
                <div class="pcm-title" style="color:#F87171">Negative</div>
                <div class="pcm-row">
                  <div class="pcm-item"><div class="pcm-val">86%</div><div class="pcm-key">Precision</div></div>
                  <div class="pcm-item"><div class="pcm-val">78%</div><div class="pcm-key">Recall</div></div>
                  <div class="pcm-item"><div class="pcm-val" style="color:#F87171">82%</div><div class="pcm-key">F1</div></div>
                  <div class="pcm-item"><div class="pcm-val">231</div><div class="pcm-key">Support</div></div>
                </div>
              </div>
              <div class="pcm">
                <div class="pcm-title" style="color:#FBBF24">Neutral</div>
                <div class="pcm-row">
                  <div class="pcm-item"><div class="pcm-val">66%</div><div class="pcm-key">Precision</div></div>
                  <div class="pcm-item"><div class="pcm-val">78%</div><div class="pcm-key">Recall</div></div>
                  <div class="pcm-item"><div class="pcm-val" style="color:#FBBF24">71%</div><div class="pcm-key">F1</div></div>
                  <div class="pcm-item"><div class="pcm-val">416</div><div class="pcm-key">Support</div></div>
                </div>
              </div>
              <div class="pcm">
                <div class="pcm-title" style="color:#34D399">Positive</div>
                <div class="pcm-row">
                  <div class="pcm-item"><div class="pcm-val">95%</div><div class="pcm-key">Precision</div></div>
                  <div class="pcm-item"><div class="pcm-val">92%</div><div class="pcm-key">Recall</div></div>
                  <div class="pcm-item"><div class="pcm-val" style="color:#34D399">93%</div><div class="pcm-key">F1</div></div>
                  <div class="pcm-item"><div class="pcm-val">1,499</div><div class="pcm-key">Support</div></div>
                </div>
              </div>
            </div>
            """)

            gr.HTML("""
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;
                        padding:12px 14px;background:#1E293B;border:1px solid #334155;border-radius:8px">
              <span style="font-size:11px;font-weight:600;color:#334155;text-transform:uppercase;
                           letter-spacing:1px;margin-right:6px">Stack</span>
              <span class="chip indigo">vinai/phobert-base-v2</span>
              <span class="chip">K-Fold Ensemble</span>
              <span class="chip green">Best Fold Deployed</span>
              <span class="chip">Label Smoothing 0.1</span>
              <span class="chip">Class-Weight Balancing</span>
              <span class="chip">underthesea Tokenizer</span>
            </div>
            """)

        # ══════════════════════════════════════
        # TAB 2 · SINGLE PREDICTION
        # ══════════════════════════════════════
        with gr.Tab("Single Prediction"):

            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    input_text = gr.Textbox(label="Review", placeholder="Nhập bình luận tiếng Việt…", lines=6)
                    predict_btn = gr.Button("Analyze", variant="primary", elem_classes=["btn-main"])
                    gr.Examples(
                        examples=[
                            "Sản phẩm cực kỳ chất lượng, shop gói hàng cẩn thận, sẽ ủng hộ tiếp!",
                            "Giao hàng quá chậm, chờ mỏi mòn. Chất lượng cũng tạm.",
                            "Hàng bị lỗi rồi shop ơi, nhắn tin không thấy phản hồi.",
                            "Màu sắc giống trên ảnh, vải hơi mỏng nhưng chấp nhận được.",
                        ],
                        inputs=input_text, label="Examples"
                    )
                with gr.Column(scale=5):
                    out_label = gr.Label(label="Prediction")
                    out_gauge = gr.Plot(label="Confidence")

            gr.HTML('<div class="sec">Explainability</div>')
            with gr.Accordion("Pipeline — tiền xử lý & tách từ", open=False):
                with gr.Row():
                    out_raw = gr.Textbox(label="Raw input", interactive=False, scale=1)
                    out_preprocessed = gr.Textbox(label="Preprocessed", interactive=False, scale=1)

            gr.HTML('<p style="font-size:12px;color:#334155;margin:6px 0 4px">'
                    'Keyword highlighting dựa trên rule-based, không phải model attention.</p>')
            out_explain = gr.HighlightedText(label="Keywords", color_map={"Positive": "green", "Negative": "red"})

            predict_btn.click(fn=process_single_prediction, inputs=input_text,
                              outputs=[out_label, out_gauge, out_raw, out_preprocessed, out_explain])

        # ══════════════════════════════════════
        # TAB 3 · BATCH PREDICTION
        # ══════════════════════════════════════
        with gr.Tab("Batch Prediction"):

            gr.HTML('<p style="font-size:13px;color:#475569;margin:4px 0 14px">'
                    'Upload file <code style="background:#0F172A;color:#818CF8;padding:1px 5px;'
                    'border-radius:4px">.csv</code> có cột review. Tối đa 100 dòng.</p>')

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="CSV File", file_types=[".csv"])
                    batch_btn = gr.Button("Run Batch", variant="primary", elem_classes=["btn-main"])
                with gr.Column(scale=3):
                    batch_summary = gr.HTML("")

            batch_df = gr.Dataframe(label="Results", wrap=True)
            batch_download = gr.File(label="Download CSV")

            batch_btn.click(fn=process_batch_prediction, inputs=file_input,
                            outputs=[batch_df, batch_summary, batch_download])

        # ══════════════════════════════════════
        # TAB 4 · SENTIMENT INSIGHTS
        # ══════════════════════════════════════
        with gr.Tab("Insights"):

            gr.HTML('<p style="font-size:13px;color:#475569;margin:4px 0 14px">'
                    'Thống kê trực quan từ 7,760 mẫu Train thực tế.</p>')
            gr.Plot(charts.create_length_stats_chart())

            gr.HTML('<div class="sec">Top 15 Keywords — phân phối theo cảm xúc</div>')
            gr.HTML('<p style="font-size:12px;color:#334155;margin:0 0 8px">'
                    'Thanh chồng: Positive (xanh) · Neutral (vàng) · Negative (đỏ).</p>')
            gr.Plot(charts.create_top_keywords_bar_chart(n=15))

        # ══════════════════════════════════════
        # TAB 5 · MODEL COMPARISON
        # ══════════════════════════════════════
        with gr.Tab("Comparison"):

            gr.HTML('<p style="font-size:13px;color:#475569;margin:4px 0 14px">'
                    'So sánh PhoBERT vs Baseline TF-IDF + Logistic Regression.</p>')

            # Overall performance card
            gr.HTML("""
            <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;
                        padding:20px 24px;margin-bottom:16px">
              <div class="sec" style="margin-top:0">Overall Performance</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                <div style="border:1px solid #334155;border-radius:8px;padding:14px">
                  <p style="font-size:12px;font-weight:600;color:#475569;margin:0 0 10px">
                    Baseline &middot; TF-IDF + Logistic Regression</p>
                  <div style="display:flex;gap:20px">
                    <div style="text-align:center">
                      <div style="font-size:20px;font-weight:700;color:#94A3B8">80.4%</div>
                      <div style="font-size:10px;color:#475569;margin-top:2px">Accuracy</div></div>
                    <div style="text-align:center">
                      <div style="font-size:20px;font-weight:700;color:#94A3B8">71.5%</div>
                      <div style="font-size:10px;color:#475569;margin-top:2px">Macro F1</div></div>
                  </div>
                  <p style="font-size:11px;color:#334155;margin:10px 0 0">Đã cập nhật theo Test Set (2,146 mẫu).</p>
                </div>
                <div style="border:1px solid rgba(99,102,241,.4);border-radius:8px;
                            background:rgba(79,70,229,.08);padding:14px">
                  <p style="font-size:12px;font-weight:600;color:#818CF8;margin:0 0 10px">
                    PhoBERT-base-v2 (Ours)</p>
                  <div style="display:flex;gap:20px">
                    <div style="text-align:center">
                      <div style="font-size:20px;font-weight:700;color:#34D399">87.6%</div>
                      <div style="font-size:10px;color:#475569;margin-top:2px">Accuracy</div></div>
                    <div style="text-align:center">
                      <div style="font-size:20px;font-weight:700;color:#34D399">82.3%</div>
                      <div style="font-size:10px;color:#475569;margin-top:2px">Macro F1</div></div>
                  </div>
                </div>
              </div>
            </div>
            """)

            # Live comparison — user types review
            gr.HTML('<div class="sec">Live Comparison</div>')
            gr.HTML('<p style="font-size:12px;color:#475569;margin:0 0 8px">'
                    'Nhập bình luận bất kỳ để xem cả hai mô hình dự đoán như thế nào.</p>')
            comp_input = gr.Textbox(label="Review", placeholder="Nhập câu đánh giá sản phẩm của bạn…", lines=3)
            comp_btn = gr.Button("Compare", variant="primary", elem_classes=["btn-main"])
            comp_output = gr.Dataframe(headers=["Model", "Prediction"], label="Results")
            comp_btn.click(fn=mock_compare_models, inputs=comp_input, outputs=comp_output)

        # ══════════════════════════════════════
        # TAB 6 · MODEL EVALUATION
        # ══════════════════════════════════════
        with gr.Tab("Evaluation"):

            gr.HTML('<p style="font-size:13px;color:#475569;margin:4px 0 14px">'
                    'Ensemble 5-Fold &middot; Test set (2,146 mẫu).</p>')

            gr.HTML("""
            <div class="kpi-grid">
              <div class="kpi"><div class="kpi-num indigo">87.6%</div><div class="kpi-lbl">Accuracy</div></div>
              <div class="kpi"><div class="kpi-num green">82.3%</div><div class="kpi-lbl">Macro F1</div></div>
              <div class="kpi"><div class="kpi-num">82.4%</div><div class="kpi-lbl">Macro Precision</div></div>
              <div class="kpi"><div class="kpi-num">82.6%</div><div class="kpi-lbl">Macro Recall</div></div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML('<div class="sec">Classification Report</div>')
                    gr.Dataframe(
                        value=[
                            ["Negative", "86.2%", "78.4%", "82.1%", "231"],
                            ["Neutral",  "65.9%", "77.6%", "71.3%", "416"],
                            ["Positive", "95.1%", "91.7%", "93.4%", "1,499"],
                        ],
                        headers=["Class", "Precision", "Recall", "F1", "Support"],
                    )
                with gr.Column(scale=3):
                    gr.Plot(charts.create_fold_performance_chart())

            gr.HTML('<div class="sec">Confusion Matrix — Test Set</div>')
            gr.Plot(charts.create_confusion_matrix())
            gr.HTML('<p style="font-size:12px;color:#334155;margin:4px 0">'
                    'Hover vào từng ô để xem số lượng và tỷ lệ Recall.</p>')

        # ══════════════════════════════════════
        # TAB 7 · ABOUT
        # ══════════════════════════════════════
        with gr.Tab("About"):

            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("""
                    <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:24px">
                      <h3 style="color:#F1F5F9;font-size:16px;font-weight:700;margin:0 0 10px">
                        Vietnamese E-commerce Review Intelligence</h3>
                      <p style="color:#64748B;font-size:13px;line-height:1.75;margin:0 0 18px">
                        Hệ thống phân tích cảm xúc cho đánh giá sản phẩm TMĐT tiếng Việt,
                        sử dụng <b style="color:#818CF8">PhoBERT-base-v2</b> fine-tune với
                        K-Fold Cross Validation Ensemble.</p>
                      <div class="sec" style="margin-top:0">Architecture</div>
                      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:16px">
                        <span class="chip indigo">vinai/phobert-base-v2</span>
                        <span class="chip">5-Fold Cross Validation</span>
                        <span class="chip green">Label Smoothing = 0.1</span>
                        <span class="chip">Class-Weight Balancing</span>
                        <span class="chip">Classifier Dropout</span>
                        <span class="chip indigo">underthesea Tokenizer</span>
                        <span class="chip">Ensemble Soft Voting</span>
                      </div>
                      <div class="sec">Dataset</div>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
                                  background:#0F172A;border-radius:8px;padding:14px">
                        <div style="text-align:center">
                          <div style="font-size:20px;font-weight:700;color:#818CF8">7,760</div>
                          <div style="font-size:10px;color:#475569;text-transform:uppercase;
                                      letter-spacing:.8px;margin-top:3px">Train</div></div>
                        <div style="text-align:center">
                          <div style="font-size:20px;font-weight:700;color:#818CF8">845</div>
                          <div style="font-size:10px;color:#475569;text-transform:uppercase;
                                      letter-spacing:.8px;margin-top:3px">Validation</div></div>
                        <div style="text-align:center">
                          <div style="font-size:20px;font-weight:700;color:#818CF8">2,146</div>
                          <div style="font-size:10px;color:#475569;text-transform:uppercase;
                                      letter-spacing:.8px;margin-top:3px">Test</div></div>
                      </div>
                    </div>
                    """)
                with gr.Column(scale=2):
                    gr.HTML("""
                    <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:24px;height:100%">
                      <div class="sec" style="margin-top:0">Tech Stack</div>
                      <div style="display:flex;flex-direction:column;gap:8px">
                        <div style="display:flex;align-items:center;gap:10px;padding:10px;background:#0F172A;border-radius:8px">
                          <div style="width:32px;text-align:center;font-size:16px">&#x1F917;</div>
                          <div><div style="font-size:13px;font-weight:600;color:#E2E8F0">Transformers</div>
                               <div style="font-size:11px;color:#475569">PhoBERT Inference Pipeline</div></div>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;padding:10px;background:#0F172A;border-radius:8px">
                          <div style="width:32px;text-align:center;font-size:16px">&#x1F525;</div>
                          <div><div style="font-size:13px;font-weight:600;color:#E2E8F0">PyTorch</div>
                               <div style="font-size:11px;color:#475569">Model Training & Inference</div></div>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;padding:10px;background:#0F172A;border-radius:8px">
                          <div style="width:32px;text-align:center;font-size:16px">&#x1F4CA;</div>
                          <div><div style="font-size:13px;font-weight:600;color:#E2E8F0">Plotly &middot; Gradio</div>
                               <div style="font-size:11px;color:#475569">Interactive Visualization</div></div>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;padding:10px;background:#0F172A;border-radius:8px">
                          <div style="width:32px;text-align:center;font-size:16px">&#x1F524;</div>
                          <div><div style="font-size:13px;font-weight:600;color:#E2E8F0">underthesea</div>
                               <div style="font-size:11px;color:#475569">Vietnamese NLP Tokenizer</div></div>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;padding:10px;background:#0F172A;border-radius:8px">
                          <div style="width:32px;text-align:center;font-size:16px">&#x1F9EE;</div>
                          <div><div style="font-size:13px;font-weight:600;color:#E2E8F0">scikit-learn</div>
                               <div style="font-size:11px;color:#475569">Metrics &middot; Class Weights</div></div>
                        </div>
                      </div>
                    </div>
                    """)

            gr.HTML("""<p style="text-align:center;color:#1E293B;font-size:12px;margin:20px 0 4px">
                Vietnamese Sentiment Analysis Research &middot; Thesis Project
            </p>""")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(theme=theme, css=custom_css, server_name="0.0.0.0", server_port=7860)
