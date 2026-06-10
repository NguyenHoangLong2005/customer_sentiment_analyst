---
title: Vietnamese E-commerce Review Intelligence
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
python_version: 3.10.13
---

# Vietnamese E-commerce Review Intelligence

Dashboard phân tích cảm xúc đánh giá sản phẩm thương mại điện tử tiếng Việt. 
Mô hình nền tảng: **PhoBERT-base-v2** fine-tuned trên tập dữ liệu đánh giá sản phẩm.

## Tính năng chính:
- 🏠 **Dashboard**: KPI tổng quan về chất lượng mô hình và dữ liệu.
- 🔍 **Single Prediction**: Phân tích đánh giá văn bản đơn lẻ, đi kèm biểu đồ độ tự tin và Explainable AI (Keyword highlighting).
- 📂 **Batch Prediction**: Phân tích hàng loạt đánh giá từ file CSV.
- 📈 **Sentiment Insights**: Dashboard Analytics với Word Cloud, phân phối từ khóa và confidence.
- ⚖️ **Model Comparison**: So sánh hiệu năng thực tế của TF-IDF Logistic Regression và PhoBERT.
- 🤖 **Model Evaluation**: Chi tiết các chỉ số đánh giá, Confusion Matrix, Fold Performance.

## Cài đặt cục bộ (Local Installation)

1. Clone repository này.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy ứng dụng:
   ```bash
   python app.py
   ```
4. Truy cập `http://localhost:7860`.
