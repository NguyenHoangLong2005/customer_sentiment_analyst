# Kế hoạch Tối ưu hoá PhoBERT (Fine-tuning & Ensemble)

Dựa trên bảng phân tích log và những vấn đề như Overfit, LR decay sớm hay thiếu room epoch ở một số fold, dưới đây là kế hoạch chi tiết để tích hợp toàn bộ các điểm cải tiến bạn vừa đề xuất vào file cấu hình sinh notebook `02_PhoBERT.ipynb`.

## Proposed Changes

### [MODIFY] [scripts/generate_phobert_notebook.py](file:///c:/userdata/sentiment_analyst/scripts/generate_phobert_notebook.py)

#### 1. Hyperparameters (Tác động lớn & nhanh)
- `NUM_EPOCHS = 15`: Cho phép mô hình hội tụ tốt hơn thay vì dừng lại ở đỉnh epoch 10.
- `EARLY_STOP_PAT = 5`: Tăng giới hạn kiên nhẫn khi Val F1 không cải thiện.
- `DROPOUT_RATE = 0.2`: Tăng regularization ở các tầng hidden.
- `WARMUP_RATIO = 0.15`: Kéo dài thời gian khởi động Learning Rate, tránh hiện tượng nhảy max_lr ngay ở epoch 1 làm bất ổn gradient.
- `MAX_LEN = 256`: Tăng giới hạn token để tránh cắt xén các đánh giá dài. Sẽ tăng thời gian train nhưng bù lại thông tin trọn vẹn hơn.

#### 2. Kiến trúc mô hình & Loss Function (Tác động trung bình)
- **Classifier Dropout**: Sau khi load mô hình (`AutoModelForSequenceClassification`), ép cứng thêm Dropout `p=0.3` vào classifier head: `model.classifier.dropout = nn.Dropout(p=0.3)`.
- **Label Smoothing**: Bổ sung cờ `LABEL_SMOOTHING = 0.1` vào tham số `label_smoothing` của `nn.functional.cross_entropy` bên trong class `FocalLoss` để giảm overfit trên nhãn cứng.

#### 3. Inference & Evaluation: K-Fold Ensemble (Nâng cấp lớn nhất)
Thay vì chọn Fold tốt nhất (như hiện tại), logic đánh giá tập **Test** và **Demo Predict** sẽ được đập đi xây lại:
- Load tuần tự cả 5 file `best_model_fold_i`.
- Khi tính toán tập Test, tiến hành lấy **average logits** (trung bình cộng của đầu ra trước khi Softmax) của toàn bộ 5 models trên cùng 1 input batch.
- Quyết định nhãn cuối cùng dựa trên argmax của `ensemble_logits`. Tương tự, phần hàm `predict()` demo cũng sẽ chạy text qua 5 models rồi lấy giá trị trung bình để ra quyết định tin cậy hơn.
- *Lưu ý*: Với cách này, ta sẽ không cần gộp train/val lại để retrain nữa (mục 4), vì Ensemble K-Fold bản chất đã là sử dụng mô hình bao trùm toàn bộ phổ dữ liệu.

## Open Questions

> [!WARNING]
> Việc lưu 5 models và chạy Ensemble lúc predict sẽ tốn thời gian tính toán gấp 5 lần so với chạy 1 model. Điều này hoàn toàn tuyệt vời cho báo cáo (Test Accuracy/F1 sẽ lên đỉnh), nhưng khi bạn định đưa lên Hugging Face Space (deploy), việc load 5 models (5 x ~500MB = 2.5GB) và predict có thể làm chậm app.
> 
> **Câu hỏi:** Bạn có đồng ý với phương án áp dụng Ensemble lên toàn bộ cho Test và Inference Demo không, hay muốn phần Demo cuối cùng chỉ dùng model tốt nhất để nhẹ máy?

## Verification Plan

### Manual Verification
- Chạy lại lệnh sinh notebook: `python scripts/generate_phobert_notebook.py`.
- Mở notebook `02_PhoBERT.ipynb` xem kỹ section 8 (Ensemble Evaluation) và hàm `predict()` để đảm bảo code lặp 5 fold được chèn đúng chuẩn PyTorch.
- Xin confirm từ bạn trước khi thực hiện viết code.
