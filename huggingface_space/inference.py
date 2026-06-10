import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from preprocessing import preprocess
import os

# Định nghĩa nhãn CHÍNH XÁC từ notebook
ID2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
MAX_LEN = 256

# Đường dẫn chứa weights
MODEL_PATH = "./model_weights"

class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
        print(f"Loading model from {self.model_path} on {self.device}...")
        
        if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, 'config.json')):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                print("[SUCCESS] Model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Error loading model: {e}.")
        else:
            print("[WARNING] Model weights not found in './model_weights'. Cannot run real inference.")

    def predict(self, text, threshold=0.4):
        """
        Dự đoán cảm xúc giống hệt hàm predict() trong notebook.
        """
        preprocessed_text = preprocess(text)
        
        if not preprocessed_text:
            return "Unknown", {ID2LABEL[i]: 0.0 for i in range(3)}, preprocessed_text
            
        if not self.is_loaded:
            # TUYỆT ĐỐI KHÔNG BỊA SỐ LIỆU THEO YÊU CẦU CỦA USER
            # Nếu chưa có model, trả về Unknown và 0.0
            return "Unknown (No Model)", {ID2LABEL[i]: 0.0 for i in range(3)}, preprocessed_text
        
        try:
            enc = self.tokenizer(
                preprocessed_text, 
                max_length=MAX_LEN, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=enc['input_ids'].to(self.device), 
                    attention_mask=enc['attention_mask'].to(self.device)
                )
            
            probs = torch.softmax(outputs.logits, dim=-1).cpu().squeeze()
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()
            
            prob_dict = {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)}
            
            if confidence < threshold:
                return "Unknown", prob_dict, preprocessed_text
                
            return ID2LABEL[pred_id], prob_dict, preprocessed_text
            
        except Exception as e:
            return "Error", {}, preprocessed_text

predictor = SentimentPredictor()

def predict_sentiment(text):
    return predictor.predict(text)

import pickle

# ─────────────────────────────────────────────
# BASELINE MODEL (TF-IDF + LR)
# ─────────────────────────────────────────────
class BaselinePredictor:
    def __init__(self):
        self.tfidf_path = "./models/tfidf_lr/tfidf.pkl"
        self.lr_path = "./models/tfidf_lr/model.pkl"
        self.is_loaded = False
        
        if os.path.exists(self.tfidf_path) and os.path.exists(self.lr_path):
            try:
                with open(self.tfidf_path, 'rb') as f:
                    self.tfidf = pickle.load(f)
                with open(self.lr_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_loaded = True
                print("[SUCCESS] Baseline TF-IDF + LR loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Error loading Baseline: {e}")
        else:
            print("[WARNING] Baseline model files not found in './models/tfidf_lr/'.")

    def predict(self, text):
        if not self.is_loaded:
            return "N/A (Thiếu file .pkl)"
        try:
            preprocessed = preprocess(text)
            if not preprocessed: 
                return "Unknown"
            vec = self.tfidf.transform([preprocessed])
            pred_id = self.model.predict(vec)[0]
            return ID2LABEL[pred_id]
        except Exception as e:
            return f"Lỗi: {e}"

baseline_predictor = BaselinePredictor()

def mock_compare_models(text):
    """
    So sánh dự đoán giữa PhoBERT (Deep Learning) và TF-IDF + LR (Machine Learning truyền thống).
    """
    phobert_label, _, _ = predict_sentiment(text)
    baseline_label = baseline_predictor.predict(text)
    
    return [
        ["Baseline (TF-IDF + LR)", baseline_label],
        ["PhoBERT Base v2 (Ours)", phobert_label]
    ]
