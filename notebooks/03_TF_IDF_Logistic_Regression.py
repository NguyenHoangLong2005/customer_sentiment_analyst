# ============================================================
# 03 — Baseline: TF-IDF + Logistic Regression
# ============================================================
# Notebook này huấn luyện mô hình Baseline (TF-IDF + Logistic Regression)
# để so sánh với PhoBERT trong luận văn.
#
# Input : data/processed/*_labeled.csv (output của notebook 01)
# Output: models/tfidf_lr/model.pkl, tfidf_lr/tfidf.pkl
#         outputs/logs/tfidf_lr_results.json
# ============================================================

# %% [1] Khởi tạo môi trường
from google.colab import drive
drive.mount('/content/drive')

# %% 
import pandas as pd
import numpy as np
import json, os, warnings, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# ⚠️ CHỈNH SỬA CHO PHÙ HỢP VỚI DRIVE CỦA BẠN
BASE_PATH = '/content/drive/MyDrive/sentiment_analyst/'
PROC_PATH = BASE_PATH + 'data/processed/'
MODEL_PATH = BASE_PATH + 'models/tfidf_lr/'
LOG_PATH = BASE_PATH + 'outputs/logs/'
FIG_PATH = BASE_PATH + 'outputs/figures/'

for p in [MODEL_PATH, LOG_PATH, FIG_PATH]:
    os.makedirs(p, exist_ok=True)

LABEL_DECODE = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# %% [2] Load dữ liệu đã tiền xử lý
print('Loading preprocessed data...')
train_df = pd.read_csv(PROC_PATH + 'train_labeled.csv')
val_df   = pd.read_csv(PROC_PATH + 'val_labeled.csv')
test_df  = pd.read_csv(PROC_PATH + 'test_labeled.csv')

# Gộp train + val cho Baseline (vì LR không cần early stopping trên val)
full_train = pd.concat([train_df, val_df], ignore_index=True)

# Loại bỏ NaN nếu có
full_train = full_train.dropna(subset=['Review']).reset_index(drop=True)
test_df    = test_df.dropna(subset=['Review']).reset_index(drop=True)

print(f'Train (train+val): {len(full_train):,} mẫu')
print(f'Test:              {len(test_df):,} mẫu')
print(f'\nPhân bố nhãn (Train):')
print(full_train['sentiment'].value_counts().to_string())

# %% [3] TF-IDF Vectorization
print('\n=== TF-IDF Vectorization ===')

tfidf = TfidfVectorizer(
    max_features=50000,      # Giới hạn vocabulary
    ngram_range=(1, 2),      # Unigram + Bigram
    sublinear_tf=True,       # Áp dụng log normalization
    min_df=2,                # Loại từ xuất hiện < 2 lần
    max_df=0.95,             # Loại từ xuất hiện > 95% documents
)

X_train = tfidf.fit_transform(full_train['Review'].astype(str))
X_test  = tfidf.transform(test_df['Review'].astype(str))

y_train = full_train['label'].values
y_test  = test_df['label'].values

print(f'TF-IDF shape (train): {X_train.shape}')
print(f'TF-IDF shape (test):  {X_test.shape}')
print(f'Vocabulary size:      {len(tfidf.vocabulary_):,}')

# %% [4] Class Weights (giống notebook 01)
classes = np.array([0, 1, 2])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
print(f'\nClass weights: {class_weight_dict}')

# %% [5] Logistic Regression Training
print('\n=== Training Logistic Regression ===')

lr_model = LogisticRegression(
    C=1.0,                        # Regularization strength
    class_weight=class_weight_dict,
    solver='lbfgs',
    max_iter=1000,
    multi_class='multinomial',
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train, y_train)
print('✅ Training completed.')

# %% [6] Evaluation trên Test set
print('\n=== Evaluation on Test Set ===')

y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)

acc       = accuracy_score(y_test, y_pred)
f1_macro  = f1_score(y_test, y_pred, average='macro')
f1_weight = f1_score(y_test, y_pred, average='weighted')
prec_mac  = precision_score(y_test, y_pred, average='macro')
rec_mac   = recall_score(y_test, y_pred, average='macro')

print(f'Accuracy:        {acc:.4f}')
print(f'F1 Macro:        {f1_macro:.4f}')
print(f'F1 Weighted:     {f1_weight:.4f}')
print(f'Precision Macro: {prec_mac:.4f}')
print(f'Recall Macro:    {rec_mac:.4f}')

print('\n=== Classification Report ===')
target_names = ['Negative', 'Neutral', 'Positive']
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [7] Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('=== Confusion Matrix ===')
print(cm)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Số lượng
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=axes[0])
axes[0].set_title('Confusion Matrix (số lượng)', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Tỷ lệ (Recall)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=axes[1])
axes[1].set_title('Confusion Matrix (% Recall)', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('TF-IDF + Logistic Regression — Confusion Matrix trên Test set',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH + 'tfidf_lr_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

# %% [8] So sánh với PhoBERT (nếu có file)
phobert_results_path = LOG_PATH + 'test_results.json'
if os.path.exists(phobert_results_path):
    with open(phobert_results_path, 'r') as f:
        phobert = json.load(f)
    
    print('\n=== SO SÁNH: TF-IDF + LR vs PhoBERT ===')
    print(f'{"Metric":<20} {"TF-IDF+LR":>12} {"PhoBERT":>12} {"Delta":>10}')
    print('-' * 56)
    
    comparisons = [
        ('Accuracy',        acc,       phobert['accuracy']),
        ('F1 Macro',        f1_macro,  phobert['f1_macro']),
        ('F1 Weighted',     f1_weight, phobert['f1_weighted']),
        ('Precision Macro', prec_mac,  phobert['precision_macro']),
        ('Recall Macro',    rec_mac,   phobert['recall_macro']),
    ]
    for name, lr_val, pb_val in comparisons:
        delta = pb_val - lr_val
        sign = '+' if delta > 0 else ''
        print(f'{name:<20} {lr_val:>11.4f} {pb_val:>12.4f} {sign}{delta:>9.4f}')
    
    # Bar chart so sánh
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comparisons))
    width = 0.35
    
    lr_vals = [c[1] for c in comparisons]
    pb_vals = [c[2] for c in comparisons]
    labels  = [c[0] for c in comparisons]
    
    bars1 = ax.bar(x - width/2, lr_vals, width, label='TF-IDF + LR', color='#94A3B8')
    bars2 = ax.bar(x + width/2, pb_vals, width, label='PhoBERT',      color='#4F46E5')
    
    ax.set_ylabel('Score')
    ax.set_title('TF-IDF + Logistic Regression vs PhoBERT', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIG_PATH + 'tfidf_vs_phobert.png', bbox_inches='tight', dpi=150)
    plt.show()
else:
    print('\n⚠️ Không tìm thấy file test_results.json của PhoBERT. Bỏ qua so sánh.')

# %% [9] Tìm các câu PhoBERT đúng nhưng LR sai (case studies cho web demo)
if os.path.exists(phobert_results_path):
    # Nếu bạn có file predictions của PhoBERT, load ở đây
    # Hiện tại chỉ in ra các câu LR dự đoán sai
    wrong_mask = y_pred != y_test
    wrong_df = test_df[wrong_mask].copy()
    wrong_df['LR_pred'] = [LABEL_DECODE[p] for p in y_pred[wrong_mask]]
    wrong_df['True']    = [LABEL_DECODE[t] for t in y_test[wrong_mask]]
    
    print(f'\n=== Câu LR dự đoán sai: {len(wrong_df)}/{len(test_df)} ({len(wrong_df)/len(test_df)*100:.1f}%) ===')
    print('\nVí dụ 5 câu sai:')
    for i, row in wrong_df.head(5).iterrows():
        print(f'  Review: {row["Review"][:80]}...')
        print(f'  True: {row["True"]}  |  LR Pred: {row["LR_pred"]}')
        print()

# %% [10] Lưu model & kết quả
print('\n=== Saving ===')

# Save TF-IDF vectorizer
with open(MODEL_PATH + 'tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print(f'✅ TF-IDF vectorizer → {MODEL_PATH}tfidf.pkl')

# Save LR model
with open(MODEL_PATH + 'model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print(f'✅ LR model → {MODEL_PATH}model.pkl')

# Save evaluation results (JSON)
results = {
    'model': 'TF-IDF + Logistic Regression',
    'tfidf_config': {
        'max_features': 50000,
        'ngram_range': [1, 2],
        'sublinear_tf': True,
        'min_df': 2,
        'max_df': 0.95,
        'vocabulary_size': len(tfidf.vocabulary_),
    },
    'lr_config': {
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': class_weight_dict,
    },
    'train_size': len(full_train),
    'test_size': len(test_df),
    'accuracy': float(acc),
    'f1_macro': float(f1_macro),
    'f1_weighted': float(f1_weight),
    'precision_macro': float(prec_mac),
    'recall_macro': float(rec_mac),
    'per_class': {
        name: {
            'precision': report[name]['precision'],
            'recall':    report[name]['recall'],
            'f1-score':  report[name]['f1-score'],
            'support':   report[name]['support'],
        }
        for name in target_names
    },
    'confusion_matrix': cm.tolist(),
}

with open(LOG_PATH + 'tfidf_lr_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'✅ Results → {LOG_PATH}tfidf_lr_results.json')

# Save predictions (for web demo comparison)
pred_df = test_df[['Review', 'sentiment', 'label']].copy()
pred_df['lr_prediction'] = [LABEL_DECODE[p] for p in y_pred]
pred_df['lr_confidence'] = [float(y_proba[i, p]) for i, p in enumerate(y_pred)]
pred_df.to_csv(LOG_PATH + 'tfidf_lr_predictions.csv', index=False, encoding='utf-8-sig')
print(f'✅ Predictions → {LOG_PATH}tfidf_lr_predictions.csv')

print('\n' + '='*60)
print('DONE! Tải 2 file sau về máy để đưa vào web demo:')
print(f'  1. {LOG_PATH}tfidf_lr_results.json')
print(f'  2. {MODEL_PATH}tfidf.pkl + model.pkl')
print('='*60)
