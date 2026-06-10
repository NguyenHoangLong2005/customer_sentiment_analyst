import re
import os

with open(r'c:\userdata\sentiment_analyst\scripts\generate_phobert_notebook.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Imports and Setup
code = code.replace(
    "!pip install transformers datasets accelerate scikit-learn seaborn -q",
    "!pip install transformers datasets accelerate scikit-learn seaborn -q"
)

old_imports = """import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
warnings.filterwarnings('ignore')"""

new_imports = """import os, json, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False"""
code = code.replace(old_imports, new_imports)

# 2. Hyperparams
old_hp = """DROPOUT_RATE     = 0.1

# ── Label mapping ─────────────────────────────────────────────"""
new_hp = """DROPOUT_RATE     = 0.1

USE_AMP          = True
ACCUM_STEPS      = 2
LLRD_FACTOR      = 0.95
USE_FOCAL_LOSS   = True
K_FOLDS          = 5

# ── Label mapping ─────────────────────────────────────────────"""
code = code.replace(old_hp, new_hp)

# 3. Data Load
old_data_loader = """train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
# ⚠️  num_workers=0 — an toàn trên Colab. Nếu muốn nhanh hơn thì thử num_workers=2
#     nhưng có thể gặp RuntimeError: DataLoader worker exited unexpectedly

print(f'Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')

# Kiểm tra 1 batch
batch = next(iter(train_loader))
print('input_ids shape :', batch['input_ids'].shape)
print('label sample    :', batch['label'][:8].tolist())"""

new_data_loader = """# K-Fold: Gộp train và val
full_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
full_train_dataset = PhoBERTDataset(full_train_df, tokenizer, MAX_LEN)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=(device.type == 'cuda'))

print(f'Full Train size: {len(full_train_dataset)} | Test: {len(test_loader)}')"""
code = code.replace(old_data_loader, new_data_loader)

# 4. Focal Loss & train_epoch
old_train = """def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()"""

new_train = """class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce   = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

def get_llrd_params(model, base_lr, factor):
    params = []
    params.append({'params': model.classifier.parameters(), 'lr': base_lr})
    for i, layer in enumerate(reversed(model.roberta.encoder.layer)):
        lr = base_lr * (factor ** (i + 1))
        params.append({'params': layer.parameters(), 'lr': lr})
    params.append({'params': model.roberta.embeddings.parameters(), 'lr': base_lr * (factor ** 13)})
    return params

def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for step, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        with autocast(enabled=USE_AMP):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS"""
code = code.replace(old_train, new_train)

# 5. Remove original model, optimizer blocks completely and put them inside the K-fold block.
# We will just replace the K-Fold execution part.
# Let's replace from cells.append(md(["## 5. Kiến trúc mô hình"])) to the end of section 8.
old_sec5_to_8_regex = re.compile(r'cells\.append\(md\(\[\n\s+"## 5\. Kiến trúc mô hình.*?# 9\. PLOT TRAINING CURVES', re.DOTALL)

new_kfold_code = """cells.append(md([
    "## 5. Huấn luyện với K-Fold Cross Validation\\n",
    "\\n",
    "Áp dụng Stratified K-Fold, LLRD, AMP và Focal Loss."
]))

cells.append(code([
    "skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)\\n",
    "\\n",
    "fold_results = []\\n",
    "\\n",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, full_train_df['label'])):\\n",
    "    print(f'\\\\n' + '='*50)\\n",
    "    print(f'🚀 FOLD {fold + 1}/{K_FOLDS}')\\n",
    "    print('='*50)\\n",
    "    \\n",
    "    train_sub = Subset(full_train_dataset, train_idx)\\n",
    "    val_sub   = Subset(full_train_dataset, val_idx)\\n",
    "    \\n",
    "    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type==\\'cuda\\'))\\n",
    "    val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type==\\'cuda\\'))\\n",
    "    \\n",
    "    # Khởi tạo lại Model cho mỗi Fold\\n",
    "    model = AutoModelForSequenceClassification.from_pretrained(\\n",
    "        PHOBERT_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID\\n",
    "    )\\n",
    "    model.config.hidden_dropout_prob = DROPOUT_RATE\\n",
    "    model.config.attention_probs_dropout_prob = DROPOUT_RATE\\n",
    "    model = model.to(device)\\n",
    "    \\n",
    "    # LLRD Optimizer\\n",
    "    optimizer_grouped_parameters = get_llrd_params(model, LEARNING_RATE, LLRD_FACTOR)\\n",
    "    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, weight_decay=0.01)\\n",
    "    \\n",
    "    total_steps  = len(train_loader) * NUM_EPOCHS // ACCUM_STEPS\\n",
    "    warmup_steps = int(total_steps * WARMUP_RATIO)\\n",
    "    scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)\\n",
    "    \\n",
    "    if USE_FOCAL_LOSS:\\n",
    "        criterion = FocalLoss(weight=class_weights, gamma=2.0)\\n",
    "    else:\\n",
    "        criterion = nn.CrossEntropyLoss(weight=class_weights)\\n",
    "    \\n",
    "    scaler = GradScaler(enabled=USE_AMP)\\n",
    "    \\n",
    "    best_val_f1 = 0.0\\n",
    "    best_epoch = 0\\n",
    "    early_stop_cnt = 0\\n",
    "    history = {\\'train_loss\\':[], \\'train_f1\\':[], \\'val_loss\\':[], \\'val_f1\\':[], \\'lr\\':[]}\\n",
    "    \\n",
    "    for epoch in range(1, NUM_EPOCHS + 1):\\n",
    "        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)\\n",
    "        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion, device)\\n",
    "        \\n",
    "        current_lr = optimizer.param_groups[0][\\'lr\\']\\n",
    "        history[\\'train_loss\\'].append(tr_loss)\\n",
    "        history[\\'train_f1\\'].append(tr_f1)\\n",
    "        history[\\'val_loss\\'].append(vl_loss)\\n",
    "        history[\\'val_f1\\'].append(vl_f1)\\n",
    "        history[\\'lr\\'].append(current_lr)\\n",
    "        \\n",
    "        note = \\'\\'\\n",
    "        if vl_f1 > best_val_f1:\\n",
    "            best_val_f1 = vl_f1\\n",
    "            best_epoch = epoch\\n",
    "            early_stop_cnt = 0\\n",
    "            note = \\'✅ best\\'\\n",
    "            model.save_pretrained(CKPT_PATH + f\\'best_model_fold_{fold+1}\\')\\n",
    "            tokenizer.save_pretrained(CKPT_PATH + f\\'best_model_fold_{fold+1}\\')\\n",
    "        else:\\n",
    "            early_stop_cnt += 1\\n",
    "            note = f\\'EStop {early_stop_cnt}/{EARLY_STOP_PAT}\\'\\n",
    "        \\n",
    "        print(f\\'Epoch {epoch:>2} | LR={current_lr:.2e} | TrL={tr_loss:.4f} TrF1={tr_f1:.4f} | VlL={vl_loss:.4f} VlF1={vl_f1:.4f} | {note}\\')\\n",
    "        \\n",
    "        if early_stop_cnt >= EARLY_STOP_PAT:\\n",
    "            print(f\\'⏹ Early stopping Fold {fold+1} tại epoch {epoch}.\\')\\n",
    "            break\\n",
    "    \\n",
    "    print(f\\'✅ Fold {fold+1} Best Val F1: {best_val_f1:.4f}\\')\\n",
    "    fold_results.append(best_val_f1)\\n",
    "    \\n",
    "    # Luu history tung fold\\n",
    "    with open(LOG_PATH + f\\'training_history_fold_{fold+1}.json\\', \\'w\\', encoding=\\'utf-8\\') as fout:\\n",
    "        json.dump(history, fout, indent=2)\\n",
    "\\n",
    "print(f\\'\\\\n🏆 K-FOLD AVERAGE BEST VAL F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}\\')\\n",
]))

# ══════════════════════════════════════════════════════════════
# 9. PLOT TRAINING CURVES"""
code = re.sub(old_sec5_to_8_regex, new_kfold_code, code)

# 6. Evaluation section
old_eval = """best_model = AutoModelForSequenceClassification.from_pretrained(
    CKPT_PATH + 'best_model',
    id2label  = ID2LABEL,
    label2id  = LABEL2ID,
).to(device)

# ✅ Định nghĩa lại criterion — cần thiết nếu cell này chạy riêng lẻ (restart kernel)
with open(DATA_PATH + 'class_weights.json', encoding='utf-8') as _f:
    _cw = json.load(_f)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(_cw['weights_list'], dtype=torch.float).to(device)
)"""

new_eval = """# CHỌN FOLD TỐT NHẤT HOẶC ENSEMBLE
# Ở đây ta lấy fold có kết quả tốt nhất trên Validation để đánh giá Test
best_fold = np.argmax(fold_results) + 1
print(f'Dùng mô hình của Fold {best_fold} để đánh giá trên Test Set.')

best_model = AutoModelForSequenceClassification.from_pretrained(
    CKPT_PATH + f'best_model_fold_{best_fold}',
    id2label  = ID2LABEL,
    label2id  = LABEL2ID,
).to(device)

with open(DATA_PATH + 'class_weights.json', encoding='utf-8') as _f:
    _cw = json.load(_f)
    
if USE_FOCAL_LOSS:
    criterion = FocalLoss(weight=torch.tensor(_cw['weights_list'], dtype=torch.float).to(device), gamma=2.0)
else:
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(_cw['weights_list'], dtype=torch.float).to(device))"""
code = code.replace(old_eval, new_eval)

# Fix plotting history to use best fold
old_plot = "epochs_range = range(1, len(history['train_loss']) + 1)"
new_plot = """# Load history của fold tốt nhất để vẽ
best_fold = np.argmax(fold_results) + 1
with open(LOG_PATH + f'training_history_fold_{best_fold}.json', 'r', encoding='utf-8') as f:
    history = json.load(f)
epochs_range = range(1, len(history['train_loss']) + 1)
best_epoch = np.argmax(history['val_f1']) + 1"""
code = code.replace(old_plot, new_plot)

# Fix predict fallback
old_predict_fallback = """CKPT_PATH + 'best_model',"""
new_predict_fallback = """CKPT_PATH + f'best_model_fold_1', # Default to fold 1 if not defined"""
code = code.replace(old_predict_fallback, new_predict_fallback)

with open(r'c:\userdata\sentiment_analyst\scripts\generate_phobert_notebook.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Applied upgrades to PhoBERT generator successfully!")
