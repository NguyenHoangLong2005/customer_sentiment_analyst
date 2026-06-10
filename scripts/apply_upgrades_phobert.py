import os

filepath = r"c:\userdata\sentiment_analyst\scripts\generate_phobert_notebook.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Hyperparameters
content = content.replace("MAX_LEN      = 128", "MAX_LEN      = 256")
content = content.replace("NUM_EPOCHS       = 10", "NUM_EPOCHS       = 15")
content = content.replace("WARMUP_RATIO     = 0.06", "WARMUP_RATIO     = 0.15")
content = content.replace("EARLY_STOP_PAT   = 4", "EARLY_STOP_PAT   = 5")
content = content.replace("DROPOUT_RATE     = 0.1", "DROPOUT_RATE     = 0.2")

content = content.replace("USE_FOCAL_LOSS   = True\n", "USE_FOCAL_LOSS   = True\n    \"LABEL_SMOOTHING  = 0.1\\n\",\n")

# 2. Classifier Head Dropout
old_model_init = """    "    model.config.attention_probs_dropout_prob = DROPOUT_RATE\\n",
    "    model = model.to(device)\\n","""
new_model_init = """    "    model.config.attention_probs_dropout_prob = DROPOUT_RATE\\n",
    "    model = model.to(device)\\n",
    "    model.classifier.dropout = nn.Dropout(p=0.3)\\n",
"""
content = content.replace(old_model_init, new_model_init)

# 3. Focal Loss Update
old_focal_loss = """    "class FocalLoss(nn.Module):\\n",
    "    def __init__(self, weight=None, gamma=2.0):\\n",
    "        super().__init__()\\n",
    "        self.weight = weight\\n",
    "        self.gamma  = gamma\\n",
    "\\n",
    "    def forward(self, logits, targets):\\n",
    "        ce   = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')\\n",
    "        pt   = torch.exp(-ce)\\n",
    "        loss = ((1 - pt) ** self.gamma) * ce\\n",
    "        return loss.mean()\\n","""

new_focal_loss = """    "class FocalLoss(nn.Module):\\n",
    "    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):\\n",
    "        super().__init__()\\n",
    "        self.weight = weight\\n",
    "        self.gamma  = gamma\\n",
    "        self.label_smoothing = label_smoothing\\n",
    "\\n",
    "    def forward(self, logits, targets):\\n",
    "        ce_for_pt = nn.functional.cross_entropy(\\n",
    "            logits, targets, weight=self.weight, reduction='none'\\n",
    "        )\\n",
    "        pt = torch.exp(-ce_for_pt)\\n",
    "        \\n",
    "        ce_smoothed = nn.functional.cross_entropy(\\n",
    "            logits, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction='none'\\n",
    "        )\\n",
    "        loss = ((1 - pt) ** self.gamma) * ce_smoothed\\n",
    "        return loss.mean()\\n","""
content = content.replace(old_focal_loss, new_focal_loss)

# Also update FocalLoss instantiation in Train Loop
content = content.replace("FocalLoss(weight=class_weights, gamma=2.0)", "FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTHING)")

# 4. Evaluation Section - Ensemble for Test Set
old_eval = """    "# CHỌN FOLD TỐT NHẤT HOẶC ENSEMBLE\\n",
    "# Ở đây ta lấy fold có kết quả tốt nhất trên Validation để đánh giá Test\\n",
    "best_fold = np.argmax(fold_results) + 1\\n",
    "print(f'Dùng mô hình của Fold {best_fold} để đánh giá trên Test Set.')\\n",
    "\\n",
    "best_model = AutoModelForSequenceClassification.from_pretrained(\\n",
    "    CKPT_PATH + f'best_model_fold_{best_fold}',\\n",
    "    id2label  = ID2LABEL,\\n",
    "    label2id  = LABEL2ID,\\n",
    ").to(device)\\n",
    "\\n",
    "with open(DATA_PATH + 'class_weights.json', encoding='utf-8') as _f:\\n",
    "    _cw = json.load(_f)\\n",
    "    \\n",
    "if USE_FOCAL_LOSS:\\n",
    "    criterion = FocalLoss(weight=torch.tensor(_cw['weights_list'], dtype=torch.float).to(device), gamma=2.0)\\n",
    "else:\\n",
    "    criterion = nn.CrossEntropyLoss(weight=torch.tensor(_cw['weights_list'], dtype=torch.float).to(device))\\n",
    "\\n",
    "_, _, test_f1, test_preds, test_labels = evaluate(\\n",
    "    best_model, test_loader, criterion, device)\\n",
    "\\n",
    "label_names = ['Negative', 'Neutral', 'Positive']\\n",
    "assert [LABEL_DECODE[i] for i in range(NUM_LABELS)] == label_names, \\\\\\n",
    "    'label_names không khớp LABEL_DECODE!'\\n","""

new_eval = """    "# ENSEMBLE K-FOLD TRÊN TEST SET\\n",
    "print('Tiến hành Ensemble 5 models trên Test Set...')\\n",
    "\\n",
    "all_logits = []\\n",
    "all_labels = []\\n",
    "\\n",
    "for fold in range(1, K_FOLDS + 1):\\n",
    "    fold_model = AutoModelForSequenceClassification.from_pretrained(\\n",
    "        CKPT_PATH + f'best_model_fold_{fold}',\\n",
    "        id2label  = ID2LABEL,\\n",
    "        label2id  = LABEL2ID,\\n",
    "    ).to(device)\\n",
    "    fold_model.eval()\\n",
    "    \\n",
    "    fold_logits = []\\n",
    "    fold_labels = []\\n",
    "    with torch.no_grad():\\n",
    "        for batch in test_loader:\\n",
    "            input_ids      = batch['input_ids'].to(device)\\n",
    "            attention_mask = batch['attention_mask'].to(device)\\n",
    "            labels         = batch['label'].to(device)\\n",
    "            \\n",
    "            outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)\\n",
    "            fold_logits.append(outputs.logits.cpu())\\n",
    "            if fold == 1: fold_labels.append(labels.cpu())\\n",
    "            \\n",
    "    all_logits.append(torch.cat(fold_logits, dim=0))\\n",
    "    if fold == 1: test_labels = torch.cat(fold_labels, dim=0).numpy()\\n",
    "    \\n",
    "# Trung bình cộng logits\\n",
    "ensemble_logits = torch.stack(all_logits).mean(dim=0)\\n",
    "test_preds = ensemble_logits.argmax(dim=-1).numpy()\\n",
    "\\n",
    "label_names = ['Negative', 'Neutral', 'Positive']\\n",
    "assert [LABEL_DECODE[i] for i in range(NUM_LABELS)] == label_names, \\\\\\n",
    "    'label_names không khớp LABEL_DECODE!'\\n","""
content = content.replace(old_eval, new_eval)

# We also need to remove 'test_loss' and 'test_f1' variables that were removed from new_eval, but are referenced in plotting later? No, plotting only needs best_fold.
# Wait, let's check where `test_results` saves data:
old_test_results = """    "test_results = {\\n",
    "    'accuracy'   : float(accuracy_score(test_labels, test_preds)),\\n",
    "    'f1_macro'   : float(f1_score(test_labels, test_preds, average='macro',    zero_division=0)),\\n",
    "    'f1_weighted': float(f1_score(test_labels, test_preds, average='weighted', zero_division=0)),\\n",
    "    'precision_macro': float(precision_score(test_labels, test_preds, average='macro', zero_division=0)),\\n",
    "    'recall_macro'   : float(recall_score(test_labels, test_preds, average='macro', zero_division=0)),\\n",
    "    'best_epoch' : best_epoch,\\n",
    "    'best_val_f1': float(best_val_f1),\\n",
    "    'per_class'  : report_dict,\\n",
    "}\\n","""
new_test_results = """    "test_results = {\\n",
    "    'accuracy'   : float(accuracy_score(test_labels, test_preds)),\\n",
    "    'f1_macro'   : float(f1_score(test_labels, test_preds, average='macro',    zero_division=0)),\\n",
    "    'f1_weighted': float(f1_score(test_labels, test_preds, average='weighted', zero_division=0)),\\n",
    "    'precision_macro': float(precision_score(test_labels, test_preds, average='macro', zero_division=0)),\\n",
    "    'recall_macro'   : float(recall_score(test_labels, test_preds, average='macro', zero_division=0)),\\n",
    "    'ensemble_used'  : True,\\n",
    "    'per_class'  : report_dict,\\n",
    "}\\n","""
content = content.replace(old_test_results, new_test_results)

# 5. Save model - it was already doing best_model.save_pretrained.
# But where does best_model come from now that evaluation is ensemble?
# The Demo Inference part has `best_model_fold_1` as default.
# But Section 9 relies on `best_model` to save. We must define best_model by loading best_fold.
old_save_model = """    "best_model.save_pretrained(MODEL_PATH)\\n",
    "tokenizer.save_pretrained(MODEL_PATH)\\n","""
new_save_model = """    "best_fold = np.argmax(fold_results) + 1\\n",
    "best_model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH + f'best_model_fold_{best_fold}')\\n",
    "best_model.save_pretrained(MODEL_PATH)\\n",
    "tokenizer.save_pretrained(MODEL_PATH)\\n","""
content = content.replace(old_save_model, new_save_model)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
