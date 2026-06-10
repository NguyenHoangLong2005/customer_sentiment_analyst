import os

filepath = r"c:\userdata\sentiment_analyst\scripts\generate_phobert_notebook.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1 & 2: Test Labels Logic and VRAM OOM in Ensemble Evaluation
old_eval_loop = """    "all_logits = []\\n",
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
    "# Trung bình cộng logits\\n","""

new_eval_loop = """    "# Lấy test_labels 1 lần trước vòng lặp\\n",
    "test_labels = []\\n",
    "with torch.no_grad():\\n",
    "    for batch in test_loader:\\n",
    "        test_labels.extend(batch['label'].numpy())\\n",
    "test_labels = np.array(test_labels)\\n",
    "\\n",
    "all_logits = []\\n",
    "for fold in range(1, K_FOLDS + 1):\\n",
    "    fold_model = AutoModelForSequenceClassification.from_pretrained(\\n",
    "        CKPT_PATH + f'best_model_fold_{fold}',\\n",
    "        id2label  = ID2LABEL,\\n",
    "        label2id  = LABEL2ID,\\n",
    "    ).to(device)\\n",
    "    fold_model.eval()\\n",
    "    \\n",
    "    fold_logits = []\\n",
    "    with torch.no_grad():\\n",
    "        for batch in test_loader:\\n",
    "            input_ids      = batch['input_ids'].to(device)\\n",
    "            attention_mask = batch['attention_mask'].to(device)\\n",
    "            \\n",
    "            outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)\\n",
    "            fold_logits.append(outputs.logits.cpu())\\n",
    "            \\n",
    "    all_logits.append(torch.cat(fold_logits, dim=0))\\n",
    "    \\n",
    "    # Giải phóng VRAM để tránh OOM trên Colab\\n",
    "    del fold_model\\n",
    "    torch.cuda.empty_cache()\\n",
    "\\n",
    "# Trung bình cộng logits\\n","""
content = content.replace(old_eval_loop, new_eval_loop)

# Fix 3: Train/inference mismatch in preprocess()
old_preprocess = """    "    for emo, val in EMOJI.items(): text = text.replace(emo, f' {val} ')\\n",
    "    text = re.sub(r'(.)\\\\1{2,}', r'\\\\1', text)\\n","""
new_preprocess = """    "    text = re.sub(r'(.)\\\\1{2,}', r'\\\\1', text)\\n",
    "    for emo, val in EMOJI.items(): text = text.replace(emo, f' {val} ')\\n","""
content = content.replace(old_preprocess, new_preprocess)

# Fix 4: missing .to(device) in Save Model cell
content = content.replace(
    "best_model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH + f'best_model_fold_{best_fold}')\n",
    "best_model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH + f'best_model_fold_{best_fold}').to(device)\n"
)

# Fix 5: Fallback inference uses best_fold
old_fallback = """    "    print('⏳ best_model chưa được định nghĩa. Đang load từ checkpoint...')\\n",
    "    best_model = AutoModelForSequenceClassification.from_pretrained(\\n",
    "        CKPT_PATH + f'best_model_fold_1',\\n",
    "        id2label=ID2LABEL, label2id=LABEL2ID,\\n",
    "    ).to(device)\\n",
    "    print('✅ Đã load best_model (mặc định fold 1).')\\n","""

new_fallback = """    "    print('⏳ best_model chưa được định nghĩa. Đang load từ checkpoint...')\\n",
    "    try:\\n",
    "        best_fold_id = np.argmax(fold_results) + 1\\n",
    "    except NameError:\\n",
    "        best_fold_id = 1\\n",
    "    best_model = AutoModelForSequenceClassification.from_pretrained(\\n",
    "        CKPT_PATH + f'best_model_fold_{best_fold_id}',\\n",
    "        id2label=ID2LABEL, label2id=LABEL2ID,\\n",
    "    ).to(device)\\n",
    "    print(f'✅ Đã load best_model từ fold {best_fold_id}.')\\n","""
content = content.replace(old_fallback, new_fallback)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied final bugfixes successfully.")
