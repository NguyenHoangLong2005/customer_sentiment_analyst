import os

def update_phobert():
    path = r"c:\userdata\sentiment_analyst\scripts\generate_phobert_notebook.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Imports and Seed
    imports_target = """import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader"""
    
    imports_replacement = """import os, json, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold"""

    content = content.replace(imports_target, imports_replacement)

    seed_target = """if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
]))"""

    seed_replacement = """if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
]))"""
    content = content.replace(seed_target, seed_replacement)

    # 2. Hyperparameters
    hp_target = """DROPOUT_RATE     = 0.1

# ── Label mapping ─────────────────────────────────────────────"""

    hp_replacement = """DROPOUT_RATE     = 0.1
USE_AMP          = True
ACCUM_STEPS      = 2
LLRD_FACTOR      = 0.95
USE_FOCAL_LOSS   = True
K_FOLDS          = 5

# ── Label mapping ─────────────────────────────────────────────"""
    content = content.replace(hp_target, hp_replacement)

    # 3. Data Loading for K-Fold
    data_target = """train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == 'cuda'))"""
    
    data_replacement = """# Combine train and val for K-Fold
full_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
full_train_dataset = PhoBERTDataset(full_train_df, tokenizer, MAX_LEN)

# train_loader and val_loader will be created inside the K-Fold loop."""
    content = content.replace(data_target, data_replacement)

    # 4. Focal Loss & LLRD in Helper Functions
    helper_target = """def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []"""
    
    helper_replacement = """class FocalLoss(nn.Module):
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
    # Classifier head
    params.append({'params': model.classifier.parameters(), 'lr': base_lr})
    # Encoder layers
    for i, layer in enumerate(reversed(model.roberta.encoder.layer)):
        lr = base_lr * (factor ** (i + 1))
        params.append({'params': layer.parameters(), 'lr': lr})
    # Embeddings
    params.append({'params': model.roberta.embeddings.parameters(), 'lr': base_lr * (factor ** 13)})
    return params

def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []"""
    content = content.replace(helper_target, helper_replacement)

    # 5. Train Loop (AMP & Accumulation)
    train_loop_target = """        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()"""

    train_loop_replacement = """        with autocast(enabled=USE_AMP):
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
    content = content.replace(train_loop_target, train_loop_replacement)
    
    # Also add step to loop
    content = content.replace("for batch in loader:", "for step, batch in enumerate(loader):")

    # 6. K-Fold Training Loop wrapper
    kfold_target = """cells.append(code([
    "history = {\\n",
    "    'train_loss':[], 'train_acc':[], 'train_f1':[],\\n",
    "    'val_loss'  :[], 'val_acc'  :[], 'val_f1'  :[],\\n",
    "}\\n",
    "\\n",
    "best_val_f1    = 0.0\\n",
    "early_stop_cnt = 0\\n",
    "best_epoch     = 0\\n",
    "\\n",
    "# ✅ Tách header ra biến riêng — tránh lỗi f-string lồng trong Python < 3.12\\n",
    "header = '{:>6} | {:>10} | {:>9} | {:>9} | {:>8} | {}'.format(\\n",
    "    'Epoch', 'Train Loss', 'Train F1', 'Val Loss', 'Val F1', 'Note')\\n",
    "print(header)\\n",
    "print('-' * 70)\\n",
    "\\n",
    "for epoch in range(1, NUM_EPOCHS + 1):\\n",
    "    tr_loss, tr_acc, tr_f1 = train_epoch(\\n",
    "        model, train_loader, optimizer, scheduler, criterion, device)\\n",
    "    vl_loss, vl_acc, vl_f1, _, _ = evaluate(\\n",
    "        model, val_loader, criterion, device)\\n",
    "\\n",
    "    history['train_loss'].append(tr_loss)\\n",
    "    history['train_acc'].append(tr_acc)\\n",
    "    history['train_f1'].append(tr_f1)\\n",
    "    history['val_loss'].append(vl_loss)\\n",
    "    history['val_acc'].append(vl_acc)\\n",
    "    history['val_f1'].append(vl_f1)\\n",
    "\\n",
    "    note = ''\\n",
    "    if vl_f1 > best_val_f1:\\n",
    "        best_val_f1    = vl_f1\\n",
    "        best_epoch     = epoch\\n",
    "        early_stop_cnt = 0\\n",
    "        note           = '✅ best'\\n",
    "        # Lưu checkpoint tốt nhất\\n",
    "        model.save_pretrained(CKPT_PATH + 'best_model')\\n",
    "        tokenizer.save_pretrained(CKPT_PATH + 'best_model')\\n",
    "    else:\\n",
    "        early_stop_cnt += 1\\n",
    "        note = f'EStop {early_stop_cnt}/{EARLY_STOP_PAT}'\\n",
    "\\n",
    "    print(f'{epoch:>6} | {tr_loss:>10.4f} | {tr_f1:>9.4f} | '\\n",
    "          f'{vl_loss:>9.4f} | {vl_f1:>8.4f} | {note}')\\n",
    "\\n",
    "    if early_stop_cnt >= EARLY_STOP_PAT:\\n",
    "        print(f'\\n⏹ Early stopping tại epoch {epoch}. Best epoch: {best_epoch}')\\n",
    "        break\\n",
    "\\n",
    "print(f'\\n✅ Best Val F1 (macro): {best_val_f1:.4f} tại epoch {best_epoch}')\\n",
    "\\n",
    "# Lưu training history\\n",
    "with open(LOG_PATH + 'training_history.json', 'w', encoding='utf-8') as fout:\\n",
    "    json.dump(history, fout, indent=2)\\n",
]))"""

    # K-Fold Replacement is huge, let's just write the code blocks in a simpler way.
    # We will remove the old Model, Optimizer setup cells and put them inside the K-Fold cell.
    # Actually, replacing large blocks programmatically is risky. Let's rewrite the script fully.
    pass

def generate_full_phobert_script():
    pass

if __name__ == "__main__":
    update_phobert()
