import os
import re

filepath = r"c:\userdata\sentiment_analyst\scripts\generate_phobert_notebook.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Hyperparameters
content = content.replace("BATCH_SIZE       = 32", "BATCH_SIZE       = 64")
content = content.replace("NUM_EPOCHS       = 15", "NUM_EPOCHS       = 25")
content = content.replace("EARLY_STOP_PAT   = 5", "EARLY_STOP_PAT   = 7")
content = content.replace("DROPOUT_RATE     = 0.2", "DROPOUT_RATE     = 0.15")
content = content.replace("LLRD_FACTOR      = 0.95", "LLRD_FACTOR      = 0.9")

# 2. Focal Loss Gamma
content = content.replace(
    "criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTHING)",
    "criterion = FocalLoss(weight=class_weights, gamma=3.0, label_smoothing=LABEL_SMOOTHING)"
)

# 3. Scheduler Import & Init
content = content.replace(
    "    get_linear_schedule_with_warmup,\\n",
    "    get_linear_schedule_with_warmup,\\n\",\n    \"from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts\\n"
)
old_scheduler = "    \"    scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)\\n\","
new_scheduler = "    \"    scheduler    = CosineAnnealingWarmRestarts(optimizer, T_0=max(2, NUM_EPOCHS // 3), T_mult=1.0, eta_min=1e-6)\\n\","
content = content.replace(old_scheduler, new_scheduler)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Hyperparameters updated successfully.")
