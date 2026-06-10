import json

def code(lines): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":lines}
def md(lines):   return {"cell_type":"markdown","metadata":{},"source":lines}

cells = []

# ── TITLE ──────────────────────────────────────────────────────────────
cells.append(md([
    "# 01 — Tiền Xử Lý Dữ Liệu (Pipeline Chung)\n",
    "\n",
    "Notebook này thực hiện **toàn bộ bước tiền xử lý dùng chung** trước khi đưa vào bất kỳ mô hình nào.\n",
    "Output là 3 file CSV sạch, sẵn sàng để notebook mô hình tải vào.\n",
    "\n",
    "```\n",
    "Load Data → EDA → Cleaning → Text Preprocessing → Rule-based Labeling → Class Weights → Save\n",
    "```\n",
    "\n",
    "**Input :** `data/raw/*.csv`  \n",
    "**Output:** `data/processed/*_labeled.csv` + `class_weights.json`\n",
]))

# ── 3.1 SETUP ──────────────────────────────────────────────────────────
cells.append(md(["## 3.1 Khởi tạo & Tải dữ liệu"]))

cells.append(code([
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
]))

cells.append(code([
    "!pip install underthesea -q\n",
]))

cells.append(code([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re, unicodedata, warnings, os, json\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as mpatches\n",
    "import seaborn as sns\n",
    "from underthesea import word_tokenize\n",
    "from collections import Counter\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ⚠️  CHỈNH SỬA BASE_PATH CHO PHÙ HỢP VỚI DRIVE CỦA BẠN\n",
    "BASE_PATH = '/content/drive/MyDrive/sentiment_analyst/'\n",
    "RAW_PATH  = BASE_PATH + 'data/raw/'        # CSV gốc (không chỉnh sửa)\n",
    "PROC_PATH = BASE_PATH + 'data/processed/'  # CSV đã xử lý + class_weights.json\n",
    "FIG_PATH  = BASE_PATH + 'outputs/figures/' # Biểu đồ cho báo cáo\n",
    "\n",
    "for p in [RAW_PATH, PROC_PATH, FIG_PATH]:\n",
    "    os.makedirs(p, exist_ok=True)\n",
    "\n",
    "ASPECT_COLUMNS = ['Price','Shipping','Outlook','Quality','Size','Shop_Service','General','Others']\n",
    "VOTE_COLUMNS   = ['Price','Shipping','Outlook','Quality','Size','Shop_Service','General']  # loại Others\n",
    "LABEL_MAP      = {-1:'None', 0:'Negative', 1:'Positive', 2:'Neutral'}\n",
    "LABEL_COLORS   = {'None':'#9E9E9E','Negative':'#EF5350','Positive':'#66BB6A','Neutral':'#FFA726'}\n",
    "\n",
    "# ⚠️  Thứ tự này phải khớp với tất cả notebook model\n",
    "LABEL_ENCODE = {'Negative': 0, 'Neutral': 1, 'Positive': 2}\n",
    "LABEL_DECODE = {v: k for k, v in LABEL_ENCODE.items()}\n",
    "\n",
    "train_df = pd.read_csv(RAW_PATH + 'train_data.csv')\n",
    "val_df   = pd.read_csv(RAW_PATH + 'val_data.csv')\n",
    "test_df  = pd.read_csv(RAW_PATH + 'test_data.csv')\n",
    "\n",
    "print('Train:', train_df.shape, '| Val:', val_df.shape, '| Test:', test_df.shape)\n",
    "train_df.head(3)\n",
]))

# ── 3.2 EDA ────────────────────────────────────────────────────────────
cells.append(md(["## 3.2 Khám phá dữ liệu (EDA)"]))

cells.append(code([
    "print('=== TRAIN INFO ===')\n",
    "train_df.info()\n",
    "print('\\n=== MISSING VALUES ===')\n",
    "print(train_df.isnull().sum())\n",
]))

cells.append(code([
    "# Phân bố độ dài review\n",
    "lengths = train_df['Review'].astype(str).apply(len)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
    "sns.histplot(lengths, bins=60, kde=True, ax=axes[0], color='steelblue')\n",
    "axes[0].set_title('Phân bố độ dài Review (Train)', fontweight='bold')\n",
    "axes[0].set_xlabel('Số ký tự')\n",
    "axes[0].set_ylabel('Số lượng')\n",
    "\n",
    "stats = lengths.describe()\n",
    "axes[1].axis('off')\n",
    "table_data = [[k, f'{v:.1f}'] for k, v in stats.items()]\n",
    "tbl = axes[1].table(cellText=table_data, colLabels=['Thống kê','Giá trị'],\n",
    "                    loc='center', cellLoc='center')\n",
    "tbl.scale(1.2, 1.5)\n",
    "axes[1].set_title('Thống kê độ dài Review', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

cells.append(code([
    "# Phân bố nhãn theo từng khía cạnh — TRAIN SET\n",
    "fig, axes = plt.subplots(2, 4, figsize=(20, 8))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(ASPECT_COLUMNS):\n",
    "    counts = train_df[col].map(LABEL_MAP).value_counts()\n",
    "    colors = [LABEL_COLORS.get(l, '#9E9E9E') for l in counts.index]\n",
    "    bars = axes[i].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)\n",
    "    axes[i].set_title(f'{col}', fontweight='bold', fontsize=11)\n",
    "    axes[i].set_ylabel('Số lượng')\n",
    "    for bar, val in zip(bars, counts.values):\n",
    "        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,\n",
    "                     str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
    "\n",
    "legend_patches = [mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()]\n",
    "fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=11,\n",
    "           title='Nhãn cảm xúc', title_fontsize=12, bbox_to_anchor=(0.5, -0.04))\n",
    "plt.suptitle('Phân bố nhãn cảm xúc theo từng khía cạnh (Train set)',\n",
    "             fontsize=14, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig(FIG_PATH + 'label_distribution.png', bbox_inches='tight', dpi=150)\n",
    "plt.show()\n",
]))

cells.append(code([
    "# Kiểm tra số review trùng lặp (thống kê, không xóa)\n",
    "for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:\n",
    "    dup = df['Review'].duplicated().sum()\n",
    "    print(f'[{name}] Trùng lặp: {dup}/{len(df)} ({dup/len(df)*100:.2f}%)')\n",
]))

# ── 3.3 CLEANING ───────────────────────────────────────────────────────
cells.append(md(["## 3.3 Làm sạch dữ liệu cơ bản"]))

cells.append(code([
    "def basic_cleaning(df):\n",
    "    df = df.copy()\n",
    "    df = df.dropna(subset=['Review'])                     # Xóa NaN\n",
    "    df = df[df['Review'].str.strip() != '']              # Xóa chuỗi rỗng\n",
    "    # KHÔNG drop_duplicates: trong TMĐT nhiều user có thể đánh giá giống nhau\n",
    "    return df.reset_index(drop=True)\n",
    "\n",
    "train_df = basic_cleaning(train_df)\n",
    "val_df   = basic_cleaning(val_df)\n",
    "test_df  = basic_cleaning(test_df)\n",
    "\n",
    "print('Sau làm sạch — Train:', len(train_df), '| Val:', len(val_df), '| Test:', len(test_df))\n",
]))

# ── 3.4 TEXT PREPROCESSING ────────────────────────────────────────────
cells.append(md([
    "## 3.4 Pipeline Tiền Xử Lý Văn Bản\n",
    "\n",
    "Thứ tự: Lowercase → Unicode NFC → URL/email/hashtag/mention → Ký tự kéo dài → Emoji → Teencode → Dấu câu → Word segmentation\n",
]))

cells.append(code([
    "teencode_dict = {\n",
    "    'ko':'không','k':'không','kh':'không','khong':'không',\n",
    "    'đc':'được','dc':'được',\n",
    "    'cx':'cũng',\n",
    "    'nma':'nhưng mà',\n",
    "    'sp':'sản phẩm',\n",
    "    'mn':'mọi người','mng':'mọi người',\n",
    "    'vl':'rất','vkl':'rất',\n",
    "    'oke':'ok','okie':'ok','oki':'ok','okela':'ok','okila':'ok',\n",
    "    'ship':'giao hàng',\n",
    "    'ib':'nhắn_tin','rep':'phản_hồi','feedback':'đánh_giá',\n",
    "}\n",
    "\n",
    "# Chỉ 2 token: tích_cực / tiêu_cực — giảm vocabulary, thống nhất tín hiệu\n",
    "emoji_dict = {\n",
    "    # Positive\n",
    "    '❤️':'tích_cực','🧡':'tích_cực','💛':'tích_cực','💚':'tích_cực',\n",
    "    '💙':'tích_cực','💜':'tích_cực','🤍':'tích_cực','❣️':'tích_cực',\n",
    "    '💗':'tích_cực','💓':'tích_cực','😍':'tích_cực','🥰':'tích_cực',\n",
    "    '😊':'tích_cực','😄':'tích_cực','😁':'tích_cực','🤩':'tích_cực',\n",
    "    '😻':'tích_cực','🌟':'tích_cực','⭐':'tích_cực','👍':'tích_cực',\n",
    "    '💪':'tích_cực','✅':'tích_cực','👌':'tích_cực',\n",
    "    # Negative\n",
    "    '😡':'tiêu_cực','😠':'tiêu_cực','🤬':'tiêu_cực','😤':'tiêu_cực',\n",
    "    '👎':'tiêu_cực','❌':'tiêu_cực','😭':'tiêu_cực','😢':'tiêu_cực','🥲':'tiêu_cực',\n",
    "}\n",
    "\n",
    "def preprocess_text(text):\n",
    "    if not isinstance(text, str): return ''\n",
    "    text = text.lower()\n",
    "    text = unicodedata.normalize('NFC', text)\n",
    "    text = re.sub(r'http\\S+|www\\.\\S+', '', text)\n",
    "    text = re.sub(r'\\S+@\\S+', '', text)\n",
    "    text = re.sub(r'#(\\w+)', r'\\1', text)   # #đẹp → đẹp\n",
    "    text = re.sub(r'@\\w+', '', text)\n",
    "    for emo, val in emoji_dict.items():\n",
    "        text = text.replace(emo, f' {val} ')\n",
    "    text = re.sub(r'(.)\\1{2,}', r'\\1', text) # ký tự kéo dài\n",
    "    words = [teencode_dict.get(w, w) for w in text.split()]\n",
    "    text  = ' '.join(words)\n",
    "    text  = re.sub(r'!+', '!', text)\n",
    "    text  = re.sub(r'\\?+', '?', text)\n",
    "    text  = re.sub(r'[^\\w\\s!?_]', ' ', text)\n",
    "    text  = re.sub(r'\\s+', ' ', text).strip()\n",
    "    if text: text = word_tokenize(text, format='text')\n",
    "    return text\n",
    "\n",
    "# Demo\n",
    "samples = [\n",
    "    'Giao hàng nhanh 👍, sp đẹppppp lắm mn ơi k mua là tiếc!!!',\n",
    "    'ship nhanh vkl ❤️❤️ ko ngờ oke vậy nha',\n",
    "]\n",
    "print('=== DEMO TIỀN XỬ LÝ ===')\n",
    "for s in samples:\n",
    "    print(f'  Gốc : {s}')\n",
    "    print(f'  Xử lý: {preprocess_text(s)}')\n",
    "    print()\n",
]))

cells.append(code([
    "# Áp dụng pipeline cho toàn bộ dữ liệu\n",
    "for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:\n",
    "    print(f'Đang xử lý {name}...')\n",
    "    df['Review'] = df['Review'].apply(preprocess_text)\n",
    "\n",
    "# Xóa câu rỗng sau xử lý\n",
    "train_df = train_df[train_df['Review'].str.strip() != ''].reset_index(drop=True)\n",
    "val_df   = val_df[val_df['Review'].str.strip() != ''].reset_index(drop=True)\n",
    "test_df  = test_df[test_df['Review'].str.strip() != ''].reset_index(drop=True)\n",
    "\n",
    "print('\\n✅ Hoàn thành — Train:', len(train_df), '| Val:', len(val_df), '| Test:', len(test_df))\n",
]))

# ── 3.5 IMBALANCED EDA (Aspect-level) ─────────────────────────────────
cells.append(md([
    "## 3.5 Phân tích Mất Cân Bằng Dữ Liệu (Aspect-level EDA)\n",
    "\n",
    "> **Phạm vi:** Phân tích mức độ mất cân bằng **theo từng khía cạnh** (EDA tham khảo).  \n",
    "> Class weights thực dùng cho model được tính ở **Section 3.6.2**.\n",
]))

cells.append(code([
    "ratio_data = {}\n",
    "for col in ASPECT_COLUMNS:\n",
    "    subset = train_df[train_df[col] != -1][col]\n",
    "    counts = subset.value_counts(normalize=True).rename(LABEL_MAP) * 100\n",
    "    ratio_data[col] = counts\n",
    "\n",
    "ratio_df = pd.DataFrame(ratio_data).T.fillna(0)\n",
    "for lbl in ['Negative', 'Positive', 'Neutral']:\n",
    "    if lbl not in ratio_df.columns: ratio_df[lbl] = 0\n",
    "ratio_df = ratio_df[['Negative', 'Positive', 'Neutral']]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "sns.heatmap(ratio_df, annot=True, fmt='.1f', cmap='RdYlGn',\n",
    "            linewidths=0.5, ax=ax, cbar_kws={'label': '%'})\n",
    "ax.set_title('Tỷ lệ % nhãn theo từng khía cạnh (Train set)', fontweight='bold', fontsize=13)\n",
    "ax.set_xlabel('Nhãn cảm xúc')\n",
    "ax.set_ylabel('Khía cạnh')\n",
    "plt.tight_layout()\n",
    "plt.savefig(FIG_PATH + 'imbalance_heatmap.png', bbox_inches='tight', dpi=150)\n",
    "plt.show()\n",
]))

# ── 3.6 RULE-BASED LABELING ───────────────────────────────────────────
cells.append(md([
    "## 3.6 Gán nhãn tổng hợp (Rule-based Labeling)\n",
    "\n",
    "| Điều kiện | Nhãn |\n",
    "|-----------|------|\n",
    "| Chỉ có Positive | **Positive** |\n",
    "| Chỉ có Negative | **Negative** |\n",
    "| Có cả Positive lẫn Negative | **Neutral** |\n",
    "| Chỉ có Neutral(2) | **Neutral** |\n",
    "| Không có aspect nào | **Loại bỏ** |\n",
    "\n",
    "> Cột `Others` bị loại khỏi voting (spam/lạc đề).\n",
]))

cells.append(code([
    "def get_sentiment(row):\n",
    "    votes = [row[a] for a in VOTE_COLUMNS if row[a] != -1]\n",
    "    if not votes: return None\n",
    "    pos = votes.count(1)\n",
    "    neg = votes.count(0)\n",
    "    if pos == 0 and neg == 0: return 'Neutral'\n",
    "    if neg == 0: return 'Positive'\n",
    "    if pos == 0: return 'Negative'\n",
    "    return 'Neutral'   # hỗn hợp → cảm xúc trung tính\n",
    "\n",
    "for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:\n",
    "    df['sentiment'] = df.apply(get_sentiment, axis=1)\n",
    "    print(f'[{name}] Phân bố nhãn:')\n",
    "    print(df['sentiment'].value_counts(dropna=False).to_string())\n",
    "    print()\n",
]))

cells.append(code([
    "# Phân bố nhãn — biểu đồ\n",
    "palette = {'Positive':'#66BB6A', 'Neutral':'#FFA726', 'Negative':'#EF5350'}\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "\n",
    "for ax, (name, df) in zip(axes, [('Train', train_df), ('Val', val_df), ('Test', test_df)]):\n",
    "    counts = df['sentiment'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0)\n",
    "    total  = counts.sum()\n",
    "    bars   = ax.bar(counts.index, counts.values,\n",
    "                    color=[palette[l] for l in counts.index], edgecolor='white')\n",
    "    for bar, val in zip(bars, counts.values):\n",
    "        pct = val / total * 100\n",
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,\n",
    "                f'{int(val):,}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
    "    ax.set_title(f'{name} set', fontweight='bold', fontsize=12)\n",
    "    ax.set_ylabel('Số lượng')\n",
    "    ax.set_ylim(0, counts.max() * 1.25)\n",
    "\n",
    "plt.suptitle('Phân bố nhãn tổng hợp (Rule-based)', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig(FIG_PATH + 'sentiment_distribution.png', bbox_inches='tight', dpi=150)\n",
    "plt.show()\n",
]))

cells.append(code([
    "# Phân tích edge case: pos >= 3*neg bị gán Neutral — limitation cần nêu báo cáo\n",
    "def is_edge_case(row):\n",
    "    votes = [row[a] for a in VOTE_COLUMNS if row[a] != -1]\n",
    "    pos = votes.count(1); neg = votes.count(0)\n",
    "    return pos > 0 and neg > 0 and pos >= 3 * neg\n",
    "\n",
    "edge_cases = train_df[train_df.apply(is_edge_case, axis=1)].copy()\n",
    "edge_cases['pos_count'] = edge_cases[VOTE_COLUMNS].apply(\n",
    "    lambda r: [v for v in r if v != -1].count(1), axis=1)\n",
    "edge_cases['neg_count'] = edge_cases[VOTE_COLUMNS].apply(\n",
    "    lambda r: [v for v in r if v != -1].count(0), axis=1)\n",
    "\n",
    "print(f'EDGE CASES (pos >= 3*neg → Neutral): {len(edge_cases)}/{len(train_df)} ({len(edge_cases)/len(train_df)*100:.2f}%)')\n",
    "print('→ Ghi chú vào báo cáo: Limitation của rule-based approach.')\n",
]))

cells.append(code([
    "# Drop các dòng không có nhãn\n",
    "before = {n: len(d) for n, d in [('train',train_df),('val',val_df),('test',test_df)]}\n",
    "train_df = train_df.dropna(subset=['sentiment']).reset_index(drop=True)\n",
    "val_df   = val_df.dropna(subset=['sentiment']).reset_index(drop=True)\n",
    "test_df  = test_df.dropna(subset=['sentiment']).reset_index(drop=True)\n",
    "\n",
    "print('Sau khi loại bỏ dòng không có nhãn:')\n",
    "for name, df, b in [('Train',train_df,before['train']),\n",
    "                     ('Val',val_df,before['val']),\n",
    "                     ('Test',test_df,before['test'])]:\n",
    "    print(f'  [{name}] {b:,} → {len(df):,}  (loại {b-len(df):,} dòng)')\n",
]))

# ── 3.6.1 EDA sau labeling ────────────────────────────────────────────
cells.append(md(["### 3.6.1 EDA nhãn tổng hợp"]))

cells.append(code([
    "label_order = ['Positive', 'Neutral', 'Negative']\n",
    "palette     = {'Positive':'#66BB6A', 'Neutral':'#FFA726', 'Negative':'#EF5350'}\n",
    "counts = train_df['sentiment'].value_counts().reindex(label_order)\n",
    "total  = counts.sum()\n",
    "\n",
    "print('=== PHÂN BỐ NHÃN TỔNG HỢP (TRAIN SET) ===')\n",
    "for label, cnt in counts.items():\n",
    "    pct = cnt / total * 100\n",
    "    bar = '█' * int(pct / 2)\n",
    "    print(f'  {label:10s}: {cnt:5,} ({pct:5.1f}%)  {bar}')\n",
    "\n",
    "imb = counts.max() / counts.min()\n",
    "print(f'\\nImbalance ratio: {imb:.1f}x')\n",
    "if imb > 5:   print('⚠️  Mất cân bằng nghiêm trọng — cần class weights.')\n",
    "elif imb > 2: print('⚠️  Mất cân bằng vừa — nên class weights.')\n",
    "else:         print('✅ Phân bố khá cân bằng.')\n",
]))

cells.append(code([
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
    "wedge_colors = [palette[l] for l in counts.index]\n",
    "\n",
    "wedges, texts, autotexts = axes[0].pie(\n",
    "    counts.values, labels=counts.index, colors=wedge_colors,\n",
    "    autopct='%1.1f%%', startangle=90, pctdistance=0.75,\n",
    "    wedgeprops=dict(edgecolor='white', linewidth=2))\n",
    "for at in autotexts: at.set_fontsize(11); at.set_fontweight('bold')\n",
    "axes[0].set_title('Tỷ lệ nhãn (Train set)', fontweight='bold', fontsize=13)\n",
    "\n",
    "bars = axes[1].bar(counts.index, counts.values,\n",
    "                   color=wedge_colors, edgecolor='white', linewidth=1.5, width=0.5)\n",
    "for bar, (label, val) in zip(bars, counts.items()):\n",
    "    pct = val / total * 100\n",
    "    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,\n",
    "                 f'{val:,}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
    "axes[1].set_title('Số lượng nhãn (Train set)', fontweight='bold', fontsize=13)\n",
    "axes[1].set_ylabel('Số lượng mẫu')\n",
    "axes[1].set_ylim(0, counts.max() * 1.25)\n",
    "axes[1].axhline(y=total/3, color='gray', linestyle='--', linewidth=1, label='Cân bằng lý tưởng')\n",
    "axes[1].legend(fontsize=9)\n",
    "\n",
    "plt.suptitle('Phân bố nhãn tổng hợp sau Rule-based Labeling', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig(FIG_PATH + 'sentiment_distribution_eda.png', bbox_inches='tight', dpi=150)\n",
    "plt.show()\n",
]))

# ── 3.6.2 CLASS WEIGHTS ───────────────────────────────────────────────
cells.append(md([
    "### 3.6.2 Class Weights (Negative=0, Neutral=1, Positive=2)\n",
    "\n",
    "> ⚠️  Thứ tự `weights_list[0]=Negative, [1]=Neutral, [2]=Positive` phải khớp với tất cả notebook model.\n",
]))

cells.append(code([
    "train_df['label'] = train_df['sentiment'].map(LABEL_ENCODE)\n",
    "val_df['label']   = val_df['sentiment'].map(LABEL_ENCODE)\n",
    "test_df['label']  = test_df['sentiment'].map(LABEL_ENCODE)\n",
    "\n",
    "classes = np.array([0, 1, 2])\n",
    "weights = compute_class_weight('balanced', classes=classes, y=train_df['label'].values)\n",
    "class_weight_sentiment = {int(c): float(w) for c, w in zip(classes, weights)}\n",
    "\n",
    "print('=== CLASS WEIGHTS ===')\n",
    "for cls, w in class_weight_sentiment.items():\n",
    "    print(f'  {LABEL_DECODE[cls]:10s} ({cls}): {w:.4f}')\n",
    "\n",
    "# Lưu ra JSON — model notebook load lại, không tính lại\n",
    "with open(PROC_PATH + 'class_weights.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump({\n",
    "        'class_weight_dict': class_weight_sentiment,\n",
    "        'weights_list'     : list(weights),\n",
    "        'label_encode'     : LABEL_ENCODE,\n",
    "        'label_decode'     : {str(k): v for k, v in LABEL_DECODE.items()},\n",
    "    }, f, indent=2, ensure_ascii=False)\n",
    "\n",
    "print('\\n✅ Đã lưu class_weights.json')\n",
]))

# ── 3.7 SAVE ──────────────────────────────────────────────────────────
cells.append(md(["## 3.7 Lưu dữ liệu"]))

cells.append(code([
    "# File preprocessed: đủ cột aspect-level\n",
    "train_df.to_csv(PROC_PATH + 'train_preprocessed.csv', index=False, encoding='utf-8-sig')\n",
    "val_df.to_csv(PROC_PATH   + 'val_preprocessed.csv',   index=False, encoding='utf-8-sig')\n",
    "test_df.to_csv(PROC_PATH  + 'test_preprocessed.csv',  index=False, encoding='utf-8-sig')\n",
    "\n",
    "# ✅ File labeled: Review + sentiment (chữ) + label (số) — bắt buộc có cột 'label'\n",
    "# Model notebook gọi df['label'].tolist() trực tiếp\n",
    "for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:\n",
    "    df[['Review', 'sentiment', 'label']].to_csv(\n",
    "        PROC_PATH + f'{name}_labeled.csv', index=False, encoding='utf-8-sig')\n",
    "\n",
    "print('✅ Đã lưu thành công:')\n",
    "print(f'  [data/processed] train_preprocessed.csv  ({len(train_df):,} mẫu)')\n",
    "print(f'  [data/processed] val_preprocessed.csv    ({len(val_df):,} mẫu)')\n",
    "print(f'  [data/processed] test_preprocessed.csv   ({len(test_df):,} mẫu)')\n",
    "print(f'  [data/processed] train_labeled.csv        ({len(train_df):,} mẫu) — Review|sentiment|label')\n",
    "print(f'  [data/processed] val_labeled.csv          ({len(val_df):,} mẫu)')\n",
    "print(f'  [data/processed] test_labeled.csv         ({len(test_df):,} mẫu)')\n",
    "print(f'  [data/processed] class_weights.json')\n",
    "print(f'  [outputs/figures] sentiment_distribution_eda.png + label_distribution.png')\n",
    "print()\n",
    "print('➡️  Bước tiếp theo: notebooks/04_PhoBERT.ipynb')\n",
]))

# ── ASSEMBLE ──────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "01_Data_Preprocessing.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(r'c:\userdata\sentiment_analyst\notebooks\01_Data_Preprocessing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Done!")
