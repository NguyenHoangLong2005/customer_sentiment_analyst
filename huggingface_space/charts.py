import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud

# Màu sắc chuẩn của hệ thống
COLOR_POS = "#22C55E" # Success Green
COLOR_NEU = "#F59E0B" # Warning Yellow/Orange
COLOR_NEG = "#EF4444" # Danger Red
COLOR_PRIMARY = "#2563EB"

def create_gauge_chart(probs, label):
    """
    Tạo Gauge Chart hiển thị confidence score thật từ mô hình.
    """
    if label == "Unknown" or label == "Unknown (No Model)":
        val = 0
        color = "gray"
    else:
        val = probs.get(label, 0) * 100
        color = COLOR_NEU
        if label == "Positive": color = COLOR_POS
        elif label == "Negative": color = COLOR_NEG

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence ({label})"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f3f4f6'},
                {'range': [50, 80], 'color': '#e5e7eb'},
                {'range': [80, 100], 'color': '#d1d5db'}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_donut_chart():
    """
    Dashboard Donut Chart. Dữ liệu thật từ notebook 01:
    Positive: 5484, Neutral: 1486, Negative: 790
    """
    labels = ['Positive', 'Neutral', 'Negative']
    values = [5484, 1486, 790]
    colors = [COLOR_POS, COLOR_NEU, COLOR_NEG]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
    fig.update_layout(title_text="Phân bố Nhãn (Train Set)", height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_stacked_bar():
    """
    Dataset Split. Dữ liệu thật từ notebook 01:
    Train: 7760 | Val: 845 | Test: 2146
    """
    splits = ['Train', 'Validation', 'Test']
    counts = [7760, 845, 2146]
    
    fig = go.Figure(data=[
        go.Bar(name='Dataset Split', x=counts, y=['Samples'], orientation='h', 
               text=splits, textposition='auto', marker_color=[COLOR_PRIMARY, '#06B6D4', '#6366F1'])
    ])
    fig.update_layout(barmode='stack', title_text="Tỉ lệ Data Split", height=150, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_fold_performance_chart():
    """
    Model Evaluation: 5-Fold Performance. 
    Dữ liệu thật từ prompt: Fold 1: 0.823, Fold 2: 0.817, Fold 3: 0.831, Fold 4: 0.819, Fold 5: 0.825
    """
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    scores = [0.823, 0.817, 0.831, 0.819, 0.825]
    
    fig = go.Figure(data=[
        go.Bar(x=folds, y=scores, text=[f"{s:.3f}" for s in scores], textposition='auto',
               marker_color=COLOR_PRIMARY)
    ])
    fig.update_layout(title_text="K-Fold Macro F1 Scores", height=300, 
                      yaxis=dict(range=[0.75, 0.85]), margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_empty_chart(message="Cần nạp dữ liệu thật để hiển thị"):
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def create_length_stats_chart():
    """
    Thay vì vẽ Histogram giả bằng np.random, ta vẽ biểu đồ Bar Chart 
    cho 4 chỉ số thống kê thật từ notebook 02 (mẫu 500 reviews):
    Mean=22, Median=19, 95th=48, Max=109.
    Tuyệt đối không bịa array.
    """
    metrics = ['Median', 'Mean', '95th Percentile', 'Max']
    values = [19, 22, 48, 109]
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, text=values, textposition='auto',
               marker_color=['#94A3B8', '#3B82F6', '#8B5CF6', '#EF4444'])
    ])
    fig.update_layout(title_text="Thống kê độ dài Review (Token)", height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_confidence_histogram():
    """
    Trả về bảng rỗng với thông báo do chưa có array prediction thực tế.
    """
    try:
        df = pd.read_json("data/test_results.json")
        # Logic vẽ biểu đồ sẽ được cập nhật khi có file thật
        return create_empty_chart("Đang chờ file preds thực tế...")
    except:
        return create_empty_chart("Cần nạp 'test_results.json' vào thư mục data/ để vẽ biểu đồ phân phối Confidence.")

def create_wordcloud(sentiment="Positive"):
    """
    Tạo WordCloud dựa trên văn bản thật từ CSV. Nếu không có file sẽ trả về ảnh trắng kèm text.
    """
    try:
        df = pd.read_csv("data/train_labeled.csv")
        # Label 2 = Positive, 0 = Negative
        label_val = 2 if sentiment == "Positive" else 0
        text_data = " ".join(df[df['label'] == label_val]['Review'].astype(str).tolist())
        
        colormap = "Greens" if sentiment == "Positive" else "Reds"
        wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap).generate(text_data)
        
        return wc.to_image()
    except Exception:
        return None

def create_top_keywords_bar_chart(n=15):
    """
    Vẽ biểu đồ thanh chồng (Stacked Bar) cho Top Keywords.
    """
    try:
        df = pd.read_csv("data/train_labeled.csv")
        stopwords = {'là','và','của','thì','mà','cho','với','như','này','các','những','đã','sẽ','đang','để','nha','nhé','ạ','thấy','cũng','mình','thêm','hơn','có','không','rất'}
        
        word_sentiment = {0: {}, 1: {}, 2: {}}
        all_words_counter = {}
        
        for idx, row in df.iterrows():
            label = row.get('label', -1)
            review = str(row.get('Review', ''))
            if label not in word_sentiment: continue
            
            words = [w for w in review.split() if w not in stopwords and len(w) > 1]
            for w in words:
                word_sentiment[label][w] = word_sentiment[label].get(w, 0) + 1
                all_words_counter[w] = all_words_counter.get(w, 0) + 1
                
        # Get top N words
        top_words = sorted(all_words_counter.items(), key=lambda x: x[1], reverse=True)[:n]
        top_words_list = [w[0] for w in top_words]
        top_words_list.reverse() # Để từ cao nhất lên trên cùng
        
        pos_counts = [word_sentiment[2].get(w, 0) for w in top_words_list]
        neu_counts = [word_sentiment[1].get(w, 0) for w in top_words_list]
        neg_counts = [word_sentiment[0].get(w, 0) for w in top_words_list]
        
        fig = go.Figure(data=[
            go.Bar(name='Negative', y=top_words_list, x=neg_counts, orientation='h', marker_color=COLOR_NEG),
            go.Bar(name='Neutral', y=top_words_list, x=neu_counts, orientation='h', marker_color=COLOR_NEU),
            go.Bar(name='Positive', y=top_words_list, x=pos_counts, orientation='h', marker_color=COLOR_POS)
        ])
        
        fig.update_layout(barmode='stack', title_text=f"Top {n} Keywords Distribution",
                          height=500, margin=dict(l=20, r=20, t=40, b=20),
                          xaxis_title="Frequency", yaxis_title="Keyword")
        return fig
    except Exception as e:
        return create_empty_chart("Cần file train_labeled.csv để vẽ biểu đồ Top Keywords.")

def create_confusion_matrix():
    """
    Vẽ Confusion Matrix bằng số liệu thật từ ảnh user cung cấp.
    """
    z = [[181, 44, 6],
         [28, 323, 65],
         [1, 123, 1375]]
    
    # Tính phần trăm theo hàng (Recall)
    z_text = []
    for row in z:
        total = sum(row)
        z_text.append([f"{val}<br>({val/total:.0%})" for val in row])
        
    x = ['Negative', 'Neutral', 'Positive']
    y = ['Negative', 'Neutral', 'Positive']
    
    fig = go.Figure(data=go.Heatmap(
                   z=z,
                   x=x,
                   y=y,
                   text=z_text,
                   texttemplate="%{text}",
                   hoverinfo="text",
                   colorscale='Blues'))
    
    fig.update_layout(
        title="Confusion Matrix (Test set)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Đảo trục Y để khớp với ảnh (Negative nằm trên cùng)
    fig.update_yaxes(autorange="reversed")
    return fig
