from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import plotly.io as pio
import json
import os
from datetime import datetime

app = Flask(__name__)
pio.json.config.default_engine = 'json'
DATABASE_URL = os.environ.get("DATABASE_URL").replace("postgres://", "postgresql://")


def write_debug_log(message, data=None, hypothesis_id=None):
    log_entry = {
        "timestamp": int(datetime.now().timestamp() * 1000),
        "location": "app.py:database_loading",
        "message": message,
        "data": data or {},
        "sessionId": "debug-session",
        "runId": "initial_run",
        "hypothesisId": hypothesis_id
    }
    with open("debug.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

engine = create_engine(DATABASE_URL)

query = 'SELECT * FROM steamout'
df = pd.read_sql(query, engine)
write_debug_log("Data loaded from database", {"rows_loaded": len(df), "columns": list(df.columns)}, "hypothesis_1")
print(f"DEBUG: Columns from DB: {df.columns.tolist()}")

# Data inspection and cleaning
print(f"\n{'='*50}")
print(f"DATA LOADED: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst few rows of Price column:")
print(df['Price'].head(10))
print(f"\nFirst few rows of Discount column:")
print(df['Discount'].head(10))
print(f"\nFirst few rows of Year column (raw):")
print(df['Year'].head(10))
print(f"\nNull counts:")
print(df.isnull().sum())
print(f"{'='*50}\n")

# Clean data
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
df['ReviewNum'] = pd.to_numeric(df['ReviewNum'], errors='coerce').fillna(0)

# Preprocess data for recommendations
def prepare_recommendation_data():
    """Prepare feature matrix for recommendation system"""
    rec_df = df.copy()
    
    tag1_encoded = pd.get_dummies(rec_df['Tag1'], prefix='tag1')
    tag2_encoded = pd.get_dummies(rec_df['Tag2'], prefix='tag2')
    
    features = pd.concat([tag1_encoded, tag2_encoded], axis=1)
    
    if 'Normalized' in rec_df.columns:
        features['normalized_score'] = rec_df['Normalized']
    if 'Positivity_Percentage' in rec_df.columns:
        features['positivity'] = rec_df['Positivity_Percentage']
    
    features = features.fillna(0).astype(np.float32)
    sparse_features = sparse.csr_matrix(features)
    nn_model = NearestNeighbors(
        metric='cosine',
        algorithm='brute'
    )
    nn_model.fit(sparse_features)

    return sparse_features, nn_model

sparse_features, nn_model = prepare_recommendation_data()

@app.route('/')
def home():
    write_debug_log("Home route called", {"original_df_rows": len(df)}, "hypothesis_5")

    df_clean = df.copy()
    write_debug_log("Data copy created for cleaning", {"df_clean_rows": len(df_clean)}, "hypothesis_2")
    
    if 'Price' in df_clean.columns:
        df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce').fillna(0)

    if 'Discount' in df_clean.columns:
        discount_cleaned = df_clean['Discount'].astype(str).str.replace('%', '').str.strip()
        df_clean['Discount_Numeric'] = pd.to_numeric(discount_cleaned, errors='coerce').fillna(0)
        if df_clean['Discount_Numeric'].max() <= 1.0:
            df_clean['Discount_Numeric'] = df_clean['Discount_Numeric'] * 100
    else:
        df_clean['Discount_Numeric'] = 0

    if 'Year' in df_clean.columns:
        df_clean['Year_Value'] = pd.to_numeric(df_clean['Year'], errors='coerce').fillna(0)

    price_data = df_clean[(df_clean['Price'] > 0) & (df_clean['Price'] <= 5000)].copy()
    print(f"DEBUG: price_data rows: {len(price_data)}")
    fig_price = go.Figure(go.Histogram(
        x=price_data['Price'].tolist(),
        nbinsx=50,
        marker_color='#636EFA'
    ))
    fig_price.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(17,24,39,1)',
        font=dict(color='white'),
        height=400,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Price (₹)', range=[0,5000]),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Number of Games')
    )

    if 'Tag1' in df_clean.columns:
        top_tags = df_clean['Tag1'].fillna('Unknown').value_counts().head(10)
        print(f"DEBUG: top_tags count: {len(top_tags)}")
        fig_tags = go.Figure(data=[go.Pie(
            labels=top_tags.index,
            values=top_tags.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Prism)
        )])
    else:
        fig_tags = go.Figure()

    fig_tags.update_layout(
        title='Top 10 Game Genres',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(17,24,39,1)',
        font=dict(color='white'),
        height=400
    )
    
    df_clean['ReviewNum'] = pd.to_numeric(df_clean['ReviewNum'], errors='coerce').fillna(0)
    top_reviewed = df_clean.nlargest(15, 'ReviewNum')[['Title', 'ReviewNum']].sort_values('ReviewNum', ascending=True)
    print(f"DEBUG: top_reviewed rows: {len(top_reviewed)}")
    
    fig_reviews = go.Figure(data=[go.Bar(
        x=top_reviewed['ReviewNum'],
        y=top_reviewed['Title'],
        orientation='h',
        marker_color='#10B981'
    )])
    fig_reviews.update_layout(
        title='Top 15 Most Reviewed Games',
        xaxis_title='Reviews',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(17,24,39,1)',
        font=dict(color='white'),
        height=400,
        yaxis=dict(tickfont=dict(size=10))
    )
    
    if 'Year_Value' in df_clean.columns:
        valid_years = df_clean[df_clean['Year_Value'] >= 1990]['Year_Value'].astype(int)
        year_counts = valid_years.value_counts().sort_index()
        print(f"DEBUG: year_counts: {len(year_counts)} unique years, range {year_counts.index.min()}-{year_counts.index.max()}")
        
        fig_year = go.Figure(data=[go.Scatter(
            x=[int(y) for y in year_counts.index],
            y=[int(v) for v in year_counts.values],
            mode='lines+markers',
            line=dict(color='#8B5CF6', width=3),
            marker=dict(size=6, color='#8B5CF6'),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)'
        )])
        fig_year.update_layout(
            title='Game Releases Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Games',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(17,24,39,1)',
            font=dict(color='white'),
            height=400,
            xaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                dtick=5, tickangle=-45
            ),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
    else:
        fig_year = go.Figure()


    discount_data = df_clean[
        (df_clean['Discount_Numeric'] > 0) & 
        (df_clean['Discount_Numeric'] < 100)
    ].copy()
    print(f"DEBUG: discount_data rows (0-100%): {len(discount_data)}")
    fig_discount = go.Figure(data=[go.Histogram(
        x=discount_data['Discount_Numeric'].to_list(),
        xbins=dict(start=0, end=100, size=5),
        marker_color='#F59E0B',
        marker_line=dict(color='rgba(0,0,0,0.3)', width=0.5)
    )])
    fig_discount.update_layout(
        title='Discount Distribution (Games with Discounts)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(17,24,39,1)',
        font=dict(color='white'),
        height=400,
        showlegend=False,
        bargap=0.05,
        xaxis=dict(
            title='Discount (%)', 
            range=[0, 100],
            dtick=10
        ),
        yaxis=dict(title='Number of Games')
    )

    scatter_data = df_clean[
        (df_clean['Price'] > 0) & 
        (df_clean['Price'] <= 5000) &
        (df_clean['ReviewNum'] > 0)
    ].copy()
    print(f"DEBUG: scatter_data rows: {len(scatter_data)}")
    
    fig_correlation = go.Figure(go.Scatter(
        x=scatter_data['Price'].tolist(),
        y=scatter_data['ReviewNum'].tolist(),
        mode='markers',
        opacity=0.4,
        text=scatter_data['Title'].tolist(),
        marker=dict(color='#3B82F6', size=4)
    ))
    fig_correlation.update_layout(
        title='Price vs. Reviews',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(17,24,39,1)',
        font=dict(color='white'),
        height=400,
        xaxis=dict(
            title='Price (₹)',
            showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0,5000]
        ),
        yaxis=dict(
            title='Number of Reviews',
            type='log', showgrid=True, gridcolor='rgba(255,255,255,0.1)'
        )
    )
    
    write_debug_log("Converting plots to JSON", {"charts_to_convert": 6}, "hypothesis_2")
    def fig_to_json(fig):
        fig_dict = fig.to_dict()
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj
        return json.dumps(convert(fig_dict))

    graphs = {
    'price': fig_to_json(fig_price),
    'tags': fig_to_json(fig_tags),
    'reviews': fig_to_json(fig_reviews),
    'year': fig_to_json(fig_year),
    'discount': fig_to_json(fig_discount),
    'correlation': fig_to_json(fig_correlation)
    }
    write_debug_log("Charts converted to JSON", {"graphs_keys": list(graphs.keys())}, "hypothesis_2")
    
    avg_discount = df_clean['Discount_Numeric'].mean() if 'Discount_Numeric' in df_clean.columns else 0

    stats = {
        'total_games': f"{len(df):,}",
        'avg_price': f"₹{df['Price'].mean():.2f}",
        'avg_discount': f"{avg_discount:.1f}%",
        'total_reviews': f"{df['ReviewNum'].sum():,}"
    }

    write_debug_log("Template rendering", {
        "graphs_keys": list(graphs.keys()),
        "stats_keys": list(stats.keys()),
        "total_games_stat": stats['total_games']
    }, "hypothesis_3")

    return render_template('home.html', graphs=graphs, stats=stats)

@app.route('/recommendations')
def recommendations():
    games = sorted(df['Title'].unique())
    return render_template('recommendations.html', games=games)

@app.route('/api/search_games', methods=['POST'])
def search_games():
    data = request.get_json()
    query = data.get('query', '').lower()
    if not query:
        return jsonify({'games': []})
    filtered_games = df[df['Title'].str.lower().str.contains(query)]['Title'].tolist()
    
    return jsonify({'games': sorted(filtered_games[:20])})


@app.route('/api/get_game_info', methods=['POST'])
def get_game_info():
    data = request.get_json()
    game_title = data.get('title', '')
    
    if not game_title:
        return jsonify({'error': 'No title provided'}), 400
    game_data = df[df['Title'] == game_title]
    
    if game_data.empty:
        return jsonify({'error': 'Game not found'}), 404
    
    game_info = game_data.iloc[0].to_dict()

    for key, value in game_info.items():
        if isinstance(value, (np.integer, np.floating)):
            game_info[key] = value.item()
        elif pd.isna(value):
            game_info[key] = None
    
    return jsonify(game_info)


@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    game_title = data.get('title', '')
    n_recommendations = data.get('n', 5)
    
    if not game_title:
        return jsonify({'error': 'No title provided'}), 400

    try:
        game_idx = df[df['Title'] == game_title].index[0]
    except IndexError:
        return jsonify({'error': 'Game not found'}), 404
    
    
    distances, indices = nn_model.kneighbors(
        sparse_features[game_idx],
        n_neighbors=n_recommendations + 1
    )

    game_indices = indices.flatten()[1:]
    similarity_scores = 1 - distances.flatten()[1:] 

    recommended_games = df.iloc[game_indices][['Title', 'Price', 'Tag1', 'Tag2', 'Image', 'Reviews']].copy()
    recommended_games['similarity'] = similarity_scores

    recommendations = recommended_games.to_dict('records')
    
    for rec in recommendations:
        for key, value in rec.items():
            if isinstance(value, (np.integer, np.floating)):
                rec[key] = value.item()
            elif pd.isna(value):
                rec[key] = None
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
