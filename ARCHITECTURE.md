# Steam Analytics Dashboard - Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │   Dashboard Page     │    │  Recommendations Page    │  │
│  │   - Plotly Charts    │    │   - Search Interface     │  │
│  │   - Statistics Cards │    │   - Game Details         │  │
│  │   - Data Viz         │    │   - ML Recommendations   │  │
│  └──────────────────────┘    └──────────────────────────┘  │
└───────────────────┬──────────────────────┬──────────────────┘
                    │                       │
                    │   HTTP Requests       │
                    ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Routes:                                              │  │
│  │  - GET  /                    → Dashboard              │  │
│  │  - GET  /recommendations     → Recommendations UI     │  │
│  │  - POST /api/search_games    → Search functionality   │  │
│  │  - POST /api/get_game_info   → Game details          │  │
│  │  - POST /api/get_recommendations → ML predictions     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ML Engine (scikit-learn):                            │  │
│  │  1. prepare_recommendation_data()                     │  │
│  │     - One-hot encode Tag1, Tag2                       │  │
│  │     - Add normalized scores                           │  │
│  │     - Create feature matrix                           │  │
│  │  2. cosine_similarity(features)                       │  │
│  │     - Compute similarity matrix                       │  │
│  │  3. Get top N similar games                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data Processing (pandas):                            │  │
│  │  - DataFrame operations                               │  │
│  │  - Grouping & aggregation                             │  │
│  │  - Statistical analysis                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Visualization (Plotly):                              │  │
│  │  - Price distribution histogram                       │  │
│  │  - Pie charts for tags                                │  │
│  │  - Bar charts for reviews                             │  │
│  │  - Time series for yearly trends                      │  │
│  │  - Scatter plots for correlations                     │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    │   SQL Queries (SQLAlchemy)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    MySQL Database                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Database: steam_data                                 │  │
│  │  Table: steamout                                      │  │
│  │  Columns:                                             │  │
│  │    - Title, Price, Discount, Image                    │  │
│  │    - ReviewNum, Reviews, Tag1, Tag2                   │  │
│  │    - Year, RAM, Size                                  │  │
│  │    - Normalized, Positivity_Percentage                │  │
│  │    - Total_revenue, DLC                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow: Recommendation System

```
User Action                ML Processing                  Result
──────────                 ─────────────                  ──────

1. User searches         → Search API called
   "Portal"                 /api/search_games
                         
2. User selects game     → Get game info API
   "Portal 2"               /api/get_game_info
                            - Fetch from database
                            - Return game details
                         
3. Display game info     → Recommendations API
                            /api/get_recommendations
                            
4. ML Engine:
   ┌─────────────────────────────────────────┐
   │ a) Get game index in dataframe          │
   │    game_idx = df[df['Title']=='Portal'] │
   │                                          │
   │ b) Get similarity scores                │
   │    sim_scores = similarity_matrix[idx]  │
   │                                          │
   │ c) Sort by similarity                   │
   │    sorted_scores = sort(sim_scores)     │
   │                                          │
   │ d) Get top 5 games (excluding self)     │
   │    recommendations = top_5_indices      │
   │                                          │
   │ e) Return with similarity %             │
   │    [{game, similarity}, ...]            │
   └─────────────────────────────────────────┘
                            
5. Display results       ← Return JSON with:
   - Similar games         - Game titles
   - Similarity %          - Prices, images, tags
   - Game details          - Similarity scores
```

## Feature Matrix Construction

```
Game Data          Feature Engineering           Similarity Matrix
─────────         ──────────────────             ────────────────

Title: Portal     Tag1: Puzzle                   Game1  Game2  Game3
Tag1: Puzzle      → One-hot: [1,0,0,...]        ┌─────┬─────┬─────┐
Tag2: Sci-fi      Tag2: Sci-fi         Game1    │ 1.0 │ 0.8 │ 0.3 │
Normalized: 0.85  → One-hot: [0,1,0,...]        ├─────┼─────┼─────┤
                  Normalized: 0.85     Game2    │ 0.8 │ 1.0 │ 0.5 │
                  → Keep as is                  ├─────┼─────┼─────┤
                  Positivity: 95%      Game3    │ 0.3 │ 0.5 │ 1.0 │
                  → Keep as is                  └─────┴─────┴─────┘
                                       
Combined Feature Vector:                Cosine Similarity:
[1,0,0,...,0,1,0,...,0.85,95]         similarity = A·B / (||A|| ||B||)

                                       Higher score = More similar
```

## Technology Stack

```
Frontend Layer:
├── HTML5 (Structure)
├── CSS3 (Styling with gradients, animations)
├── JavaScript (Interactivity)
└── Plotly.js (Chart rendering)

Backend Layer:
├── Flask (Web framework)
├── SQLAlchemy (ORM)
├── pandas (Data manipulation)
├── NumPy (Numerical operations)
├── scikit-learn (ML algorithms)
└── Plotly (Visualization generation)

Data Layer:
└── MySQL (Steam games database)
```

## Key Files & Responsibilities

```
app.py
├── Database connection setup
├── Data loading and preprocessing
├── ML model initialization (similarity matrix)
├── Route handlers for pages
├── API endpoints for AJAX calls
├── Data visualization creation (Plotly)
└── Recommendation algorithm implementation

templates/home.html
├── Dashboard layout
├── Statistics cards
├── Chart containers
└── Plotly chart rendering

templates/recommendations.html
├── Search interface
├── Game info display
├── Recommendations grid
└── API integration (fetch calls)

static/style.css
├── Dark theme variables
├── Component styling
├── Animations and transitions
├── Responsive design
└── Glassmorphism effects
```

## Deployment Checklist

- [x] Flask app created
- [x] ML recommendation engine implemented
- [x] Dashboard with 6 visualizations
- [x] Recommendation page with search
- [x] REST API endpoints
- [x] Premium UI with animations
- [x] Responsive design
- [x] Documentation (README, QUICKSTART)
- [ ] Install dependencies
- [ ] Configure database credentials
- [ ] Test MySQL connection
- [ ] Run application
- [ ] Test all features
