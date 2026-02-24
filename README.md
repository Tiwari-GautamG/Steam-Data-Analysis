# Steam Games Analytics Dashboard

A modern Flask web application featuring interactive data visualizations and a machine learning-powered game recommendation system.

## Features

### 🎮 Dashboard Page
- **Interactive Visualizations** using Plotly
  - Price distribution analysis
  - Top game genres/tags by revenue
  - Review sentiment breakdown
  - Annual release trends
  - Discount patterns
  - Price vs Review correlation
  
- **Key Statistics Cards**
  - Total games count
  - Average price
  - Average discount
  - Total reviews

### 🤖 Recommendation System Page
- **ML-Powered Recommendations** using scikit-learn
  - Content-based filtering using cosine similarity
  - Tag-based similarity matching
  - Normalized scoring integration
  - Real-time search with autocomplete
  
- **Similarity Algorithm**
  - One-hot encoding for game tags (Tag1, Tag2)
  - Numerical feature normalization
  - Cosine similarity matrix computation
  - Top 5 similar games with match percentage

## Technology Stack

### Backend (Python)
- **Flask** - Web framework
- **SQLAlchemy** - Database ORM
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning (cosine similarity)
- **numpy** - Numerical computations
- **Plotly** - Interactive visualizations

### Frontend
- **HTML5** - Structure
- **CSS3** - Modern styling with gradients and animations
- **JavaScript** - Interactivity and API calls
- **Plotly.js** - Chart rendering

### Database
- **MySQL** - Steam games data storage

## Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure database connection**
Edit `app.py` and update the database credentials:
```python
db_username = 'your_username'
db_password = 'your_password'
db_host = 'localhost'
db_name = 'steam_data'
```

3. **Ensure your database has the required table**
The app expects a table named `steamout` with these columns:
- Title (varchar)
- Price (int)
- Discount (text)
- Image (text)
- ReviewNum (int)
- Reviews (text)
- Tag1 (text)
- Tag2 (text)
- Year (datetime)
- RAM (text)
- Size (text)
- And additional analysis columns (Normalized, Positivity_Percentage, Total_revenue, DLC)

## Running the Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Project Structure

```
Dashboard/
│
├── app.py                  # Flask application & ML logic
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── templates/
│   ├── home.html          # Dashboard page
│   └── recommendations.html # Recommendations page
│
└── static/
    └── style.css          # Modern CSS styling
```

## How the Recommendation System Works

1. **Feature Engineering**
   - Creates one-hot encoded vectors for Tag1 and Tag2
   - Includes normalized game scores and positivity percentages
   - Combines all features into a single matrix

2. **Similarity Calculation**
   - Uses scikit-learn's `cosine_similarity` function
   - Computes similarity between all games
   - Returns top 5 most similar games

3. **User Flow**
   - User searches for a game
   - System displays game information
   - ML algorithm finds similar games
   - Shows recommendations with similarity percentage

## API Endpoints

- `GET /` - Dashboard page
- `GET /recommendations` - Recommendations page
- `POST /api/search_games` - Search games by title
- `POST /api/get_game_info` - Get detailed game information
- `POST /api/get_recommendations` - Get ML-powered recommendations

## Future Enhancements

- Add user ratings for collaborative filtering
- Implement hybrid recommendation system
- Add more visualization types (heatmaps, treemaps)
- Include game comparison features
- Export analysis reports
- User preference learning

## Credits

Built with Flask, scikit-learn, and modern web technologies.
Based on Steam games dataset with comprehensive game metadata.
