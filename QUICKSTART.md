# Quick Start Guide - Steam Analytics Dashboard

## What You Have

A complete Flask web application with:

### ✅ Home Page (Dashboard)
- **6 Interactive Charts**: Price distribution, top tags, reviews breakdown, yearly trends, discounts, and price-review correlation
- **4 Statistics Cards**: Total games, average price, discount, and reviews
- **Premium UI**: Dark theme with gradients, animations, and modern design

### ✅ Recommendations Page  
- **ML-Powered Search**: Type to search games with autocomplete
- **Game Details**: Shows selected game info with image, price, tags, reviews
- **Smart Recommendations**: Uses scikit-learn cosine similarity to find 5 similar games
- **Similarity Scores**: Each recommendation shows match percentage

## How the ML Recommendation Works

The system uses **scikit-learn's cosine_similarity** function:

1. **Feature Engineering** (in `app.py`):
   ```python
   - One-hot encode Tag1 and Tag2
   - Add normalized scores
   - Add positivity percentages
   - Create feature matrix
   ```

2. **Similarity Calculation**:
   ```python
   similarity_matrix = cosine_similarity(features)
   ```

3. **Get Recommendations**:
   - Find index of selected game
   - Get similarity scores for all other games
   - Return top 5 most similar games

## Running the Application

### Option 1: Using the batch file
Double-click `run.bat`

### Option 2: Command line
```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## Before Running - Important!

Make sure:
1. ✅ MySQL is running
2. ✅ Database `steam_data` exists
3. ✅ Table `steamout` has data
4. ✅ Database credentials in `app.py` are correct (lines 15-18)

## File Structure

```
Dashboard/
│
├── app.py                      # Main Flask app with ML logic
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows startup script
├── README.md                   # Full documentation
├── QUICKSTART.md              # This file
│
├── templates/
│   ├── home.html              # Dashboard with charts
│   └── recommendations.html    # ML recommendation page
│
└── static/
    └── style.css              # Premium modern styling
```

## API Endpoints

The app provides these REST APIs:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Dashboard page |
| `/recommendations` | GET | Recommendations page |
| `/api/search_games` | POST | Search games by title |
| `/api/get_game_info` | POST | Get game details |
| `/api/get_recommendations` | POST | Get ML recommendations |

## Customization

### Change Number of Recommendations
In `app.py`, line in the `/api/get_recommendations` route:
```python
n_recommendations = data.get('n', 5)  # Change 5 to desired number
```

### Modify Database Connection
Edit lines 15-18 in `app.py`:
```python
db_username = 'root'
db_password = 'root'
db_host = 'localhost'
db_name = 'steam_data'
```

### Adjust Similarity Algorithm
In the `prepare_recommendation_data()` function, you can:
- Add more features (Price, ReviewNum, etc.)
- Change the similarity metric
- Weight different features

## Troubleshooting

**Error: Can't connect to database**
- Check MySQL is running
- Verify credentials in app.py
- Ensure database and table exist

**Error: Module not found**
- Run: `pip install -r requirements.txt`

**Charts not showing**
- Check browser console for errors
- Verify Plotly CDN is accessible
- Check data is loading from database

**No recommendations appearing**
- Verify games have Tag1 and Tag2 data
- Check browser network tab for API errors
- Ensure scikit-learn is installed correctly

## Features Highlights

### Visual Design
- ✨ Modern dark theme with gradients
- 🎨 Animated cards and smooth transitions
- 📱 Fully responsive design
- 🎯 Premium aesthetics with glassmorphism

### Machine Learning
- 🤖 Content-based filtering
- 📊 Feature engineering with one-hot encoding
- 🎯 Cosine similarity computation
- 📈 Real-time similarity scoring

### Data Analysis
- 📊 6 different visualization types
- 📈 Interactive Plotly charts
- 🔍 Comprehensive game statistics
- 📉 Trend analysis over time

## Next Steps

After the app is running:
1. Visit the Dashboard to explore your data
2. Navigate to Recommendations page
3. Search for a game you like
4. View ML-generated similar games
5. Check the similarity percentage for each recommendation

Enjoy your Steam Analytics Dashboard! 🎮
