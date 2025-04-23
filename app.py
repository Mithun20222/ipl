from flask import Flask, render_template, request
import pandas as pd
import pickle
import traceback

app = Flask(__name__)

# Load models with verification
try:
    print("Loading models...")
    with open('toss_model.pkl', 'rb') as f:
        grid_search_toss = pickle.load(f)
    
    with open('match_model.pkl', 'rb') as f:
        grid_search_match = pickle.load(f)
        
    with open('le_toss.pkl', 'rb') as f:
        le_toss = pickle.load(f)
        
    with open('le_match.pkl', 'rb') as f:
        le_match = pickle.load(f)
    
    # Verify models
    if not hasattr(grid_search_toss, 'predict'):
        raise ValueError("Toss model not properly loaded")
    if not hasattr(grid_search_match, 'predict'):
        raise ValueError("Match model not properly loaded")
        
    models_loaded = True
    print("Models loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    traceback.print_exc()
    models_loaded = False
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if not models_loaded:
            return render_template('index.html', 
                               error="Models not loaded. Please try again later.")
        
        try:
            # Get all required form data
            team1 = request.form['team1'].strip()
            team2 = request.form['team2'].strip()
            venue = request.form['venue'].strip()
            season = request.form['season'].strip()
            match_type = request.form['match_type'].strip()
            city = request.form['city'].strip()
            
            # Create feature dictionary with EXACTLY the same structure as training
            input_data = {
                'team1': [team1],
                'team2': [team2],
                'venue': [venue],
                'season': [season],
                'match_type': [match_type],
                'city': [city],
                'matchup': [f"{team1}_vs_{team2}"],
                'is_home_team': [int(city.lower() in team1.lower())]  # Simplified home team logic
            }
            
            # Create DataFrame
            new_data = pd.DataFrame(input_data)
            
            # Debug: Print the input data structure
            print("Input data columns:", new_data.columns.tolist())
            print("Input data values:", new_data.to_dict('records'))
            
            # Make predictions
            toss_pred = grid_search_toss.predict(new_data)
            match_pred = grid_search_match.predict(new_data)
            
            # Decode predictions
            toss_winner = le_toss.inverse_transform(toss_pred)[0]
            match_winner = le_match.inverse_transform(match_pred)[0]
            
            return render_template('index.html',
                                toss_winner=toss_winner,
                                match_winner=match_winner,
                                success=True)
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return render_template('index.html', error=error_msg)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)