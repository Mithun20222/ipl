import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv("matches.csv")

# Data cleaning - remove rows with missing values
df = df.dropna(subset=['team1', 'team2', 'venue', 'toss_winner', 'winner', 'season', 'city'])

# Feature engineering
df['matchup'] = df['team1'] + "_vs_" + df['team2']
df['is_home_team'] = df.apply(lambda row: int(row['city'].lower() in row['team1'].lower() or 
                                             row['city'].lower() in row['team2'].lower()), axis=1)

# Encode targets
le_toss = LabelEncoder()
le_match = LabelEncoder()
df['toss_winner_encoded'] = le_toss.fit_transform(df['toss_winner'])
df['winner_encoded'] = le_match.fit_transform(df['winner'])

# Features and targets - removed 'toss_decision'
features = ['team1', 'team2', 'venue', 'season', 'match_type', 'city', 'matchup', 'is_home_team']
X = df[features]
y_toss = df['toss_winner_encoded']
y_match = df['winner_encoded']

# Train-test split
X_train, X_test, y_toss_train, y_toss_test, y_match_train, y_match_test = train_test_split(
    X, y_toss, y_match, test_size=0.2, random_state=42
)

# Preprocessing - removed 'toss_decision' from categorical features
categorical_features = ['team1', 'team2', 'venue', 'season', 'match_type', 'city', 'matchup']
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Models
xgb_toss = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_match = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Pipelines
pipeline_toss = Pipeline([('encoder', encoder), ('classifier', xgb_toss)])
pipeline_match = Pipeline([('encoder', encoder), ('classifier', xgb_match)])

# Hyperparameters
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [5],
    'classifier__learning_rate': [0.1],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.8]
}

# Training
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Training toss model...")
grid_search_toss = GridSearchCV(pipeline_toss, param_grid, cv=skf, n_jobs=-1)
grid_search_toss.fit(X_train, y_toss_train)

print("Training match model...")
grid_search_match = GridSearchCV(pipeline_match, param_grid, cv=skf, n_jobs=-1)
grid_search_match.fit(X_train, y_match_train)

# Save models using pickle
with open('toss_model.pkl', 'wb') as f:
    pickle.dump(grid_search_toss, f)
    
with open('match_model.pkl', 'wb') as f:
    pickle.dump(grid_search_match, f)
    
with open('le_toss.pkl', 'wb') as f:
    pickle.dump(le_toss, f)
    
with open('le_match.pkl', 'wb') as f:
    pickle.dump(le_match, f)

print("Models saved successfully using pickle!")