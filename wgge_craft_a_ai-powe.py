# wgge_craft_a_ai-powe.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Game Data
games_data = [
    {'game_id': 1, 'genre': 'action', 'platform': 'pc', 'rating': 4.5},
    {'game_id': 2, 'genre': 'adventure', 'platform': 'console', 'rating': 4.2},
    {'game_id': 3, 'genre': 'rpg', 'platform': 'pc', 'rating': 4.8},
    {'game_id': 4, 'genre': 'sports', 'platform': 'console', 'rating': 4.0},
    {'game_id': 5, 'genre': 'strategy', 'platform': 'pc', 'rating': 4.6},
]

# Create a DataFrame
games_df = pd.DataFrame(games_data)

# Define Features and Target
X = games_df[['genre', 'platform']]
y = games_df['rating']

# One-Hot Encoding for categorical features
X_onehot = pd.get_dummies(X, columns=['genre', 'platform'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a function to track new games
def track_game(game_id, genre, platform, rating):
    new_game = pd.DataFrame({'game_id': [game_id], 'genre': [genre], 'platform': [platform], 'rating': [rating]})
    new_game_onehot = pd.get_dummies(new_game, columns=['genre', 'platform'])
    new_game_pred = clf.predict(new_game_onehot)
    print(f"Predicted Rating for Game {game_id}: {new_game_pred[0]}")

# Track a new game
track_game(6, 'action', 'pc', 4.7)