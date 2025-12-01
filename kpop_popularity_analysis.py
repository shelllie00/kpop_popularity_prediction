import os
from dotenv import load_dotenv
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


load_dotenv()
# Spotify API credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

# Function to get Spotify API token
def get_spotify_token():
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()['access_token']

# Function to get top tracks from Spotify Chart
def get_top_tracks(token):
    # Replace with the actual Spotify Chart API or playlist URL
    # Use "KPOP ON!" playlist
    playlist_id = '2Yh8Eqy57CRyB0VPwLVxNO'
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(url, headers=headers)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
    tracks = response.json()['items']
    track_data = []
    for track in tracks:
        track_info = track['track']
        track_data.append({
            'track_id': track_info['id'],
            'name': track_info['name'],
            'popularity': track_info['popularity'],
            'release_date': track_info['album']['release_date']
        })
    return pd.DataFrame(track_data)

# Function to get audio features for tracks
def get_audio_features(token, track_ids):
    url = 'https://api.spotify.com/v1/audio-features'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    audio_features = []
    for i in range(0, len(track_ids), 100):  # Spotify API allows max 100 IDs per request
        ids = ','.join(track_ids[i:i+100])
        response = requests.get(f'{url}?ids={ids}', headers=headers)
        audio_features.extend(response.json()['audio_features'])
    return pd.DataFrame(audio_features)

# Main script
if __name__ == '__main__':
    # Step 1: Authenticate and get token
    token = get_spotify_token()
    

    # Step 2: Get top tracks
    top_tracks = get_top_tracks(token)

    # Step 3: Get audio features
    audio_features = get_audio_features(token, top_tracks['track_id'].tolist())

    # Step 4: Merge data
    data = top_tracks.merge(audio_features, left_on='track_id', right_on='id')

    # Step 5: Preprocess data
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness']
    target = 'popularity'
    X = data[features]
    y = data[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Baseline model - Simple Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))

    # Step 7: Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))