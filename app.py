from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# Load the data and model
df = pd.read_csv("spotify_millsongdata.csv")
df.drop("link", axis=1, inplace=True)
df = df[0:20000]
df.drop_duplicates(subset="song", inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={"artist": 'Artist', 'song': 'Song', 'text': 'Lyrics'}, inplace=True)
df['Combined_features'] = df['Artist'] + " " + df['Song'] + " " + df['Lyrics']

# Path to the directory containing the model files
model_path = os.path.dirname(os.path.abspath(__file__))

# Load the vectorizer and cosine similarity matrix
tfv = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))
cosine_sim = joblib.load(os.path.join(model_path, 'cosine_similarity_matrix.pkl'))

# Create a Flask app
app = Flask(__name__)

# Function to recommend songs
def recommended_songs(song_user_likes):
    if song_user_likes not in df['Song'].values:
        return ["Song not found in the dataset."]
    
    song_index = df[df.Song == song_user_likes].index[0]
    similar_songs = list(enumerate(cosine_sim[song_index]))
    similar_song_sorted = sorted(similar_songs, key=lambda x: x[1], reverse=True)
    
    recommended = []
    for song in similar_song_sorted[1:11]:
        similar_song = df.iloc[song[0]]["Song"]
        recommended.append(similar_song)
    
    return recommended

# Define routes for the web app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song_user_likes = request.form['song']
    recommendations = recommended_songs(song_user_likes)
    return render_template('recommendations.html', recommendations=recommendations, song=song_user_likes)

if __name__ == '__main__':
    app.run(debug=True)
