# Spotify-Mood-Mapper
An AI-powered music explorer that uses K-Means clustering to group Spotify tracks by audi features rather than traditional genres.
The Concept
Traditional genres like Rock and Pop are often too broad. This app uses Machine Learning to analyze the technical DNA of your music—things like danceability, energy, and valence—to create Mood Clusters.

How it Works
Data Input: Upload a CSV of your Spotify library.

Feature Scaling: The AI normalizes audio features so that "Tempo" (BPM) doesn't outweigh "Acousticness."

K-Means Clustering: The algorithm finds hidden patterns in the data to group similar-sounding tracks together.

Vibe Selection: You pick a Mood Group, and the AI generates a fresh 10-song playlist for you.

Tech Stack
Python

Streamlit (UI)

Scikit-Learn (K-Means Clustering)

Pandas (Data Manipulation)
