import pandas as pd
import streamlit as st 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Spotify AI Mood Mapper", page_icon=":musical_note:", layout="wide")

@st.cache_data
def run_clustering(data, features, num_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    data['mood_cluster'] = kmeans.fit_predict(scaled_features)
    return data, kmeans

st.title("Spotify AI Mood Mapper :musical_note:")
st.markdown("This App Uses K-Means Clustering to Group Music By Vibe Rather Than Genre ")

uploaded_file = st.file_uploader("Upload Your Spotify CSV Dataset Here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    df_clean = df.dropna(subset=features)

    st.sidebar.header("AI Settings")
    cluster_count = st.sidebar.slider("How many mood groups?", 3, 10, 5)

    df_result, model = run_clustering(df_clean, features, cluster_count)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select a Mood")
        cluster_choice = st.selectbox("Which vibe are you looking for?",
                                       options=range(cluster_count),
                                       format_func=lambda x: f"Mood {x}")
        if st.button("Generate Playlist"):
            playlist = df_result[df_result['mood_cluster'] == cluster_choice].sample(min(10, len(df_result)))
            st.session_state['playlist'] = playlist

    # Column 2: Display
    with col2:
        st.subheader("Your Generated Playlist")
        if 'playlist' in st.session_state:
            display_cols = ['track_name', 'artists', 'album_name', 'danceability', 'energy', 'valence', 'tempo']
            st.table(st.session_state['playlist'][display_cols])
        else:
            st.info("Select a mood and click 'Generate Playlist' to see your songs!")


else:
    st.info("Please upload a Spotify CSV dataset to get started")