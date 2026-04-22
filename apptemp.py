import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dataset 3 Dashboard",
    layout="wide"
)

st.title("Spotify Playlist Popularity Analysis")
st.markdown("Track Popularity Analysis using Dataset 3")

df = pd.read_csv("data/processed/dataset3_cleaned.csv")

st.sidebar.header("Filters")

selected_genre = st.sidebar.selectbox(
    "Select Playlist Genre",
    ["All"] + sorted(df["playlist_genre"].dropna().unique().tolist())
)

popularity_range = st.sidebar.slider(
    "Track Popularity Range",
    int(df["track_popularity"].min()),
    int(df["track_popularity"].max()),
    (
        int(df["track_popularity"].min()),
        int(df["track_popularity"].max())
    )
)

filtered_df = df.copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["playlist_genre"] == selected_genre
    ]

filtered_df = filtered_df[
    (filtered_df["track_popularity"] >= popularity_range[0]) &
    (filtered_df["track_popularity"] <= popularity_range[1])
]

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

st.subheader("Quick Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Tracks", len(filtered_df))
col2.metric(
    "Average Popularity",
    round(filtered_df["track_popularity"].mean(), 2)
)
col3.metric(
    "Total Genres",
    filtered_df["playlist_genre"].nunique()
)

st.subheader("Statistical Summary")

mean_pop = round(filtered_df["track_popularity"].mean(), 2)
median_pop = round(filtered_df["track_popularity"].median(), 2)
std_pop = round(filtered_df["track_popularity"].std(), 2)
common_genre = filtered_df["playlist_genre"].mode()[0]

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Mean Popularity", mean_pop)
col2.metric("Median Popularity", median_pop)
col3.metric("Std Deviation", std_pop)
col4.metric("Most Common Genre", common_genre)

st.subheader("Track Popularity Distribution")

fig1 = px.histogram(
    filtered_df,
    x="track_popularity",
    nbins=20,
    title="Track Popularity Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

st.subheader("Top Playlist Genres")

top_genres = (
    filtered_df["playlist_genre"]
    .value_counts()
    .head(10)
    .reset_index()
)

top_genres.columns = ["Genre", "Count"]

fig2 = px.bar(
    top_genres,
    x="Genre",
    y="Count",
    title="Top Playlist Genres"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Correlation Heatmap")

features = [
    "track_popularity",
    "danceability",
    "energy",
    "acousticness",
    "tempo",
    "valence",
    "loudness",
    "liveness",
    "speechiness",
    "instrumentalness"
]

corr = filtered_df[features].corr()

fig3 = px.imshow(
    corr,
    text_auto=True,
    title="Feature Correlation Heatmap"
)

st.plotly_chart(fig3, use_container_width=True)

st.subheader("Energy vs Track Popularity")

fig4 = px.scatter(
    filtered_df,
    x="energy",
    y="track_popularity",
    hover_data=["track_name"],
    title="Energy vs Track Popularity"
)

st.plotly_chart(fig4, use_container_width=True)

st.subheader("Top 10 Most Popular Tracks")

top_tracks = filtered_df.sort_values(
    by="track_popularity",
    ascending=False
)[
    ["track_name", "track_artist", "playlist_genre", "track_popularity"]
].head(10)

st.dataframe(top_tracks)