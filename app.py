import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Spotify Music Popularity Dashboard",
    layout="wide"
)

st.title("🎵 Spotify Music Popularity Analysis Dashboard")
st.markdown("Analyze how audio features influence song popularity on Spotify")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/masterdata/master_dataset.csv")

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🎯 Filters")

# Genre Filter
selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(df["track_genre"].dropna().unique().tolist())
)

# Popularity Range Filter
popularity_range = st.sidebar.slider(
    "Select Popularity Range",
    int(df["popularity"].min()),
    int(df["popularity"].max()),
    (
        int(df["popularity"].min()),
        int(df["popularity"].max())
    )
)

# Apply Filters
filtered_df = df.copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["track_genre"] == selected_genre
    ]

filtered_df = filtered_df[
    (filtered_df["popularity"] >= popularity_range[0]) &
    (filtered_df["popularity"] <= popularity_range[1])
]

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("📂 Dataset Preview")
st.dataframe(filtered_df.head())

# --------------------------------------------------
# QUICK STATISTICS
# --------------------------------------------------

st.subheader("📊 Quick Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Songs", len(filtered_df))
col2.metric(
    "Average Popularity",
    round(filtered_df["popularity"].mean(), 2)
)
col3.metric(
    "Total Genres",
    filtered_df["track_genre"].nunique()
)

# --------------------------------------------------
# STATISTICAL SUMMARY
# --------------------------------------------------

st.subheader("📈 Statistical Summary")

mean_popularity = round(filtered_df["popularity"].mean(), 2)
median_popularity = round(filtered_df["popularity"].median(), 2)
std_popularity = round(filtered_df["popularity"].std(), 2)
most_common_genre = filtered_df["track_genre"].mode()[0]

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Mean Popularity", mean_popularity)
col2.metric("Median Popularity", median_popularity)
col3.metric("Std Deviation", std_popularity)
col4.metric("Most Common Genre", most_common_genre)

# --------------------------------------------------
# CORRELATION VALUES
# --------------------------------------------------

st.subheader("🔗 Correlation with Popularity")

corr_energy = round(
    filtered_df["popularity"].corr(filtered_df["energy"]), 3
)

corr_dance = round(
    filtered_df["popularity"].corr(filtered_df["danceability"]), 3
)

corr_valence = round(
    filtered_df["popularity"].corr(filtered_df["valence"]), 3
)

st.write(f"Popularity vs Energy: {corr_energy}")
st.write(f"Popularity vs Danceability: {corr_dance}")
st.write(f"Popularity vs Valence: {corr_valence}")

# --------------------------------------------------
# POPULARITY DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Popularity Distribution")

fig1 = px.histogram(
    filtered_df,
    x="popularity",
    nbins=20,
    title="Distribution of Song Popularity"
)

st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------
# TOP 10 GENRES
# --------------------------------------------------

st.subheader("🎶 Top Genres by Average Popularity")

top_genres = (
    filtered_df
    .groupby("track_genre")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_genres.columns = ["Genre", "Average Popularity"]

fig2 = px.bar(
    top_genres,
    x="Genre",
    y="Average Popularity",
    title="Top 10 Genres by Average Popularity"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

st.subheader("🥧 Genre Popularity Distribution")

genre_popularity = (
    filtered_df
    .groupby("track_genre")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

genre_popularity.columns = ["Genre", "Average Popularity"]

fig3 = px.pie(
    genre_popularity,
    names="Genre",
    values="Average Popularity",
    title="Genre Distribution by Average Popularity"
)

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------------

st.subheader("🔥 Correlation Heatmap")

features = [
    "popularity",
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

corr_matrix = filtered_df[features].corr()

fig4 = px.imshow(
    corr_matrix,
    text_auto=True,
    title="Feature Correlation Heatmap"
)

st.plotly_chart(fig4, use_container_width=True)

# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

st.subheader("⚡ Energy vs Popularity")

fig5 = px.scatter(
    filtered_df,
    x="energy",
    y="popularity",
    hover_data=["track_name"],
    title="Energy vs Popularity"
)

st.plotly_chart(fig5, use_container_width=True)

# --------------------------------------------------
# TOP SONGS TABLE
# --------------------------------------------------

st.subheader("🏆 Top 10 Most Popular Songs")

top_songs = filtered_df.sort_values(
    by="popularity",
    ascending=False
)[
    ["track_name", "artists", "track_genre", "popularity"]
].head(10)

st.dataframe(top_songs)