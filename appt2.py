import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy.stats import ttest_ind, pearsonr, spearmanr, chi2_contingency, f_oneway

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Spotify Final Data Analytics Project",
    layout="wide"
)

st.title("Spotify Music Popularity Analytics")
#st.markdown()


df = pd.read_csv("data/masterdata/master_dataset.csv")

#data wranglinggg

df = df.drop_duplicates()

important_cols = [
    "popularity",
    "track_genre",
    "energy",
    "danceability",
    "valence",
    "loudness"
]

df = df.dropna(subset=important_cols)

if "explicit" in df.columns:
    df["explicit"] = df["explicit"].astype(int)

#sidebar filteringgg

st.sidebar.header("🎯 Filters")

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(df["track_genre"].unique().tolist())
)

popularity_range = st.sidebar.slider(
    "Popularity Range",
    int(df["popularity"].min()),
    int(df["popularity"].max()),
    (
        int(df["popularity"].min()),
        int(df["popularity"].max())
    )
)

filtered_df = df.copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["track_genre"] == selected_genre
    ]

filtered_df = filtered_df[
    (filtered_df["popularity"] >= popularity_range[0]) &
    (filtered_df["popularity"] <= popularity_range[1])
]

#dataset preview

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

#summary statsss

st.subheader("Summary Statistics")

mean_pop = round(filtered_df["popularity"].mean(), 2)
median_pop = round(filtered_df["popularity"].median(), 2)
mode_genre = filtered_df["track_genre"].mode()[0]
std_pop = round(filtered_df["popularity"].std(), 2)
variance_pop = round(filtered_df["popularity"].var(), 2)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Mean Popularity", mean_pop)
c2.metric("Median Popularity", median_pop)
c3.metric("Mode Genre of tracks", mode_genre)
c4.metric("Standard Deviation", std_pop)
c5.metric("Variance", variance_pop)

#historgram

st.subheader("Popularity Distribution")

fig1 = px.histogram(
    filtered_df,
    x="popularity",
    nbins=25,
    title="Popularity Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

st.text("we use binning here to divide data into 25 bins and show how many songs lie under which popularity")
st.subheader("Insights from Popularity Distribution")

st.text("1. Most tracks have very low popularity (0–5), showing many songs are less discovered.")
st.text("2. The majority of songs fall in the moderate popularity range (20–60).")
st.text("3. Popularity peaks around 40–45, indicating many songs perform at an average level.")
st.text("4. Very few tracks have high popularity above 80, showing hit songs are rare.")
st.text("5. The distribution is right-skewed, meaning only a small number of songs become highly popular.")

#boxplot

st.subheader("📦 Popularity Boxplot")

fig2 = px.box(
    filtered_df,
    y="popularity",
    title="Outlier Detection"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Insights from Popularity Boxplot")

st.text("1. The median popularity is around 35, showing that most songs have average popularity levels.")
st.text("2. The middle 50% of tracks lie approximately between 17 and 50, indicating moderate variation in popularity.")
st.text("3. The lower whisker reaches close to 0, which means many songs have very low popularity.")
st.text("4. The upper whisker extends near 100, showing that some tracks achieve extremely high popularity.")
st.text("5. The wide spread of the boxplot suggests significant variation in song popularity across the dataset.")

#scatter plottt

st.subheader("Energy vs Popularity")

fig3 = px.scatter(
    filtered_df,
    x="energy",
    y="popularity",
    hover_data=["track_name"],
    title="Energy vs Popularity"
)

st.plotly_chart(fig3, use_container_width=True)

st.subheader("Insights from Energy vs Popularity")

st.text("1. The points are widely scattered, showing there is no strong direct relationship between energy and popularity.")
st.text("2. Most tracks are concentrated in the mid-to-high energy range (around 0.4 to 0.9).")
st.text("3. This suggests that energy alone is not a strong factor in determining a song’s popularity.")

#top genre

st.subheader("Top Genres by Average Popularity")

top_genres = (
    filtered_df.groupby("track_genre", as_index=False)["popularity"]
    .mean()
    .sort_values(by="popularity", ascending=False)
    .head(10)
)

fig4 = px.bar(
    top_genres,
    x="track_genre",
    y="popularity",
    title="Top Genres by Average Popularity"
)

st.plotly_chart(fig4, use_container_width=True)

#pie chart

st.subheader("🥧 Genre Popularity Distribution")

fig5 = px.pie(
    top_genres,
    names="track_genre",
    values="popularity",
    title="Genre Distribution"
)

st.plotly_chart(fig5, use_container_width=True)

st.subheader("Insights from Top Genres by Average Popularity")

st.text("1. Pop-film has the highest average popularity among all genres, followed closely by K-pop.")
st.text("2. Genres like chill and sad also perform well, showing strong listener engagement.")
st.text("3. Popular genres such as pop and emo have slightly lower average popularity compared to the top-ranked genres.")

#correlation map 

st.subheader("Correlation Heatmap")

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

corr = filtered_df[features].corr()

fig6 = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap"
)

st.plotly_chart(fig6, use_container_width=True)

st.subheader("Insights from Correlation Heatmap")

st.text("1. Popularity shows only weak correlations with most audio features, suggesting no single feature strongly determines popularity.")
st.text("2. Energy and loudness have a positive correlation, meaning energetic songs are often louder.")
st.text("3. Acousticness has a negative correlation with energy and loudness, indicating acoustic songs are usually less energetic and softer.")

#correlation test 

st.subheader("Correlation Analysis")
st.markdown("Pearson Analysis to check for linear relation b/w energy and popularity.")
st.markdown("Spearsman Analysis to check rank based relation b/w danceability and popularity.")

pearson_corr, _ = pearsonr(
    filtered_df["energy"],
    filtered_df["popularity"]
)

spearman_corr, _ = spearmanr(
    filtered_df["danceability"],
    filtered_df["popularity"]
)

st.write(f"Pearson Correlation: {round(pearson_corr, 3)}")
st.write(f"Spearman Correlation: {round(spearman_corr, 3)}")

#t test
st.subheader("T-Test")
st.markdown("T-Test compares the average popularity of explicit vs non-explicit songs.")

explicit_yes = filtered_df[
    filtered_df["explicit"] == 1
]["popularity"]

explicit_no = filtered_df[
    filtered_df["explicit"] == 0
]["popularity"]

t_stat, p_value = ttest_ind(
    explicit_yes,
    explicit_no,
    nan_policy="omit"
)

st.write(f"T-Statistic: {round(t_stat, 3)}")
st.write(f"P-Value: {round(p_value, 5)}")

#chi square test

st.subheader("Chi-Square Test")
st.markdown("Chi-Square Test checks whether track_genre and explicit are related.")

contingency_table = pd.crosstab(
    filtered_df["track_genre"],
    filtered_df["explicit"]
)

chi2, p, dof, expected = chi2_contingency(contingency_table)

st.write(f"Chi-Square Statistic: {round(chi2, 3)}")
st.write(f"P-Value: {round(p, 5)}")

#anova test

st.subheader("ANOVA Test")
st.markdown("ANOVA Test compares popularity across the top 3 genres.")

top3 = filtered_df["track_genre"].value_counts().head(3).index.tolist()

groups = [
    filtered_df[
        filtered_df["track_genre"] == genre
    ]["popularity"]
    for genre in top3
]

anova_stat, anova_p = f_oneway(*groups)

st.write(f"ANOVA Statistic: {round(anova_stat, 3)}")
st.write(f"P-Value: {round(anova_p, 5)}")


st.subheader("Insights from Statistical Tests")

st.text("1. Pearson Correlation (-0.002) shows almost no linear relationship between energy and popularity, meaning energy does not strongly affect popularity.")
st.text("2. Spearman Correlation (0.026) also indicates a very weak relationship between danceability and popularity, showing danceability alone is not a major factor.")
st.text("3. The T-Test p-value (0.0) shows a significant difference in popularity between explicit and non-explicit songs, meaning explicit content may influence popularity.")
st.text("4. The Chi-Square Test p-value (0.0) suggests a strong association between track genre and explicit content, meaning some genres are more likely to contain explicit songs.")
st.text("5. The ANOVA Test p-value (0.0) confirms that popularity differs significantly across genres, showing genre plays an important role in song popularity.")

#pca dimension reduction

st.subheader("🧠 PCA (Dimensionality Reduction)")

X_pca = filtered_df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    principal_components,
    columns=["PC1", "PC2"]
)

fig7 = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    title="PCA Visualization"
)

st.plotly_chart(fig7, use_container_width=True)

st.subheader("PCA Insights")

st.write("""
1. PC1 captures the main musical variation (likely energy, loudness, or tempo).
2. PC2 reflects secondary traits like mood or acoustic vs electronic feel.
3. Most songs vary along one dominant vibe rather than multiple independent factors.
4. Smooth spread indicates gradual transitions between song styles, not strict genres.
5. Extreme points may represent very unique tracks (e.g., highly energetic or very calm).
""")

#k means clusttering

st.subheader("🎯 K-Means Clustering")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca_df["Cluster"] = clusters.astype(str)

fig8 = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="K-Means Clustering"
)

st.plotly_chart(fig8, use_container_width=True)

st.subheader("K-Means Clustering Insights")

st.write("""
1. Songs are grouped into 3 clusters based on audio features (like energy, danceability, tempo, etc.).
2. Separation along PC1 suggests one dominant musical dimension (likely energy/tempo or loudness).
3. One cluster (lower PC2) may represent calmer/softer tracks, while others are more energetic.
4. Overlap between clusters shows some songs share mixed characteristics (genre blending).
5. Clusters can be used to identify song moods or create playlists with similar vibes.
""")

#regression model

st.subheader("Supervised Learning (Regression)")

ml_features = [
    "danceability",
    "energy",
    "acousticness",
    "tempo",
    "valence",
    "loudness"
]

X = filtered_df[ml_features]
y = filtered_df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

st.write(f"MAE: {round(mae, 2)}")
st.write(f"RMSE: {round(rmse, 2)}")
st.write(f"R² Score: {round(r2, 3)}")

st.subheader("Regression Insights (Spotify Popularity)")

st.write("""
1. Model explains ~50% variance → moderate prediction power, other factors also matter.
2. Acousticness, tempo, and loudness are the strongest drivers of popularity.
3. Energy and valence have comparatively lower influence on popularity.
4. Errors (MAE ~11) show predictions are reasonably close but not highly precise.
5. Popularity depends on more than audio features (e.g., artist, trends, marketing).
""")

#feature importance

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": ml_features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

fig9 = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Feature Importance"
)

st.plotly_chart(fig9, use_container_width=True)

st.subheader("Feature Importance Insights")

st.write("""
1. Acousticness, tempo, and loudness are the most influential features.
2. Danceability has moderate impact on predicting popularity.
3. Valence contributes but is not a major driver.
4. Energy has the least importance among selected features.
5. Overall, multiple features contribute fairly evenly—no single dominant factor.
""")
