import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset1_cleaned.csv")

plt.hist(df['popularity'], bins=20)
plt.xlabel("Popularity Score")
plt.ylabel("Number of Songs")
plt.title("Distribution of Song Popularity")

plt.show()
