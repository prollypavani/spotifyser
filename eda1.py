import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("master_dataset.csv")

plt.figure(figsize=(8,5))
plt.hist(df["popularity"], bins=20)

plt.xlabel("Popularity Score")
plt.ylabel("Number of Songs")
plt.title("Distribution of Song Popularity")

plt.savefig("popularity_histogram.png")

print("Graph saved successfully as popularity_histogram.png")