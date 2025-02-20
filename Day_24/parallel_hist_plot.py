import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
file_path = "histogram.csv"
df = pd.read_csv(file_path)

# Convert intervals to readable labels
df["Label"] = df["Start_Char"] + "-" + df["End_Char"]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(df["Label"], df["Occurrences"], color="skyblue", edgecolor="black")

# Formatting
plt.xlabel("Character Range")
plt.ylabel("Occurrences")
plt.title("Character Frequency Histogram (Grouped by 4)")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot as an image
plt.savefig("./images/histogram_plot.png")

# Show plot
plt.show()
