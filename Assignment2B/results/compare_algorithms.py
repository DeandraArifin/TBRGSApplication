import pandas as pd
import matplotlib.pyplot as plt

# file paths of results
files = {
    "RNN": "rnn_metrics.csv",
    "LSTM": "lstm_metrics.csv",
    "GRU": "gru_metrics.csv"
}

# will be graphing averages
averages = {}

for name, file in files.items():
    df = pd.read_csv(file)
    averages[name] = {
        "MAE": df["mae"].mean(),
        "MSE": df["mse"].mean(),
        "R2": df["r2"].mean()
    }

# convert to DataFrame for plotting
avg_df = pd.DataFrame(averages).T

# plotting logic
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["MAE", "MSE", "R2"]
colors = ["skyblue", "lightgreen", "salmon"]

for i, metric in enumerate(metrics):
    avg_df[metric].plot(kind="bar", ax=axes[i], title=metric, legend=False, color=colors[i])
    axes[i].set_ylabel(metric)
    axes[i].set_xlabel("Algorithm")

plt.tight_layout()
plt.suptitle("Average Metrics Comparison by Algorithm", y=1.05)
plt.savefig("Average Metric Comparison.png")
plt.show()