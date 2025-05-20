import pandas as pd
import os

def load_interval_data(path=None):
    if path is None:
        base_directory = os.path.dirname(__file__)
        path = os.path.join(base_directory, "Resources", "interval_data.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df["timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    return df[["SCATS Number","timestamp","Flow"]]


def aggregate_hourly(flows15: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        flows15
          .groupby(["SCATS Number","timestamp"], as_index=False)["Flow"]
          .sum()
          .rename(columns={"Flow":"flow"})
    )
    return hourly


def load_site_coords(path=None):
    if path is None:
        base_directory = os.path.dirname(__file__)
        path = os.path.join(base_directory, "Resources", "merged_data.csv")
    meta = pd.read_csv(path, usecols=["SCATS Number","NB_LATITUDE","NB_LONGITUDE"])
    # ensure consistent types
    meta["SCATS Number"] = meta["SCATS Number"].astype(str)
    meta = meta.drop_duplicates("SCATS Number")
    coords = {
        row["SCATS Number"]: (row["NB_LATITUDE"], row["NB_LONGITUDE"])
        for _, row in meta.iterrows()
    }
    return meta, coords


if __name__ == "__main__":
    flows15 = load_interval_data()
    print("15-min data head:")
    print(flows15.head(), "\n")

    hourly = aggregate_hourly(flows15)
    print("Hourly aggregates head:")
    print(hourly.head(), "\n")

    meta, coords = load_site_coords()
    print(f"Loaded {len(coords)} sites. Example coord:", next(iter(coords.items())))
