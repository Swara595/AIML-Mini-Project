# features.py - Phase 3: build features from the raw transactions
import pandas as pd

df = pd.read_csv("data/upi_transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 1. hour of the day
df["hour"] = df["timestamp"].dt.hour

# 2. amount compared to the sender's normal (average) amount
avg_amount = df.groupby("sender_id")["amount"].mean()
amount_vs_avg = []
for i in range(len(df)):
    sender = df.loc[i, "sender_id"]
    amount_vs_avg.append(df.loc[i, "amount"] / avg_amount[sender])
df["amount_vs_avg"] = amount_vs_avg

# 3. speed check: how many payments did this sender make in the last 2 minutes?
recent_times = {}   # sender -> list of their recent payment times
speed = []
for i in range(len(df)):
    sender = df.loc[i, "sender_id"]
    now = df.loc[i, "timestamp"]
    if sender not in recent_times:
        recent_times[sender] = []
    still_recent = []
    for t in recent_times[sender]:
        if (now - t).total_seconds() <= 120:
            still_recent.append(t)
    speed.append(len(still_recent))
    still_recent.append(now)
    recent_times[sender] = still_recent
df["txn_last_2min"] = speed

# 4. does the message contain a link?
bad_words = ["http", "bit.ly", "tinyurl",
             "cutt.ly", ".xyz", ".top", ".info", ".com"]
has_url = []
for i in range(len(df)):
    msg = str(df.loc[i, "message"])
    found = 0
    for word in bad_words:
        if word in msg:
            found = 1
    has_url.append(found)
df["has_url"] = has_url

# 5. turn transaction type (text) into numbers
df["type_code"] = df["txn_type"].astype("category").cat.codes

# save
df.to_csv("data/features.csv", index=False)

print("Saved data/features.csv")
print("Shape:", df.shape)
print(df[["amount", "hour", "amount_vs_avg", "txn_last_2min",
      "has_url", "type_code", "is_fraud"]].head(10))
print("\nAverage of each feature (normal vs fraud):")
print(df.groupby("is_fraud")[
      ["amount_vs_avg", "txn_last_2min", "has_url", "is_new_payee", "is_new_device"]].mean())
