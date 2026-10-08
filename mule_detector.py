# mule_detector.py - Phase 8: find mule accounts using a graph (NetworkX)
# A mule account = gets money from MANY different people and quickly passes it on.
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import joblib
import mlflow
import matplotlib
matplotlib.use("Agg")          # save pictures to a file, don't open a window

# ---------- settings (our rules) ----------
MIN_SENDERS = 10       # received money from at least this many different people
MIN_PASS_ON = 0.8      # sent out at least 80% of what came in
MAX_DELAY_MIN = 60     # first payment out happened within 60 min of the last payment in


# ---------- 1. build the graph ----------
# node = an account, edge = "sender paid receiver"
def build_graph(df):
    G = nx.DiGraph()
    senders = df["sender_id"].tolist()
    receivers = df["receiver_id"].tolist()
    amounts = df["amount"].tolist()
    times = df["timestamp"].tolist()

    for i in range(len(df)):
        u = senders[i]
        v = receivers[i]
        if G.has_edge(u, v):
            G[u][v]["amount"] += amounts[i]
            G[u][v]["count"] += 1
            G[u][v]["times"].append(times[i])
        else:
            G.add_edge(u, v, amount=amounts[i], count=1, times=[times[i]])
    return G


# ---------- 2. look at every account and flag mules ----------
def find_mules(G):
    mules = {}
    for node in G.nodes:
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))
        if len(in_edges) < MIN_SENDERS or len(out_edges) == 0:
            continue

        total_in = 0
        last_in = None
        for u, v, d in in_edges:
            total_in += d["amount"]
            for t in d["times"]:
                if last_in is None or t > last_in:
                    last_in = t

        total_out = 0
        first_out = None
        for u, v, d in out_edges:
            total_out += d["amount"]
            for t in d["times"]:
                if first_out is None or t < first_out:
                    first_out = t

        pass_on = total_out / total_in
        delay_min = (first_out - last_in).total_seconds() / 60.0

        if pass_on >= MIN_PASS_ON and 0 <= delay_min <= MAX_DELAY_MIN:
            mules[node] = {
                "senders": len(in_edges),
                "total_in": total_in,
                "total_out": total_out,
                "pass_on": pass_on,
                "delay_min": delay_min,
                "cashout_accounts": [v for u, v, d in out_edges],
            }
    return mules


def mule_reasons(info):
    return ["received money from %d different people" % info["senders"],
            "passed on %.0f%% of it" % (100 * info["pass_on"]),
            "first payment out only %.0f minutes after the last payment in" % info["delay_min"]]


# used by the dashboard / rules: is this account a known mule?
def is_mule(account, mules):
    if account in mules:
        return True, mule_reasons(mules[account])
    return False, []


# ---------- 3. draw one mule and its neighbours ----------
def draw_mule(G, mule, info, filename):
    victims = [u for u, v in G.in_edges(mule)]
    victims = victims[:12]                       # keep the picture readable
    cashouts = info["cashout_accounts"]

    # fixed positions: victims on the left, mule in the middle, cash-out on the right
    pos = {}
    for i in range(len(victims)):
        pos[victims[i]] = (0, i)
    pos[mule] = (1.5, (len(victims) - 1) / 2.0)
    for i in range(len(cashouts)):
        pos[cashouts[i]] = (3, i * 2 + (len(victims) - 1) /
                            2.0 - len(cashouts) + 1)

    H = nx.DiGraph()
    for v in victims:
        H.add_edge(v, mule)
    for c in cashouts:
        H.add_edge(mule, c)

    colors = []
    for n in H.nodes:
        if n == mule:
            colors.append("#d62728")      # red = mule
        elif n in cashouts:
            colors.append("#ff7f0e")      # orange = cash-out
        else:
            colors.append("#1f77b4")      # blue = victims
    plt.figure(figsize=(9, 6))
    nx.draw(H, pos, with_labels=True, node_color=colors, node_size=900,
            font_size=7, font_color="white", arrows=True, edge_color="#888888")
    plt.title("Mule account %s: %d people paid it (showing %d), then it forwarded the money"
              % (mule, info["senders"], len(victims)))
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()


# ---------- run: `python mule_detector.py` ----------
if __name__ == "__main__":
    df = pd.read_csv("data/upi_transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    G = build_graph(df)
    print("Graph built: %d accounts (nodes), %d payment links (edges)"
          % (G.number_of_nodes(), G.number_of_edges()))

    # sparse adjacency matrix (PDF topics: Graphs + Sparse Matrices)
    A = nx.to_scipy_sparse_array(G)
    cells = A.shape[0] * A.shape[1]
    print("Adjacency matrix: %s, shape %s, only %.3f%% of cells are filled"
          % (type(A).__name__, A.shape, 100.0 * A.nnz / cells))

    mules = find_mules(G)
    print("\nFlagged mule accounts:", len(mules))

    # ----- how good is it? compare with the real answer in the data -----
    true_mules = set(df[df["fraud_type"] == "mule_inbound"]["receiver_id"])
    flagged = set(mules.keys())
    caught = flagged & true_mules
    precision = len(caught) / len(flagged) if len(flagged) > 0 else 0
    recall = len(caught) / len(true_mules)
    print("Real mule accounts in data:", len(true_mules))
    print("Correctly flagged: %d | wrongly flagged: %d | missed: %d"
          % (len(caught), len(flagged - true_mules), len(true_mules - flagged)))
    print("Precision: %.3f  Recall: %.3f" % (precision, recall))

    # ----- how many mule payments does this catch? -----
    mule_pay = df[df["fraud_type"] == "mule_inbound"]
    hit = 0
    for r in mule_pay["receiver_id"]:
        if r in flagged:
            hit += 1
    print("Mule payments whose receiver we flagged: %d of %d" %
          (hit, len(mule_pay)))

    # ----- show a few mules -----
    print("\nExamples:")
    shown = 0
    for acc in mules:
        info = mules[acc]
        print(" ", acc, "->", ", ".join(mule_reasons(info)))
        print("     forwards money to:", info["cashout_accounts"])
        shown += 1
        if shown == 3:
            break

    # ----- cash-out accounts (where mules send the money) -----
    cashouts = set()
    for acc in mules:
        for c in mules[acc]["cashout_accounts"]:
            cashouts.add(c)
    print("\nCash-out accounts that receive from mules:", sorted(cashouts))

    # ----- save + draw + log -----
    joblib.dump(mules, "models/mule_accounts.pkl")
    print("\nSaved models/mule_accounts.pkl")

    first_mule = list(mules.keys())[0]
    draw_mule(G, first_mule, mules[first_mule], "mule_graph.png")
    print("Saved mule_graph.png")

    mlflow.set_experiment("upi_fraud_detection")
    with mlflow.start_run(run_name="Mule_detection_graph"):
        mlflow.log_param("min_senders", MIN_SENDERS)
        mlflow.log_param("min_pass_on", MIN_PASS_ON)
        mlflow.log_param("max_delay_min", MAX_DELAY_MIN)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accounts_flagged", len(mules))
        mlflow.log_artifact("mule_graph.png")
    print("Logged to MLflow")
