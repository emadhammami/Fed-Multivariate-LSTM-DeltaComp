# ───────── Install if needed ───────── #
# !pip install tensorflow==2.18.0 numpy==1.26.4 pandas==2.2.2 datasets==2.18.0 pyarrow==14.0.2 --force-reinstall
# import os; os.kill(os.getpid(), 9)

import numpy as np
import tensorflow as tf
import gzip, io, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from datasets import load_dataset

# ───────── Load dataset ───────── #

dataset = load_dataset("tulipa762/electricity_load_diagrams")
df = dataset["train"].to_pandas()

# Extract target and calendar features
df["target_scalar"] = df["target"].apply(lambda x: x[0])
df["datetime"] = pd.to_datetime(df["start"])
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

# Full feature matrix: (target_scalar, hour, dayofweek, month)
features = df[["target_scalar", "hour", "dayofweek", "month"]].to_numpy().astype("float32")

# ───────── Build sequences ───────── #

SEQUENCE_LENGTH = 24  # last 24 hours

X_seq, y_seq = [], []
for i in range(len(features) - SEQUENCE_LENGTH):
    X_seq.append(features[i:i+SEQUENCE_LENGTH])
    y_seq.append(features[i+SEQUENCE_LENGTH][0])  # target_scalar is first column
X_seq = np.array(X_seq)  # shape: (samples, 24, 4)
y_seq = np.array(y_seq).reshape(-1, 1)

# ───────── Normalize ───────── #

scaler_x = StandardScaler().fit(X_seq.reshape(-1, X_seq.shape[-1]))
X_seq_scaled = scaler_x.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

scaler_y = StandardScaler().fit(y_seq)
y_seq_scaled = scaler_y.transform(y_seq)

# ───────── Train/Val/Test split ───────── #

x_train, x_tmp, y_train, y_tmp = train_test_split(X_seq_scaled, y_seq_scaled, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.50, random_state=42)

NUM_CLIENTS = 3
chunk = len(x_train) // NUM_CLIENTS
clients = [(x_train[i*chunk:(i+1)*chunk], y_train[i*chunk:(i+1)*chunk]) for i in range(NUM_CLIENTS)]

# ───────── DeltaCompressor ───────── #

class DeltaCompressor:
    def __init__(self, kf=1e-3):
        self.kf = kf
        self.res = None

    @staticmethod
    def _flat(ws):
        flat, shp = [], []
        for w in ws:
            shp.append(w.shape)
            flat.append(w.ravel())
        return np.concatenate(flat), shp

    @staticmethod
    def _unflat(vec, shp):
        out, i = [], 0
        for s in shp:
            n = np.prod(s)
            out.append(vec[i:i+n].reshape(s))
            i += n
        return out

    def compress(self, delta):
        if self.res is not None:
            delta = [d + r for d, r in zip(delta, self.res)]
        v, shp = self._flat(delta)
        mad = np.mean(np.abs(v))
        mask = np.abs(v) >= mad
        k = max(1, int(self.kf * v.size))
        if k < v.size:
            thr = np.partition(np.abs(v), -k)[-k]
            mask &= np.abs(v) >= thr
        idx = np.where(mask)[0].astype(np.int32)
        sign = np.sign(v[idx]).astype(np.int8)
        scale = np.float32(np.abs(v[idx]).mean()) if idx.size else np.float32(0.0)
        recon = np.zeros_like(v)
        recon[idx] = sign.astype(np.float32) * scale
        self.res = self._unflat(v - recon, shp)
        payload = {"i": idx, "s": sign, "k": scale, "sh": shp}
        b = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
            f.write(b)
        return buf.getvalue()

    def decompress(self, blob):
        p = pickle.loads(gzip.GzipFile(fileobj=io.BytesIO(blob)).read())
        vec = np.zeros(sum(np.prod(s) for s in p["sh"]), np.float32)
        vec[p["i"]] = p["s"].astype(np.float32) * p["k"]
        return self._unflat(vec, p["sh"])

# ───────── Build LSTM model ───────── #

def build_lstm(lr=1e-2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 4)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9), loss="mae")
    return model

# ───────── Federated Learning Loop ───────── #

ROUNDS, LOCAL_EPOCHS, BATCH = 100, 2, 64
comp = DeltaCompressor(1e-3)
global_model = build_lstm()
global_w = global_model.get_weights()

results = []

for r in range(1, ROUNDS + 1):
    uplink_bytes, deltas = [], []
    for cx, cy in clients:
        local = build_lstm()
        local.set_weights(global_w)
        local.fit(cx, cy, LOCAL_EPOCHS, BATCH,
                  validation_data=(x_val, y_val),
                  callbacks=[EarlyStopping("val_loss", patience=2, restore_best_weights=True, verbose=0)],
                  verbose=0)
        delta = [p - g for p, g in zip(local.get_weights(), global_w)]
        blob = comp.compress(delta)
        uplink_bytes.append(len(blob))
        deltas.append(comp.decompress(blob))

    agg = [np.mean([d[i] for d in deltas], axis=0) for i in range(len(global_w))]
    global_w = [g + a for g, a in zip(global_w, agg)]
    global_model.set_weights(global_w)

    if r == 1 or r % 10 == 0:
        val_preds = global_model.predict(x_val, verbose=0)
        val_preds_rescaled = scaler_y.inverse_transform(val_preds)
        val_y_rescaled = scaler_y.inverse_transform(y_val)
        val_mae = np.mean(np.abs(val_preds_rescaled - val_y_rescaled))
        print(f"Round {r:3d} | upload {np.mean(uplink_bytes)/1024:.2f} KB | val MAE {val_mae:.2f}")

    results.append({"round": r, "uplink_kb": np.mean(uplink_bytes)/1024})

# ───────── Final Evaluation ───────── #

test_preds = global_model.predict(x_test, verbose=0)
test_preds_rescaled = scaler_y.inverse_transform(test_preds)
y_test_rescaled = scaler_y.inverse_transform(y_test)
test_mae = np.mean(np.abs(test_preds_rescaled - y_test_rescaled))
print("\nFinal test MAE:", test_mae)

pd.DataFrame(results).to_csv("federated_lstm_time_series_with_features.csv", index=False)
print("Saved uplink stats.")

# ───────── Reproduced Results ───────── #
# Dataset : tulipa762/electricity_load_diagrams
# Clients : 3  |  Rounds : 100  |  Local epochs : 2  |  Batch : 64
# Sequence length : 24  |  Input : multivariate (target + hour + dayofweek + month)
# Compressor : DeltaCompressor(kf=1e-3)
#
# Round   1 | upload 0.29 KB | val MAE  826.43
# Round  10 | upload 0.30 KB | val MAE  622.01
# Round  20 | upload 0.30 KB | val MAE  655.33
# Round  30 | upload 0.30 KB | val MAE  621.19
# Round  40 | upload 0.31 KB | val MAE  637.62
# Round  50 | upload 0.31 KB | val MAE  623.13
# Round  60 | upload 0.30 KB | val MAE  626.01
# Round  70 | upload 0.31 KB | val MAE  645.06
# Round  80 | upload 0.31 KB | val MAE  645.78
# Round  90 | upload 0.31 KB | val MAE  644.23
# Round 100 | upload 0.31 KB | val MAE  634.17
#
# Final test MAE : 1150.56
# ─────────────────────────────────────── #
