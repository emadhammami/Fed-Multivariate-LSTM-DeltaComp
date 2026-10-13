# Fed-Multivariate-LSTM-DeltaComp

> **Historical exploratory prototype.** The current example script takes the first value of each dataset-level `target` series and creates sliding windows across rows rather than consecutive hourly observations. Its reported MAE values therefore should not be treated as validated hourly electricity-load forecasting results. Proper temporal reconstruction and chronological evaluation are needed before drawing performance conclusions. The code is retained for inspection.


Open-source implementation of **compressed federated learning** for **multivariate** time-series forecasting using an LSTM with DeltaCompressor on the [Electricity Load Diagrams](https://huggingface.co/datasets/tulipa762/electricity_load_diagrams) dataset.

## How this differs from the previous repos

| | [Repo 1: Compressed-Federated-Time-Series-Forecasting](https://github.com/emadhammami/Compressed-Federated-Time-Series-Forecasting) | [Repo 2: Compressed-FL-Univariate-LSTM-Forecasting](https://github.com/emadhammami/Compressed-FL-Univariate-LSTM-Forecasting) | **This repo** |
|---|---|---|---|
| **Input features** | `feat_static_cat`, `hour`, `dayofweek`, `month` (no target history) | Raw load values only (univariate) | `target_scalar` + `hour` + `dayofweek` + `month` (target history + calendar) |
| **Sequence length** | 10 | 24 | 24 |
| **Local epochs** | 2 | 10 | 2 |
| **Model input shape** | `(10, 4)` | `(24, 1)` | `(24, 4)` |
| **Key idea** | Tabular features, short window | Long window, pure signal | Long window, signal + calendar context |

> **Note:** None of these repos use an LLM (Large Language Model). All use an **LSTM** (Long Short-Term Memory), a classical recurrent neural network.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python federated_lstm_multivariate.py
```

The script will:
1. Download the dataset automatically via Hugging Face `datasets`
2. Train the federated LSTM for 100 rounds across 3 clients
3. Print per-round upload size (KB) and validation MAE
4. Save uplink stats to `federated_lstm_time_series_with_features.csv`

## License

MIT
