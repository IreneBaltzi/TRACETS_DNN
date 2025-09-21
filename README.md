## **TRACETS**: Autoregressive Transformer for Sepsis Detection in Clinical Time Series
Deep Neural Networks course project. We adapt TRACE-style embeddings to clinical time series and introduce TRACETS, a decoder-only transformer with causal masking and previous-label conditioning for hour-by-hour sepsis risk estimation.


`Note` This codebase is adapted from TRACE (original repo and paper cited below). The project reorganizes TRACE components for time-series modeling and adds **TRACETS**. Some upstream files may remain for compatibility. The sections below point only to the parts you need to run this project.

## 🚀 How to Run

All experiments are launched from the same entry point: 

```
python main_trace.py
```

- Modify hyperparameters directly in ```configs/tracets_sepsis.yaml``` (e.g., learning rate, optimizer, loss, model size, baselines run).
- No need to edit code for hyperparameters, **just update the YAML.**

## 🐳 Running with Docker

This project is packaged with a `Dockerfile` so you can run experiments in an isolated environment without installing Python packages manually.

### 1️⃣ Build the Docker image
From the root of the repository:
```bash
docker build -t tracets-sepsis:latest .
```

### 2️⃣ Run the container (CPU)
Mount the project, data, and outputs so everything persists outside the container:
```bash
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/Dataset":/workspace/Dataset \
  -v "$(pwd)/outputs":/workspace/outputs \
  -w /workspace \
  tracets-sepsis:latest bash
```

### 3️⃣ Run the container (GPU)
If you have a CUDA-capable GPU and the NVIDIA Container Toolkit installed:
```bash
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/Dataset":/workspace/Dataset \
  -v "$(pwd)/outputs":/workspace/outputs \
  -w /workspace \
  tracets-sepsis:latest bash
```


## 📁 Repo Structure
```
.
├─ .devcontainer/            # VS Code Devcontainer setup (optional)
│   └─ devcontainer.json
├─ .vscode/                  # VS Code configs
│   └─ launch.json
├─ config/                   # YAML configs for experiments
│   ├─ nnmlp_cdc22.yaml      # legacy TRACE configs (not used)
│   ├─ trace_cdc14.yaml      # legacy TRACE configs (not used)
│   ├─ trace_cdc22.yaml      # legacy TRACE configs (not used)
│   └─ tracets_sepsis.yaml   # config used for this project
├─ utils/                    
│   ├─ baselines.py          # LR, XGBoost, LightGBM
│   ├─ config.py             # config parser
│   ├─ datasets.py           # dataset + collate + masking
│   ├─ engine.py             # training / eval loop
│   ├─ losses.py             # legacy TRACE
│   ├─ metrics_sepsis.py     # AUROC, AUPRC, F1, Utility
│   ├─ misc.py               # helper functions
│   ├─ models.py             # TRACETS + TRACE-style embeddings
│   └─ sampler.py            # ThreeToOneEpochSampler (ablation)
├─ data_preprocessing.py     # preprocessing (optional, not needed)
├─ main_trace.py             # entry point (train/eval/baselines)
├─ main_nnmlp.py             # legacy (not used in this project)
├─ sepsis_metadata.json      # feature schema (vitals, labs, demographics)
├─ requirements.txt          # Python dependencies
├─ Dockerfile                # Docker image (PyTorch + deps)
└─ README.md                 # project documentation
```

## 📊 Dataset 
- Source: PhysioNet/CinC 2019 Sepsis Challenge (hourly ICU time series).
- Split: 90/10 patient-level stratified split. The 10% “validation” is used as a held-out test set (the official challenge test set is hidden).
- Features: defined in ```sepsis_metadata.json```.
- Dataset can be found in Google Drive: https://drive.google.com/drive/folders/1bHoMRMRnNUpcA8OJeM_HUpxpPJ8Y7liV?usp=sharing



## 🏆 Results (10% held-out validation)

| Model               | AUROC | AUPRC | Acc  | F1    | Utility |
|---------------------|:-----:|:-----:|:----:|:-----:|:-------:|
| Logistic Regression | 0.760 | 0.053 | 0.81 | 0.090 | 0.237   |
| XGBoost             | 0.790 | 0.074 | 0.86 | 0.110 | 0.262   |
| LightGBM            | 0.750 | 0.052 | 0.86 | 0.090 | 0.199   |
| **TRACETS**  | **0.794** | 0.070 | 0.85 | **0.149** | **0.267** |

_Ablations varying hidden size, heads, normalization, pos-weight, and learning rate are included in the report.  
The best config used `hidden=64`, `heads=2`, `mlp_ratio=2`, `pos_weight=15`, and `lr=1e-5`._

## Reference

This project is adapted from [TRACE: Transformer-Based Risk Assessment for Clinical Evaluation](https://doi.org/10.1109/ACCESS.2025.3577973),  
D. Christopoulos, S. Spanos, V. Ntouskos and K. Karantzalos, *IEEE Access*, 2025.
