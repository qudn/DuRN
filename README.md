# DuRN: Dual-View Uncertainty-Aware Representation Learning for Tabular Anomaly Detection

Official implementation of the paper **"Dual-View Uncertainty-Aware Representation Learning for Tabular Anomaly Detection"**.

DuRN introduces a dual-view learning framework that synergizes a non-parametric **Topological Safety Net** with a parametric **Deep Semantic Booster** to resolve the stability-plasticity dilemma in tabular anomaly detection.

---

## 🌟 Key Features

* **Manifold Unfolding**: Utilizes Residual Randomized Projection Regressors (ResRPR) to enforce manifold unfolding and prevent model collapse.
* **Uncertainty-Aware Gating**: Implements a **dynamic sample-aware execution**. The model automatically switches between views based on real-time uncertainty (variance) for each specific sample.
* **Zero-tuning Robustness**: Achieves SOTA performance across 25 diverse benchmarks using a **globally unified decision criterion ($\tau=0.5$)** without per-dataset hyperparameter optimization.
* **High Throughput**: Sustains a real-time inference speed of **>3.5M samples/s** on a single NVIDIA RTX 3060 Ti.

---

## 🛠️ Requirements

The project is implemented in Python 3.8+ and PyTorch. Install dependencies via:

```bash
pip install -r requirements.txt
