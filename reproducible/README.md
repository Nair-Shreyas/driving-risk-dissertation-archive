# Reproduce the pipeline

Two ways to run this work, from a 5-minute demo to the full study.

| | What you need | Time |
|---|---|---|
| **Option A — Demo** | Nothing — a browser and a Google account | ~5 minutes |
| **Option B — Full study** | Hugging Face account with NVIDIA dataset access, Google Drive, Colab GPU | Several hours |

The code is version 2 of the pipeline: optimised, more robust to dataset updates and cleaner than the
archived notebooks, with a few bug fixes. See [`CHANGES.md`](CHANGES.md).

---

## Option A — Demo (no NVIDIA data needed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nair-Shreyas/driving-risk-dissertation-archive/blob/main/reproducible/demo/demo_pipeline.ipynb)

1. Click **Open in Colab**.
2. **Runtime → Run all.**

The demo generates synthetic clips with the same columns the real pipeline produces, then runs
Notebook 4 (dataset assembly) and Notebook 5 (proxy labels, baselines, tuned XGBoost, cross-validation,
SHAP, ablation) unchanged. You get every table and chart of the method in a few minutes.
The data is invented, so the numbers are **not** the study's results.

Run it locally instead:

```bash
pip install -r requirements.txt jupyter
jupyter notebook reproducible/demo/demo_pipeline.ipynb
```

`demo/make_demo_data.py` is the same synthetic-data generator as a standalone script.

---

## Option B — Full study on the NVIDIA dataset

| # | Notebook | Open |
|---|---|---|
| 1 | Data preparation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nair-Shreyas/driving-risk-dissertation-archive/blob/main/reproducible/notebooks/1_data_preparation.ipynb) |
| 2 | Ego-motion features | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nair-Shreyas/driving-risk-dissertation-archive/blob/main/reproducible/notebooks/2_ego_motion_features.ipynb) |
| 3 | Visual features (GPU) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nair-Shreyas/driving-risk-dissertation-archive/blob/main/reproducible/notebooks/3_visual_features.ipynb) |
| 4 | Dataset assembly | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nair-Shreyas/driving-risk-dissertation-archive/blob/main/reproducible/notebooks/4_dataset_assembly.ipynb) |
| 5 | Modelling and evaluation (GPU) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nair-Shreyas/driving-risk-dissertation-archive/blob/main/reproducible/notebooks/5_modelling_and_evaluation.ipynb) |

1. **Get data access.** Sign in to [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
   accept NVIDIA's licence on the dataset page, and create a read token. On Colab, add it as a secret named
   `HF_TOKEN` (key icon in the left sidebar) or paste it when the notebook asks.
2. **Open Notebook 1** and check the **SETTINGS** cell: `DATA_FOLDER` is where all outputs go
   (default: `MyDrive/driving-risk/pipeline_data` on Colab, `./pipeline_data` locally).
3. **Runtime → Run all**, then do the same for Notebooks 2 → 5 in order, using the same `DATA_FOLDER`.
   Choose a GPU runtime for Notebooks 3 and 5.
4. Results, tables and figures are saved to `DATA_FOLDER/05_modelling_outputs/`.

For a quick end-to-end check before the full run, set `FAST_MODE = True` in Notebook 5's SETTINGS cell.

> **Data licence.** The NVIDIA dataset is licensed by NVIDIA and must not be redistributed. Keep the
> files the notebooks create in your own storage; do not commit them to a public repository.

> **Colab buttons and private repositories.** While this repository is private, Colab asks you to sign in
> to GitHub and tick *Include private repos* before it can open the notebooks.
