<p align="center">
  <img src="docs/images/project_overview.png" alt="Multimodal Driving Risk Prediction" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Manuscript_in_preparation-c9440c?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Runs_on-Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Data-NVIDIA_PhysicalAI_AV-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
</p>

# Multimodal Driving Risk Prediction Using Machine Learning on Autonomous Vehicle Data

This repository contains the code, result tables and figures, dissertation and manuscript drafts for
a research project on **predicting driving risk from autonomous vehicle recordings**. It combines how
the vehicle moved (ego-motion), when it was driving (time-of-day context), and what the front camera
saw (detected objects and deep visual embeddings) into a single machine-learning pipeline.

The work was carried out as an MSc Business Analytics dissertation at Dublin Business School (2026) and is
being developed into a journal manuscript.

---

## Contents

- [Research question](#research-question)
- [The data](#the-data)
- [Approach](#approach)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Running the notebooks](#running-the-notebooks)
- [Results](#results)
- [Dissertation and manuscript](#dissertation-and-manuscript)
- [Authors](#authors)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Research question

Real driving datasets rarely come with crash or near-miss labels, which makes supervised risk prediction
difficult. This project asks whether a **multimodal framework** — combining vehicle telemetry, temporal
context and visual scene information — can learn a useful notion of driving risk from **proxy labels**
derived from the data itself, and **which kinds of information** drive those predictions.

## The data

<p align="center">
  <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles">
    <img src="docs/images/data_card.png" alt="NVIDIA PhysicalAI Autonomous Vehicles dataset: 306,152 clips, 1,700 hours, 25 countries" width="100%">
  </a>
</p>

<p align="center">
  <a href="https://cdn-uploads.huggingface.co/production/uploads/667f467e563b0640e37fca79/t_yqQTwuTRFiiK8--mzm3.gif">
    <img src="docs/images/footage_button.png" alt="View sample footage on Hugging Face" width="380">
  </a>
</p>

This project uses the [NVIDIA PhysicalAI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset. Access is gated: sign in to Hugging Face, accept NVIDIA's licence on the dataset page, then run the notebooks to download and process the clips yourself. The [NVIDIA Autonomous Vehicle Dataset License](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/blob/main/LICENSE.pdf) does not allow the dataset, or anything derived from it, to be redistributed, so camera frames, per-clip tables and the trained model are not included here.

To get access:

1. Create a Hugging Face account and accept NVIDIA's licence on the dataset page.
2. Create an access token (Settings → Access Tokens) with read access.
3. Notebooks 1–3 call `login()` and download only the metadata and data chunks they need.

## Approach

| Stage | What is done |
|---|---|
| **Sampling** | Clips with valid ego-motion and front-camera data are clustered (K-Means) and screened for unusual metadata (Isolation Forest), then sampled with a balance of rare and typical scenarios and a per-stratum country cap. |
| **Ego-motion features** | Speed, acceleration, jerk and braking-event counts computed from each clip's vehicle trajectory. |
| **Visual features** | The middle frame of the front wide-angle camera is processed with **YOLOv8** (object counts), **ResNet-50** and **ViT-B/16** (embeddings reduced with PCA), plus brightness and contrast. |
| **Proxy risk labels** | After an 80/20 train/test split, an **Isolation Forest** is fitted on training clips only, using ego-motion and time-of-day features; the most anomalous 25% are labelled *high risk*, and the same fitted model is applied to the test set. |
| **Modelling** | Dummy, Logistic Regression and Random Forest baselines; **XGBoost** with two-phase hyperparameter tuning and class weighting. |
| **Evaluation** | ROC-AUC, precision/recall/F1, 5-fold cross-validation with labels recreated inside each fold, **SHAP** explanations, and ablation studies that remove feature groups. |

## Pipeline

<p align="center">
  <img src="docs/images/pipeline.png" alt="Pipeline" width="100%"/>
</p>

| # | Notebook | Purpose | Main output |
|---|---|---|---|
| 0 | [`0_exploratory_dataset_study.ipynb`](notebooks/0_exploratory_dataset_study.ipynb) | Explore the dataset's structure and metadata | — |
| 1 | [`1_data_preparation.ipynb`](notebooks/1_data_preparation.ipynb) | Filter, cluster and sample clips | `pipeline_data/01_data_preparation/prepared_dataset_2000.csv` |
| 2 | [`2_ego_motion_features.ipynb`](notebooks/2_ego_motion_features.ipynb) | Compute speed, acceleration, jerk and braking features | `pipeline_data/02_ego_features/ego_features_2000.csv` |
| 3 | [`3_visual_features.ipynb`](notebooks/3_visual_features.ipynb) | Extract frames; YOLOv8, ResNet-50, ViT-B/16, brightness/contrast | `pipeline_data/03_visual_features/visual_features_final.csv` |
| 4 | [`4_dataset_assembly.ipynb`](notebooks/4_dataset_assembly.ipynb) | Merge all features and engineer composite features | `pipeline_data/04_dataset/final_dataset.csv` |
| 5 | [`5_modelling_and_evaluation.ipynb`](notebooks/5_modelling_and_evaluation.ipynb) | Labels, models, tuning, SHAP, ablation | `pipeline_data/05_modelling_outputs/` |

## Repository structure

```
.
├── notebooks/                    # The six pipeline notebooks (run in order)
├── pipeline_data/                # Folders the notebooks read from and write to
│   ├── 01_data_preparation/      # Empty: created by Notebook 1
│   ├── 02_ego_features/          # Empty: created by Notebook 2
│   ├── 03_visual_features/       # Empty: created by Notebook 3
│   ├── 04_dataset/               # Empty: created by Notebook 4
│   └── 05_modelling_outputs/     # Result tables, tuning logs and figures
├── dissertation/
│   ├── report/                   # Dissertation (PDF)
│   └── presentation/             # Presentation slides (PPTX)
├── manuscript/                   # Manuscript drafts (Word, LaTeX, Overleaf project)
└── docs/images/                  # Figures used in this README
```

## Running the notebooks

Clip-level data is not included (see [The data](#the-data)), so the notebooks are run from the start with
your own dataset access. Notebook outputs that displayed dataset rows have been cleared; charts and
printed summaries are kept.

The notebooks were written for **Google Colab** with a **T4 GPU** runtime and **Google Drive** for storage.

1. Copy the `pipeline_data/` folder structure to a folder in your Google Drive.
2. In each notebook, set `base_path` to that folder.
3. Run the notebooks in order (1 → 5). Notebook 0 is optional.

| Notebook | GPU needed | Notes |
|---|---|---|
| 1, 2, 4 | No | Notebook 2 downloads ego-motion chunks |
| 3 | Recommended | Downloads camera chunks and runs YOLOv8, ResNet-50 and ViT-B/16 |
| 5 | Recommended | Hyperparameter tuning and SHAP benefit most from a GPU |

## Results

Results will be summarised here when the manuscript is finalised. The result tables and figures produced
by the pipeline are in [`pipeline_data/05_modelling_outputs/`](pipeline_data/05_modelling_outputs/). The
trained model is not included because it is derived from the dataset.

## Dissertation and manuscript

| Document | Location |
|---|---|
| MSc dissertation (Dublin Business School, May 2026) | [`dissertation/report/`](dissertation/report/) |
| Dissertation presentation | [`dissertation/presentation/`](dissertation/presentation/) |
| Manuscript drafts — Nair & Izima (in preparation) | [`manuscript/`](manuscript/) |

## Authors

**Prasanna Syam Shreyas Nair** — MSc Business Analytics, Dublin Business School · [LinkedIn](https://www.linkedin.com/in/psshreyasnair)

**Obinna Izima** — Dissertation supervisor and manuscript co-author, Dublin Business School · [LinkedIn](https://www.linkedin.com/in/obinna-c-izima-ph-d-52b75524)

## Citation

Citation details for the published article will be added here. To cite the dissertation:

```bibtex
@mastersthesis{nair2026multimodal,
  author = {Nair, Prasanna Syam Shreyas},
  title  = {Multimodal Driving Risk Prediction Using Machine Learning on Autonomous Vehicle Data (NVIDIA Physical AI Dataset)},
  school = {Dublin Business School},
  type   = {MSc Applied Research Project},
  year   = {2026},
  month  = {May}
}
```

## Acknowledgements

This work uses the NVIDIA PhysicalAI Autonomous Vehicles dataset. Thanks to Dublin Business School for
supporting the research and its publication.
