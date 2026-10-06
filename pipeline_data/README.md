# pipeline_data

The notebooks read and write their files in these folders. Folders `01` to `04` are empty here on
purpose: they hold clip-level data derived from the NVIDIA PhysicalAI Autonomous Vehicles dataset,
which its licence does not allow to be redistributed. Running Notebooks 1 to 4 with your own
dataset access recreates them.

| Folder | Created by | Contents after running |
|---|---|---|
| `01_data_preparation/` | Notebook 1 | Sampled clip list with metadata |
| `02_ego_features/` | Notebook 2 | Ego-motion features per clip |
| `03_visual_features/` | Notebook 3 | Extracted frames, visual features, extraction logs |
| `04_dataset/` | Notebook 4 | Assembled modelling table |
| `05_modelling_outputs/` | Notebook 5 | Result tables and figures (included); trained model (not included) |
