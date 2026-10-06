# About the reproducible (v2) notebooks

The notebooks in `reproducible/notebooks/` are version 2 of the pipeline: an optimised, more robust and
cleaner version of the archived notebooks in [`notebooks/`](../notebooks/). The method and pipeline design
are unchanged.

v2 includes a few optimisations, fixes a few bugs found while re-running the pipeline, and is more robust
to changes in the NVIDIA dataset, which is updated over time. Changes are marked in the code with
`# [v2]` comments.

Because of these updates, a fresh run of the v2 notebooks will not give exactly the numbers reported in
the archived dissertation.

## Built for reproducibility

- **One settings cell.** A single **SETTINGS** cell at the top of each notebook is the only cell to edit.
- **Runs on Colab or locally.** Google Drive is mounted only on Colab; locally the notebooks use
  `./pipeline_data`. Output folders are created automatically.
- **Simple data access.** The Hugging Face login picks up an `HF_TOKEN` secret or environment variable.
- **One-click Colab.** Every notebook has an "Open in Colab" button.
- **Quick check mode.** Notebook 5 has a `FAST_MODE` switch (10× fewer tuning iterations).
- **Up to date.** Compatible with current versions of the libraries used.
- **Fresh start.** Stored outputs are cleared, so each notebook runs from scratch.
- **No data needed to try it.** A 5-minute [demo](demo/demo_pipeline.ipynb) runs the modelling pipeline on
  synthetic data.
