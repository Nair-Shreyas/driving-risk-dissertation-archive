"""Generate synthetic inputs for the demo.

Writes the three files Notebooks 1-3 normally produce, with the same columns and plausible
ranges, so Notebooks 4 and 5 can run without NVIDIA data. All values are invented.
Usage:  python make_demo_data.py <output_folder> [n_clips]
"""
import sys

# --- generator ---
import os
import uuid
import numpy as np
import pandas as pd


def make_demo_inputs(out_dir, n_clips=600, seed=42):
    """Create synthetic prepared / ego-motion / visual feature tables in out_dir."""
    rng = np.random.default_rng(seed)
    n = n_clips
    clip_id = [str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63)))) for _ in range(n)]
    chunk = rng.integers(0, 3000, n)

    # ---- context (Notebook 1 style) ----
    w = np.array([1, 1, 1, 1, 1, 2, 3, 6, 7, 6, 5, 5, 5, 5, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1], float)
    hour = rng.choice(24, n, p=w / w.sum())
    is_night = ((hour >= 20) | (hour < 7)).astype(int)
    is_peak = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))).astype(int)
    latent = rng.normal(0, 1, n) + 0.9 * is_peak - 0.3 * is_night        # hidden "riskiness"
    scenario = np.where(latent + rng.normal(0, 0.8, n) > 0.4, "rare", "normal")
    prepared = pd.DataFrame({
        "clip_id": clip_id, "chunk": chunk,
        "split": rng.choice(["train", "val", "test"], n),
        "clip_is_valid": True,
        "country": rng.choice(["Country A", "Country B", "Country C", "Country D", "Country E"], n,
                              p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "month": rng.integers(1, 13, n), "hour_of_day": hour,
        "platform_class": rng.choice(["platform_1", "platform_2"], n, p=[0.75, 0.25]),
        "radar_config": rng.choice(["low", "med", "high", None], n, p=[0.2, 0.3, 0.1, 0.4]),
        "camera_front_wide_120fov": True, "egomotion.offline": True,
        "cluster": rng.integers(0, 4, n),
        "anomaly_label": np.where(scenario == "rare", -1, 1),
        "scenario_type": scenario, "is_night": is_night, "is_peak_hour": is_peak,
    })

    # ---- ego-motion (Notebook 2 style; m/s, m/s^2, m/s^3) ----
    avg_speed = np.clip(rng.normal(11, 4, n) + 2.5 * latent, 0.5, 35)
    max_speed = avg_speed + np.abs(rng.normal(4, 2, n)) + 1.5 * np.maximum(latent, 0)
    avg_acc = rng.normal(0, 0.15, n) - 0.08 * latent
    max_acc = np.abs(rng.normal(2.0, 0.8, n)) + 0.6 * np.maximum(latent, 0)
    avg_jerk = rng.normal(0, 0.02, n)
    max_jerk = np.clip(np.abs(rng.normal(20, 12, n)) + 6 * np.maximum(latent, 0), 0, 50)
    braking = rng.poisson(np.clip(4 + 2 * latent, 0.5, None))
    hard_braking = rng.poisson(np.clip(1 + 2.5 * np.maximum(latent, 0), 0.1, None))
    ego = pd.DataFrame({
        "clip_id": clip_id, "avg_speed": avg_speed, "max_speed": max_speed,
        "avg_acceleration": avg_acc, "max_acceleration": max_acc,
        "avg_jerk": avg_jerk, "max_jerk": max_jerk,
        "braking_events": braking, "hard_braking_events": hard_braking, "chunk": chunk,
    })

    # ---- visual (Notebook 3 style) ----
    brightness = np.clip(np.where(is_night == 1, rng.normal(45, 15, n), rng.normal(115, 25, n)), 0, 255)
    contrast = np.clip(rng.normal(45, 12, n), 5, 120)
    counts = {
        "person_count": rng.poisson(0.6, n), "bicycle_count": rng.poisson(0.1, n),
        "car_count": rng.poisson(4.0, n), "motorcycle_count": rng.poisson(0.1, n),
        "bus_count": rng.poisson(0.15, n), "truck_count": rng.poisson(0.5, n),
        "traffic_light_count": rng.poisson(0.6, n), "stop_sign_count": rng.poisson(0.05, n),
    }
    visual = pd.DataFrame({"clip_id": clip_id, "brightness": brightness, "contrast": contrast, **counts})
    visual["total_objects"] = visual[list(counts)].sum(axis=1)
    for prefix in ("cnn", "vit"):                       # PCA-style embeddings: decreasing variance
        for k in range(10):
            visual[f"{prefix}_feature_{k}"] = rng.normal(0, 3.0 / (k + 1), n)

    paths = {
        "01_data_preparation/prepared_dataset_2000.csv": prepared,
        "02_ego_features/ego_features_2000.csv": ego,
        "03_visual_features/visual_features_final.csv": visual,
    }
    for rel, df in paths.items():
        p = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False)
        print(f"wrote {rel}: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Synthetic inputs ready — values are invented, for demonstration only.")


if __name__ == "__main__":
    make_demo_inputs(sys.argv[1] if len(sys.argv) > 1 else "demo_pipeline_data",
                     int(sys.argv[2]) if len(sys.argv) > 2 else 600)
