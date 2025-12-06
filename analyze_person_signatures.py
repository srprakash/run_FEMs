#!/usr/bin/env python3
"""
Analyze a person_signatures.pkl file:
1) List distinct individuals and number of snapshots.
2) For each person:
   - Extract 32-bin grayscale histogram from every snapshot (first 32 dims of 128-dim vector)
   - Overlay histograms in a figure
   - Compute and display pairwise cosine similarities between snapshots
3) Compute and display cosine similarities between averaged histograms of each person
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_signatures(signature_file: Path) -> Dict:
    if not signature_file.exists():
        raise FileNotFoundError(f"Signature file not found: {signature_file}")
    with open(signature_file, "rb") as f:
        data = pickle.load(f)
    # Handle two possible layouts:
    # - {'known_persons': {...}, ...}
    # - direct dict of persons (legacy)
    if isinstance(data, dict) and "known_persons" in data:
        return data["known_persons"]
    return data


def extract_histograms(features: List[np.ndarray]) -> np.ndarray:
    """
    Extract 32-bin histograms from a list of feature vectors.
    The current feature vectors are 128-dim with the first 32 dims being the histogram.
    """
    if not features:
        return np.empty((0, 32), dtype=float)
    # Ensure each feature is a 1D array and slice first 32 elements
    hists = []
    for vec in features:
        if vec is None:
            continue
        arr = np.asarray(vec).reshape(-1)
        if arr.size < 32:
            # Pad if needed
            arr = np.pad(arr, (0, 32 - arr.size))
        hists.append(arr[:32])
    if not hists:
        return np.empty((0, 32), dtype=float)
    return np.vstack(hists)


def plot_overlaid_histograms(identifier: str, hists: np.ndarray, save_dir: Path = None, show: bool = True):
    if hists.size == 0:
        print(f"  No histograms to plot for '{identifier}'")
        return
    plt.figure(figsize=(8, 5))
    bins = np.arange(32)
    for i in range(hists.shape[0]):
        plt.plot(bins, hists[i], alpha=0.35)
    plt.title(f"Overlaid 32-bin histograms - {identifier} (n={hists.shape[0]})")
    plt.xlabel("Bin (0-31)")
    plt.ylabel("Normalized frequency")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{safe_name(identifier)}_histograms.png"
        plt.savefig(out_path, dpi=150)
        print(f"  Saved histogram overlay to: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_similarity_matrix(identifier: str, hists: np.ndarray, save_dir: Path = None, show: bool = True):
    if hists.shape[0] <= 1:
        print(f"  Not enough snapshots to compute pairwise similarities for '{identifier}'")
        return
    sims = cosine_similarity(hists, hists)
    print(f"  Pairwise cosine similarities for '{identifier}' (shape={sims.shape}):")
    # Print a small textual summary (min/max/mean)
    tri = sims[np.triu_indices_from(sims, k=1)]
    if tri.size > 0:
        print(f"    min={tri.min():.4f}, max={tri.max():.4f}, mean={tri.mean():.4f}")
    else:
        print("    only self-similarities available")
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(sims, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Pairwise snapshot similarities - {identifier}")
    plt.xlabel("Snapshot index")
    plt.ylabel("Snapshot index")
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{safe_name(identifier)}_pairwise_similarities.png"
        plt.savefig(out_path, dpi=150)
        print(f"  Saved pairwise similarity matrix to: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def compute_person_means(known_persons: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    person_to_mean = {}
    for identifier, pdata in known_persons.items():
        feats = pdata.get("features", [])
        hists = extract_histograms(feats)
        if hists.size == 0:
            continue
        mean_hist = hists.mean(axis=0)
        # Normalize mean to unit L2 length for cosine comparability
        norm = np.linalg.norm(mean_hist) + 1e-12
        person_to_mean[identifier] = mean_hist / norm
    return person_to_mean


def plot_between_person_similarities(person_to_mean: Dict[str, np.ndarray], save_dir: Path = None, show: bool = True):
    if not person_to_mean:
        print("No person means available for between-person similarities.")
        return
    people = list(person_to_mean.keys())
    mat = np.stack([person_to_mean[p] for p in people], axis=0)
    sims = cosine_similarity(mat, mat)
    print("\nCosine similarities between averaged histograms of persons:")
    # Print a compact table
    header = " " * 18 + " ".join(f"{p[:12]:>14}" for p in people)
    print(header)
    for i, p in enumerate(people):
        row = " ".join(f"{sims[i, j]:>14.4f}" for j in range(len(people)))
        print(f"{p[:16]:>16}  {row}")
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(sims, vmin=0.0, vmax=1.0, cmap="magma")
    plt.xticks(ticks=np.arange(len(people)), labels=[truncate(p, 16) for p in people], rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(people)), labels=[truncate(p, 16) for p in people])
    plt.colorbar(label="Cosine similarity")
    plt.title("Between-person cosine similarities (averaged histograms)")
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "between_person_similarities.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved between-person similarity matrix to: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (name or "unknown"))


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def main():
    parser = argparse.ArgumentParser(description="Analyze person_signatures.pkl contents")
    parser.add_argument(
        "--signature-file",
        "-s",
        type=str,
        required=True,
        help="Path to person_signatures.pkl",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save figures (if not provided, figures are only shown)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open figure windows (useful for headless runs); saves if --save-dir is provided",
    )
    args = parser.parse_args()

    signature_path = Path(args.signature_file)
    save_dir = Path(args.save_dir) if args.save_dir else None
    show = not args.no_show

    print(f"Loading signatures from: {signature_path}")
    known_persons = load_signatures(signature_path)
    if not isinstance(known_persons, dict) or not known_persons:
        print("No known persons found.")
        return

    # 1) List individuals and number of snapshots
    print("\nIndividuals and snapshot counts:")
    summary: List[Tuple[str, int]] = []
    for identifier, pdata in known_persons.items():
        feats = pdata.get("features", [])
        count = len(feats) if feats is not None else 0
        summary.append((identifier, count))
    summary.sort(key=lambda x: x[0])
    for identifier, count in summary:
        print(f"  {identifier}: {count} snapshot(s)")

    # 2) For each person: plot overlay histograms, compute pairwise similarities
    print("\nPer-person analysis:")
    per_person_dir = save_dir / "per_person" if save_dir else None
    for identifier, pdata in known_persons.items():
        print(f"- {identifier}")
        feats = pdata.get("features", [])
        hists = extract_histograms(feats)
        print(f"  snapshots: {hists.shape[0]}")
        plot_overlaid_histograms(identifier, hists, save_dir=per_person_dir, show=show)
        plot_similarity_matrix(identifier, hists, save_dir=per_person_dir, show=show)

    # 3) Between-person averaged histogram similarities
    person_to_mean = compute_person_means(known_persons)
    plot_between_person_similarities(person_to_mean, save_dir=save_dir, show=show)


if __name__ == "__main__":
    main()



