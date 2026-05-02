"""
split_dataset.py

Organizes the 60 collected bowl-on-plate demos into train/test splits using
symlinks (no file copying). Randomly selects holdout episodes from across the
full collection order so different scene configurations are represented.

Usage (run on login node, no GPU needed):
    python experiments/robot/franka/split_dataset.py [--source_dir DIR] [--n_test N] [--seed S]

Defaults:
    source_dir : /users/mfenner1/scratch/datasets/bowl_on_plate_pickplace_60
    n_test     : 7
    seed       : 42

Output:
    <source_dir>/train/episode_* -> symlinks to source episodes
    <source_dir>/test/episode_*  -> symlinks to source episodes

The source episodes must be direct children of source_dir named episode_*.
If the zip extracted with a deep path, pass the inner directory as --source_dir.
"""

import argparse
import os
import random
from pathlib import Path


def find_episodes(source_dir: Path) -> list[Path]:
    """Find all episode_* directories directly under source_dir."""
    episodes = sorted([
        p for p in source_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    ])
    return episodes


def make_split(source_dir: Path, output_dir: Path, n_test: int, seed: int):
    episodes = find_episodes(source_dir)
    if not episodes:
        raise RuntimeError(
            f"No episode_* directories found in {source_dir}.\n"
            "If the zip extracted with a deep path, pass the inner dir with --source_dir."
        )

    print(f"Found {len(episodes)} episodes in {source_dir}")

    rng = random.Random(seed)
    test_indices = sorted(rng.sample(range(len(episodes)), n_test))
    test_set = {episodes[i] for i in test_indices}
    train_set = [ep for ep in episodes if ep not in test_set]
    test_list = [episodes[i] for i in test_indices]

    print(f"\nTest episodes ({n_test}) [indices in collection order: {test_indices}]:")
    for ep in test_list:
        print(f"  {ep.name}")

    print(f"\nTrain episodes ({len(train_set)}):")
    for ep in train_set:
        print(f"  {ep.name}")

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Clear any existing symlinks so stale entries from prior runs don't linger
    for d in (train_dir, test_dir):
        for existing in d.iterdir():
            if existing.is_symlink():
                existing.unlink()

    for ep in train_set:
        link = train_dir / ep.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(ep.resolve())

    for ep in test_list:
        link = test_dir / ep.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(ep.resolve())

    print(f"\nDone.")
    print(f"  Train: {train_dir}  ({len(train_set)} episodes)")
    print(f"  Test:  {test_dir}  ({len(test_list)} episodes)")
    print("\nNote: train/ and test/ contain symlinks — the source episodes are not moved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default="/users/mfenner1/scratch/datasets/media/mingxi/daaata1/duo_policy/data/raw_datasets/bowl_on_plate_pickplace_60",
        help="Directory containing episode_* subdirectories (the raw extracted path)",
    )
    parser.add_argument(
        "--output_dir",
        default="/users/mfenner1/scratch/datasets/bowl_on_plate_pickplace_60",
        help="Where to create train/ and test/ symlink directories",
    )
    parser.add_argument("--n_test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_split(Path(args.source_dir), Path(args.output_dir), args.n_test, args.seed)
