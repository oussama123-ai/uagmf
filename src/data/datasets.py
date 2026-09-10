def get_dataset(
    name: str,
    root: str,
    split: str = "all",
    n_frames: int = 64,
    stride: int = 32,
    augment: bool = False,
) -> PainClipDataset:
    """
    Factory function to load a dataset by name.

    Args:
        name: one of {"biovid", "unbc", "emopain", "mdnpl"}.
        root: path to preprocessed features directory.
        split: "all" (for LOSO) or "train"/"test" (for 5-fold).
        n_frames: frames per clip.
        stride: clip stride.
        augment: apply training augmentation.

    Returns:
        PainClipDataset instance.
    """
    if name not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_SPECS.keys())}"
        )

    spec = DATASET_SPECS[name]
    n_subjects = spec["n_subjects"]

    # Build subject ID list matching the repository convention
    if name == "mdnpl":
        prefix = "neonate"
    else:
        prefix = "subject"
    subjects = [f"{prefix}_{i:03d}" for i in range(1, n_subjects + 1)]

    # If split != "all", filter to a subset (used by 5-fold path)
    if split == "train":
        subjects = subjects[: int(0.8 * len(subjects))]
    elif split == "test":
        subjects = subjects[int(0.8 * len(subjects)):]

    return PainClipDataset(
        feature_dir=root,
        subjects=subjects,
        dataset_name=name,
        n_frames=n_frames,
        stride=stride,
        augment=augment,
    )