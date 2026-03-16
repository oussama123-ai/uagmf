#!/usr/bin/env python3
"""
Three-site federated simulation script (BioVid + UNBC + EmoPain as proxy sites).
See Section 3.7 and Table 4 of the paper.

IMPORTANT: This is a SIMULATION. No real hospital network.
ε = 8.0 is a relatively loose DP budget. See Section 6.5.

Usage:
    python scripts/federated_sim.py \\
        --data_root data/features \\
        --output_dir experiments/federated \\
        --rounds 50 --local_epochs 10 \\
        --dp_epsilon 8.0 --dp_delta 1e-5
"""

import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.data.datasets import build_dataloaders
from src.models.uagmf import UAGMF
from src.training.federated import FederatedSite, FederatedSimulation
from src.training.losses import UAGMFLoss
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("uagmf.federated_sim")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="experiments/federated")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=10)
    p.add_argument("--dp_epsilon", type=float, default=8.0)
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--client_fraction", type=float, default=0.75)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir, run_name="federated_sim")
    logger.warning(
        "SIMULATION: Using public benchmarks as proxy sites. "
        f"DP budget ε={args.dp_epsilon} is relatively loose. "
        "See Section 6.5 of the paper."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = UAGMF()
    loss_fn = UAGMFLoss()
    SITE_DATASETS = ["biovid", "unbc", "emopain"]
    sites = []

    for ds_name in SITE_DATASETS:
        try:
            train_loader, _ = build_dataloaders(
                args.data_root, fold=0, dataset_name=ds_name, batch_size=16
            )
            site_model = UAGMF()
            site = FederatedSite(
                site_id=ds_name, model=site_model,
                train_loader=train_loader,
                n_samples=len(train_loader.dataset),
                dp_epsilon=args.dp_epsilon, dp_delta=args.dp_delta,
                local_epochs=args.local_epochs,
            )
            sites.append(site)
            logger.info(f"Registered site: {ds_name} "
                        f"({len(train_loader.dataset)} samples)")
        except Exception as e:
            logger.warning(f"Could not load site {ds_name}: {e}")

    if not sites:
        logger.error("No sites loaded. Check --data_root.")
        return

    sim = FederatedSimulation(
        global_model=global_model, sites=sites,
        n_rounds=args.rounds,
        client_fraction=args.client_fraction,
        device=device,
    )
    history = sim.run(loss_fn)

    import json
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "federated_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Federated simulation complete. History saved.")


if __name__ == "__main__":
    main()
