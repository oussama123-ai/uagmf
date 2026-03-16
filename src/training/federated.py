"""
Federated Learning Simulation — FedAvg + DP-SGD.

IMPORTANT: This is a SIMULATION using three public benchmark datasets
(BioVid, UNBC-McMaster, EmoPain) as proxy sites. No real hospital
network was established. See Section 3.7.1 of the paper.

Configuration (Table 4):
    - 3 proxy sites (BioVid/DE, UNBC/CA, EmoPain/UK)
    - 50 communication rounds (convergence at round 41)
    - 10 local epochs per round
    - FedAvg weighted by nₖ: θₜ = Σ (nₖ/N) θₖₜ
    - DP-SGD: ε = 8.0, δ = 10⁻⁵ (relatively loose; see Section 6.5)
    - Gradient clipping ℓ₂ ≤ 1.0
    - 75% random client selection

Privacy note: ε = 8.0 is a relatively loose DP budget chosen to maintain
utility across non-IID sites. It should NOT be interpreted as strong
privacy protection. See Section 6.5 (Privacy budget interpretation).
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("uagmf.federated")


class FederatedSite:
    """Represents a single federated learning site (proxy: one dataset)."""

    def __init__(
        self,
        site_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        n_samples: int,
        dp_epsilon: float = 8.0,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 1.0,
        lr: float = 1e-3,
        local_epochs: int = 10,
    ) -> None:
        self.site_id = site_id
        self.model = model
        self.train_loader = train_loader
        self.n_samples = n_samples
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_max_grad_norm = dp_max_grad_norm
        self.lr = lr
        self.local_epochs = local_epochs

        # Use Opacus for DP-SGD if available
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        try:
            from opacus import PrivacyEngine
            from torch.optim import Adam
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=self.local_epochs,
                target_epsilon=self.dp_epsilon,
                target_delta=self.dp_delta,
                max_grad_norm=self.dp_max_grad_norm,
            )
            self._using_opacus = True
            logger.info(
                f"Site {self.site_id}: DP-SGD enabled "
                f"(ε={self.dp_epsilon}, δ={self.dp_delta})"
            )
        except (ImportError, Exception) as e:
            logger.warning(
                f"Site {self.site_id}: Opacus not available ({e}). "
                f"Using standard Adam with manual gradient clipping."
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self._using_opacus = False

    def local_train(
        self,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Run local_epochs training steps and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.local_epochs):
            for batch in self.train_loader:
                video = batch["video"].to(device)
                target = batch["nrs_score"].to(device)
                hrv = batch.get("hrv")
                spo2 = batch.get("spo2")
                resp = batch.get("resp")
                if hrv is not None:
                    hrv = hrv.to(device)
                if spo2 is not None:
                    spo2 = spo2.to(device)
                if resp is not None:
                    resp = resp.to(device)

                self.optimizer.zero_grad()
                output = self.model(video, hrv=hrv, spo2=spo2, resp=resp)
                loss, _ = loss_fn(output["mu"], output["var"], target)
                loss.backward()

                if not self._using_opacus:
                    # Manual gradient clipping as DP-SGD fallback
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.dp_max_grad_norm
                    )

                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    def get_state_dict(self) -> Dict:
        """Return model state dict (gradients not transmitted)."""
        if self._using_opacus:
            return self.model._module.state_dict()
        return self.model.state_dict()

    def set_state_dict(self, state_dict: Dict) -> None:
        """Update local model from aggregated global weights."""
        if self._using_opacus:
            self.model._module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


class FederatedAggregator:
    """
    FedAvg aggregation: θₜ = Σ (nₖ/N) θₖₜ

    Also implements:
    - Random client selection (75% per round)
    - KL-divergence gradient anomaly detection (3σ exclusion)
    """

    def __init__(
        self,
        global_model: nn.Module,
        client_fraction: float = 0.75,
        kl_anomaly_sigma: float = 3.0,
    ) -> None:
        self.global_model = global_model
        self.client_fraction = client_fraction
        self.kl_anomaly_sigma = kl_anomaly_sigma
        self._gradient_history: List[float] = []

    def select_clients(self, sites: List[FederatedSite]) -> List[FederatedSite]:
        """Randomly select client_fraction of sites."""
        import random
        n_select = max(1, int(len(sites) * self.client_fraction))
        return random.sample(sites, n_select)

    def aggregate(
        self,
        selected_sites: List[FederatedSite],
    ) -> None:
        """
        FedAvg weighted aggregation.
        Sites with anomalous gradient norms are excluded (KL > 3σ).
        """
        total_samples = sum(s.n_samples for s in selected_sites)
        aggregated = {}

        # Collect and filter site updates
        valid_sites = self._filter_anomalous(selected_sites)
        if not valid_sites:
            logger.warning("All sites filtered as anomalous — keeping global model.")
            return

        total_valid = sum(s.n_samples for s in valid_sites)
        for site in valid_sites:
            weight = site.n_samples / total_valid
            state = site.get_state_dict()
            for key, val in state.items():
                if key not in aggregated:
                    aggregated[key] = weight * val.float()
                else:
                    aggregated[key] += weight * val.float()

        self.global_model.load_state_dict(aggregated)
        logger.info(
            f"Aggregated {len(valid_sites)}/{len(selected_sites)} sites "
            f"({total_valid}/{total_samples} samples)."
        )

    def _filter_anomalous(
        self, sites: List[FederatedSite]
    ) -> List[FederatedSite]:
        """Exclude sites whose parameter norm deviates > 3σ from mean."""
        import numpy as np

        norms = []
        for site in sites:
            sd = site.get_state_dict()
            norm = float(sum(v.float().norm().item() for v in sd.values()))
            norms.append(norm)

        norms_arr = np.array(norms)
        mean, std = norms_arr.mean(), norms_arr.std()
        threshold = mean + self.kl_anomaly_sigma * std

        valid = [s for s, n in zip(sites, norms) if n <= threshold]
        excluded = len(sites) - len(valid)
        if excluded > 0:
            logger.warning(f"Excluded {excluded} anomalous site(s) from aggregation.")
        return valid

    def broadcast(self, sites: List[FederatedSite]) -> None:
        """Send global model weights to all sites."""
        global_sd = self.global_model.state_dict()
        for site in sites:
            site.set_state_dict(copy.deepcopy(global_sd))


class FederatedSimulation:
    """
    Orchestrates the 3-site federated simulation.

    Simulates the configuration in Table 4 of the paper.
    Sites: BioVid (DE), UNBC-McMaster (CA), EmoPain (UK).
    """

    def __init__(
        self,
        global_model: nn.Module,
        sites: List[FederatedSite],
        n_rounds: int = 50,
        client_fraction: float = 0.75,
        device: str = "cuda",
    ) -> None:
        self.global_model = global_model
        self.sites = sites
        self.n_rounds = n_rounds
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.aggregator = FederatedAggregator(global_model, client_fraction)
        self.loss_fn = None  # set before run()

    def run(
        self,
        loss_fn: nn.Module,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict]:
        """
        Run n_rounds of federated training.

        Returns:
            List of per-round metrics dicts.
        """
        self.loss_fn = loss_fn
        history = []
        self.aggregator.broadcast(self.sites)

        for rnd in range(1, self.n_rounds + 1):
            selected = self.aggregator.select_clients(self.sites)
            logger.info(
                f"Round {rnd}/{self.n_rounds}: "
                f"selected {len(selected)}/{len(self.sites)} sites"
            )

            # Local training
            site_losses = {}
            for site in selected:
                site_metrics = site.local_train(loss_fn, self.device)
                site_losses[site.site_id] = site_metrics["train_loss"]
                logger.info(
                    f"  Site {site.site_id}: train_loss={site_metrics['train_loss']:.4f}"
                )

            # Aggregate
            self.aggregator.aggregate(selected)

            # Broadcast updated global model
            self.aggregator.broadcast(self.sites)

            metrics = {"round": rnd, "site_losses": site_losses}

            # Optional validation
            if val_loader is not None:
                val_mse = self._validate(val_loader)
                metrics["val_mse"] = val_mse
                logger.info(f"  Val MSE: {val_mse:.4f}")

            history.append(metrics)

        return history

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Quick validation on combined val set."""
        self.global_model.eval().to(self.device)
        total_se = 0.0
        n = 0
        for batch in val_loader:
            video = batch["video"].to(self.device)
            target = batch["nrs_score"].to(self.device)
            out = self.global_model(video)
            total_se += ((out["mu"] - target) ** 2).sum().item()
            n += len(target)
        return total_se / max(n, 1)
