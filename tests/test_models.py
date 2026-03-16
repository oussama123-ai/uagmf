"""Unit tests for UAG-MF model components."""
import sys
from pathlib import Path
import pytest, torch, numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.occlusion_detector import OcclusionDetector, OCCLUSION_CLASSES
from src.models.generative_reconstruction import UNetGenerator, VAERecon
from src.models.multimodal_fusion import CrossAttentionMultimodalFusion
from src.models.temporal_model import TemporalTransformer
from src.models.uq_layer import DualUQLayer
from src.models.symbolic_engine import SymbolicConflictEngine
from src.models.uagmf import UAGMF

B, T, H, W = 2, 16, 112, 112
D = 256


class TestOcclusionDetector:
    def test_output_keys(self):
        det = OcclusionDetector(pretrained=False)
        x = torch.randn(B, 3, H, W)
        out = det(x)
        for k in ("logits", "occlusion_class", "mask", "occlusion_ratio"):
            assert k in out

    def test_logits_shape(self):
        det = OcclusionDetector(pretrained=False)
        out = det(torch.randn(B, 3, H, W))
        assert out["logits"].shape == (B, len(OCCLUSION_CLASSES))

    def test_ratio_in_range(self):
        det = OcclusionDetector(pretrained=False)
        out = det(torch.randn(B, 3, H, W))
        assert (out["occlusion_ratio"] >= 0).all() and (out["occlusion_ratio"] <= 1).all()


class TestGenerativeReconstruction:
    def test_unet_output_shape(self):
        gen = UNetGenerator(in_channels=4, base_ch=16)
        x = torch.randn(B, 3, H, W)
        mask = torch.zeros(B, 1, H, W)
        out = gen(x, mask)
        assert out.shape == (B, 3, H, W)

    def test_vae_output_shapes(self):
        vae = VAERecon(latent_dim=64)
        x = torch.randn(B, 3, 64, 64)
        mask = torch.zeros(B, 1, 64, 64)
        x_hat, mu, logvar = vae(x, mask)
        assert mu.shape == (B, 64)
        assert logvar.shape == (B, 64)


class TestMultimodalFusion:
    def test_with_all_modalities(self):
        fusion = CrossAttentionMultimodalFusion(video_dim=D, d_model=D, fusion_layers=1)
        video = torch.randn(B, T, D)
        hrv = torch.randn(B, T, 4)
        spo2 = torch.randn(B, T, 1)
        resp = torch.randn(B, T, 1)
        f, delta_u = fusion(video, hrv=hrv, spo2=spo2, resp=resp)
        assert f.shape == (B, D)
        assert delta_u == 0.0  # no absent modalities

    def test_missing_modalities(self):
        fusion = CrossAttentionMultimodalFusion(video_dim=D, d_model=D, fusion_layers=1)
        video = torch.randn(B, T, D)
        f, delta_u = fusion(video)  # all physio absent
        assert f.shape == (B, D)
        assert delta_u == pytest.approx(3 * 0.08)  # 3 absent × 0.08

    def test_gradient_flow(self):
        fusion = CrossAttentionMultimodalFusion(video_dim=D, d_model=D, fusion_layers=1)
        video = torch.randn(B, T, D, requires_grad=True)
        f, _ = fusion(video)
        f.sum().backward()
        assert video.grad is not None


class TestTemporalTransformer:
    def test_output_shape(self):
        t = TemporalTransformer(d_model=D, n_layers=2, n_heads=4, d_ff=D*2)
        x = torch.randn(B, T, D)
        pooled, _ = t(x)
        assert pooled.shape == (B, D)

    def test_padding_mask(self):
        t = TemporalTransformer(d_model=D, n_layers=2, n_heads=4, d_ff=D*2)
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -4:] = True
        pooled_masked, _ = t(x, src_key_padding_mask=mask)
        pooled_full, _ = t(x)
        assert not torch.allclose(pooled_masked, pooled_full)


class TestDualUQLayer:
    def test_output_range(self):
        uq = DualUQLayer(d_model=D, n_mc_samples=5, n_ensemble=3)
        x = torch.randn(B, D)
        out = uq(x)
        assert (out["mu"] >= 0).all() and (out["mu"] <= 10).all()
        assert (out["sigma"] >= 0).all()

    def test_alert_fires_high_variance(self):
        uq = DualUQLayer(d_model=D, n_mc_samples=5, n_ensemble=3, alert_threshold=0.001)
        out = uq(torch.randn(B, D))
        assert out["alert"].all()

    def test_residual_inflation(self):
        uq = DualUQLayer(d_model=D, n_mc_samples=5, n_ensemble=3, gamma=1.0)
        x = torch.randn(B, D)
        out_no_res = uq(x, residual=None)
        big_res = torch.ones(B) * 100.0
        out_with_res = uq(x, residual=big_res)
        assert (out_with_res["var"] > out_no_res["var"]).all()

    def test_delta_u_elevation(self):
        uq = DualUQLayer(d_model=D, n_mc_samples=5, n_ensemble=3)
        x = torch.randn(B, D)
        out_no_delta = uq(x, delta_u=0.0)
        out_with_delta = uq(x, delta_u=0.24)
        assert (out_with_delta["var"] > out_no_delta["var"]).all()


class TestSymbolicEngine:
    def test_tier1_fires(self):
        engine = SymbolicConflictEngine()
        acts = torch.zeros(12)
        acts[0] = 0.9   # c_hrv_up
        acts[4] = 0.9   # c_au4
        acts[9] = 0.9   # c_cry
        y, tier, exp, var = engine.evaluate(acts, current_var=0.1)
        assert tier == 1
        assert y > 0

    def test_tier3_escalates_variance(self):
        engine = SymbolicConflictEngine()
        acts = torch.zeros(12)
        acts[0] = 0.9   # c_hrv_up
        acts[11] = 0.9  # c_neutral (conflict with hrv_up)
        y, tier, exp, updated_var = engine.evaluate(acts, current_var=0.1)
        assert tier == 3
        assert updated_var > 0.35

    def test_batch_evaluate(self):
        engine = SymbolicConflictEngine()
        acts = torch.rand(B, 12)
        vars_ = torch.rand(B) * 0.2
        results = engine.batch_evaluate(acts, vars_)
        assert len(results["y_hat"]) == B
        assert len(results["tier"]) == B


class TestUAGMFPipeline:
    @pytest.fixture
    def small_model(self):
        return UAGMF(d_model=64, n_mc_samples=3, n_ensemble=2)

    def test_forward_shape(self, small_model):
        video = torch.randn(B, T, 3, H, W)
        out = small_model(video)
        assert out["mu"].shape == (B,)
        assert out["sigma"].shape == (B,)
        assert out["var"].shape == (B,)
        assert len(out["alert"]) == B

    def test_output_range(self, small_model):
        video = torch.randn(B, T, 3, H, W)
        out = small_model(video)
        assert (out["mu"] >= 0).all() and (out["mu"] <= 10).all()

    def test_with_physio(self, small_model):
        video = torch.randn(B, T, 3, H, W)
        hrv = torch.randn(B, T, 4)
        spo2 = torch.randn(B, T, 1)
        out = small_model(video, hrv=hrv, spo2=spo2)
        assert out["mu"].shape == (B,)

    def test_intermediates(self, small_model):
        video = torch.randn(B, T, 3, H, W)
        out = small_model(video, return_intermediates=True)
        assert "occlusion_ratio" in out
        assert "residual" in out
        assert "concept_acts" in out
        assert out["concept_acts"].shape == (B, 12)

    def test_parameter_count(self, small_model):
        assert small_model.count_parameters() > 0
        inf_params = small_model.count_parameters(inference_only=True)
        assert inf_params < small_model.count_parameters()

    def test_backward_pass(self, small_model):
        video = torch.randn(B, T, 3, H, W)
        target = torch.rand(B) * 10
        out = small_model(video)
        loss = ((out["mu"] - target) ** 2).mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in small_model.parameters()
                       if p.requires_grad)
        assert has_grad
