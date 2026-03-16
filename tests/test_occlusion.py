"""Unit tests for occlusion augmentation."""
import sys
import numpy as np
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.occlusion_augmentation import (
    apply_occlusion, apply_medical_mask, apply_breathing_tube,
    apply_partial_occlusion, OcclusionAugmentor, OCCLUSION_TYPES,
)


class TestApplyOcclusion:
    @pytest.fixture
    def frame(self):
        return (np.random.rand(112, 112, 3) * 255).astype(np.uint8)

    def test_none_passthrough(self, frame):
        occ, mask, ratio = apply_occlusion(frame, "none")
        np.testing.assert_array_equal(occ, frame)
        assert ratio == 0.0

    def test_mask_output_shape(self, frame):
        occ, mask, ratio = apply_occlusion(frame, "mask")
        assert occ.shape == frame.shape
        assert mask.shape == frame.shape[:2]

    def test_tube_ratio_higher(self, frame):
        _, _, r_mask = apply_occlusion(frame, "mask")
        _, _, r_tube = apply_occlusion(frame, "tube")
        assert r_tube >= r_mask

    def test_partial_ratio_lower(self, frame):
        _, _, r_partial = apply_occlusion(frame, "partial", occlusion_ratio=0.20)
        _, _, r_mask = apply_occlusion(frame, "mask")
        assert r_partial < r_mask

    def test_mask_binary(self, frame):
        _, mask, _ = apply_occlusion(frame, "mask")
        unique = np.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_occluded_pixels_zero(self, frame):
        occ, mask, _ = apply_occlusion(frame, "mask")
        assert (occ.astype(float)[mask == 1] == 0).all()

    def test_unknown_type_raises(self, frame):
        with pytest.raises(ValueError):
            apply_occlusion(frame, "unknown_type")


class TestOcclusionAugmentor:
    @pytest.fixture
    def augmentor(self):
        return OcclusionAugmentor()

    def test_clip_output_shapes(self, augmentor):
        frames = (np.random.rand(10, 112, 112, 3) * 255).astype(np.uint8)
        occ_frames, masks, occ_type, ratio = augmentor.augment_clip(frames, training=True)
        assert occ_frames.shape == frames.shape
        assert masks.shape == (10, 112, 112)
        assert occ_type in OCCLUSION_TYPES
        assert 0.0 <= ratio <= 1.0

    def test_eval_mode_fixed_type(self, augmentor):
        augmentor.eval_occlusion_type = "mask"
        frames = (np.random.rand(5, 64, 64, 3) * 255).astype(np.uint8)
        _, _, occ_type, _ = augmentor.augment_clip(frames, training=False)
        assert occ_type == "mask"
