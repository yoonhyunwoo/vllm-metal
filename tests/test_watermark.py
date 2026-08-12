# SPDX-License-Identifier: Apache-2.0
"""Tests for watermark embedding and detection.

These tests import the watermark module directly, bypassing the vllm package
__init__ (which requires Python 3.10+). On Python 3.10+ with vllm installed,
change the import to: from vllm_metal.watermark import ...

Run: python -m pytest tests/test_watermark.py -v --noconftest -p no:cacheprovider
"""

import importlib.util
import math
import os
import random
import sys

import pytest
import torch

# --- Load watermark module without triggering vllm/__init__.py -------------

_WATERMARK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vllm_metal", "watermark.py"
)
_spec = importlib.util.spec_from_file_location(
    "vllm_metal.watermark", os.path.abspath(_WATERMARK_PATH)
)
wm = importlib.util.module_from_spec(_spec)
sys.modules["vllm_metal.watermark"] = wm
_spec.loader.exec_module(wm)

WatermarkConfig = wm.WatermarkConfig
WatermarkLogitsProcessor = wm.WatermarkLogitsProcessor
WatermarkDetector = wm.WatermarkDetector
DetectResult = wm.DetectResult
_green_mask = wm._green_mask
_green_token_set = wm._green_token_set
_make_rng = wm._make_rng


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer for detection tests."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % self.vocab_size for c in text]


@pytest.fixture
def cfg():
    return WatermarkConfig(delta=2.0, gamma=0.5, seed=42)


@pytest.fixture
def tokenizer():
    return MockTokenizer(vocab_size=1000)


# ---------------------------------------------------------------------------
# Green/red partition
# ---------------------------------------------------------------------------

class TestGreenRedPartition:
    def test_partition_size(self, cfg):
        green = _green_token_set(cfg.seed, 5, 1000, cfg.gamma)
        assert 400 <= len(green) <= 600

    def test_partition_changes_with_prev_token(self, cfg):
        g1 = _green_token_set(cfg.seed, 1, 1000, cfg.gamma)
        g2 = _green_token_set(cfg.seed, 2, 1000, cfg.gamma)
        assert g1 != g2

    def test_partition_reproducible(self, cfg):
        g1 = _green_token_set(cfg.seed, 7, 1000, cfg.gamma)
        g2 = _green_token_set(cfg.seed, 7, 1000, cfg.gamma)
        assert g1 == g2

    def test_different_seed_different_partition(self, cfg):
        other = WatermarkConfig(seed=999, gamma=cfg.gamma)
        g1 = _green_token_set(cfg.seed, 3, 1000, cfg.gamma)
        g2 = _green_token_set(other.seed, 3, 1000, other.gamma)
        assert g1 != g2

    def test_gamma_controls_size(self):
        cfg_half = WatermarkConfig(gamma=0.5, seed=42)
        cfg_quarter = WatermarkConfig(gamma=0.25, seed=42)
        g_half = _green_token_set(cfg_half.seed, 5, 1000, cfg_half.gamma)
        g_quarter = _green_token_set(cfg_quarter.seed, 5, 1000, cfg_quarter.gamma)
        assert len(g_half) > len(g_quarter)

    def test_make_rng_deterministic(self):
        rng1 = _make_rng(seed=42, prev_token=5)
        rng2 = _make_rng(seed=42, prev_token=5)
        assert rng1.random() == rng2.random()


# ---------------------------------------------------------------------------
# Logits processor
# ---------------------------------------------------------------------------

class TestWatermarkLogitsProcessor:
    def test_bias_adds_delta_to_green(self, cfg):
        processor = WatermarkLogitsProcessor(cfg)
        logits = torch.zeros(500)
        green = _green_token_set(cfg.seed, 10, 500, cfg.gamma)
        result = processor([10], logits)
        for i in range(500):
            if i in green:
                assert result[i].item() == pytest.approx(cfg.delta)
            else:
                assert result[i].item() == pytest.approx(0.0)

    def test_empty_token_ids_no_change(self, cfg):
        processor = WatermarkLogitsProcessor(cfg)
        logits = torch.randn(500)
        result = processor([], logits)
        assert torch.equal(result, logits)

    def test_cache_hit(self, cfg):
        processor = WatermarkLogitsProcessor(cfg)
        logits = torch.zeros(500)
        processor([5], logits.clone())
        assert len(processor._mask_cache) == 1
        processor([5], logits.clone())
        assert len(processor._mask_cache) == 1

    def test_different_prev_different_bias(self, cfg):
        processor = WatermarkLogitsProcessor(cfg)
        r1 = processor([1], torch.zeros(500))
        r2 = processor([2], torch.zeros(500))
        b1 = set(torch.nonzero(r1, as_tuple=True)[0].tolist())
        b2 = set(torch.nonzero(r2, as_tuple=True)[0].tolist())
        assert b1 != b2


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TestWatermarkDetector:
    def test_short_text_not_flagged(self, cfg, tokenizer):
        detector = WatermarkDetector(cfg, tokenizer)
        result = detector.detect("hi")
        assert not result.is_watermarked

    def test_unwatermarked_not_flagged(self, cfg, tokenizer):
        random.seed(123)
        tokens = [random.randint(0, 999) for _ in range(500)]
        detector = WatermarkDetector(cfg, tokenizer)
        result = detector.detect_tokens(tokens)
        assert 0.40 <= result.green_ratio <= 0.60
        assert not result.is_watermarked

    def test_watermarked_detected(self, cfg, tokenizer):
        random.seed(42)
        tokens = [random.randint(0, 999)]
        for _ in range(500):
            prev = tokens[-1]
            green = _green_token_set(cfg.seed, prev, 1000, cfg.gamma)
            if random.random() < 0.80:
                tokens.append(random.choice(list(green)))
            else:
                tokens.append(random.randint(0, 999))
        detector = WatermarkDetector(cfg, tokenizer)
        result = detector.detect_tokens(tokens)
        assert result.is_watermarked
        assert result.green_ratio > 0.65
        assert result.z_score > 4.0

    def test_wrong_seed_not_detected(self, cfg, tokenizer):
        random.seed(42)
        tokens = [random.randint(0, 999)]
        for _ in range(500):
            prev = tokens[-1]
            green = _green_token_set(cfg.seed, prev, 1000, cfg.gamma)
            if random.random() < 0.80:
                tokens.append(random.choice(list(green)))
            else:
                tokens.append(random.randint(0, 999))
        wrong = WatermarkDetector(
            WatermarkConfig(seed=999, gamma=0.5), tokenizer
        )
        result = wrong.detect_tokens(tokens)
        assert not result.is_watermarked

    def test_z_score_formula(self, cfg, tokenizer):
        tokens = list(range(1000))
        detector = WatermarkDetector(cfg, tokenizer)
        result = detector.detect_tokens(tokens)
        n = len(tokens) - 1
        expected = n * cfg.gamma
        std = math.sqrt(n * cfg.gamma * (1 - cfg.gamma))
        manual_z = (result.green_ratio * n - expected) / std
        assert result.z_score == pytest.approx(manual_z, rel=1e-4)

    def test_custom_z_threshold(self, cfg, tokenizer):
        detector = WatermarkDetector(cfg, tokenizer, z_threshold=100.0)
        random.seed(42)
        tokens = [random.randint(0, 999)]
        for _ in range(300):
            prev = tokens[-1]
            green = _green_token_set(cfg.seed, prev, 1000, cfg.gamma)
            if random.random() < 0.80:
                tokens.append(random.choice(list(green)))
            else:
                tokens.append(random.randint(0, 999))
        result = detector.detect_tokens(tokens)
        assert not result.is_watermarked


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_processor_then_detect(self, cfg, tokenizer):
        random.seed(0)
        processor = WatermarkLogitsProcessor(cfg)
        tokens = [random.randint(0, 999)]
        for _ in range(300):
            logits = torch.randn(1000)
            biased = processor([tokens[-1]], logits.clone())
            probs = torch.softmax(biased, dim=-1)
            tokens.append(torch.multinomial(probs, 1).item())
        detector = WatermarkDetector(cfg, tokenizer)
        result = detector.detect_tokens(tokens)
        assert result.is_watermarked, (
            f"z={result.z_score:.2f}, green={result.green_ratio:.1%}"
        )
