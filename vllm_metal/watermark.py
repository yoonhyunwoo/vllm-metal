# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLM watermarking via green/red-list logit biasing."""

import argparse
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

@dataclass(frozen=True)
class WatermarkConfig:
    delta: float = 2.0
    gamma: float = 0.5
    seed: int = 42

def _make_rng(seed: int, prev_token: int) -> random.Random:
    return random.Random(seed * 1_000_003 + prev_token)

def _green_mask(seed: int, prev_token: int, vocab_size: int, gamma: float, device):
    import torch
    rng = _make_rng(seed, prev_token)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(rng.randint(0, 2 ** 31 - 1))
    token_scores = torch.rand(vocab_size, generator=generator)
    threshold = gamma
    mask = token_scores < threshold
    if str(device) != 'cpu':
        mask = mask.to(device)
    return mask

def _green_token_set(seed: int, prev_token: int, vocab_size: int, gamma: float) -> frozenset:
    import torch
    mask = _green_mask(seed, prev_token, vocab_size, gamma, 'cpu')
    return frozenset(mask.nonzero(as_tuple=True)[0].tolist())

class WatermarkLogitsProcessor:

    def __init__(self, config: WatermarkConfig):
        self.config = config
        self._mask_cache: dict = {}

    def _get_mask(self, prev_token, vocab_size, device):
        cache_key = (prev_token, vocab_size, str(device))
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = _green_mask(self.config.seed, prev_token, vocab_size, self.config.gamma, device)
        return self._mask_cache[cache_key]

    def __call__(self, token_ids, logits):
        import torch
        if isinstance(token_ids, torch.Tensor):
            if token_ids.numel() == 0:
                return logits
            prev_tokens = token_ids[:, -1].tolist()
        else:
            if not token_ids:
                return logits
            prev_tokens = [token_ids[-1]]
        vocab_size = logits.shape[-1]
        for (row_idx, prev_token) in enumerate(prev_tokens):
            mask = self._get_mask(prev_token, vocab_size, logits.device)
            row = logits[row_idx] if logits.dim() > 1 else logits
            row[mask] = row[mask] + self.config.delta
        return logits

@dataclass
class DetectResult:
    is_watermarked: bool
    z_score: float
    green_ratio: float
    num_tokens: int
    z_threshold: float = 4.0

    def __str__(self) -> str:
        verdict = 'WATERMARKED' if self.is_watermarked else 'not watermarked'
        return f'{verdict} | z={self.z_score:.2f} green_ratio={self.green_ratio:.1%} tokens={self.num_tokens} (threshold z>{self.z_threshold})'

class WatermarkDetector:

    def __init__(self, config: WatermarkConfig, tokenizer, z_threshold: float=4.0):
        self.config = config
        self.tokenizer = tokenizer
        self._vocab_size = tokenizer.vocab_size
        self._cache: dict = {}
        self.z_threshold = z_threshold

    def _get_token_set(self, prev_token: int) -> frozenset:
        if prev_token not in self._cache:
            self._cache[prev_token] = _green_token_set(self.config.seed, prev_token, self._vocab_size, self.config.gamma)
        return self._cache[prev_token]

    def detect(self, text: str) -> DetectResult:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.detect_tokens(token_ids)

    def detect_tokens(self, token_ids: Sequence[int]) -> DetectResult:
        n = len(token_ids)
        if n < 2:
            return DetectResult(is_watermarked=False, z_score=0.0, green_ratio=0.0, num_tokens=n)
        green_count = 0
        for i in range(1, n):
            prev_token = token_ids[i - 1]
            current_token = token_ids[i]
            token_set = self._get_token_set(prev_token)
            if current_token in token_set:
                green_count += 1
        num_pairs = n - 1
        expected_ratio = self.config.gamma
        expected = num_pairs * expected_ratio
        std_dev = math.sqrt(num_pairs * expected_ratio * (1 - expected_ratio))
        z_score = (green_count - expected) / std_dev if std_dev > 0 else 0.0
        return DetectResult(is_watermarked=z_score > self.z_threshold, z_score=z_score, green_ratio=green_count / num_pairs, num_tokens=n)

def _cli_detect():
    parser = argparse.ArgumentParser(description='Detect LLM watermark in text (Kirchenbauer et al. 2023).')
    parser.add_argument('--model', required=True, help='HuggingFace model name')
    parser.add_argument('--text', help='Text to analyze')
    parser.add_argument('--text-file', help='File containing text to analyze')
    parser.add_argument('--seed', type=int, default=42, help='Watermark seed')
    parser.add_argument('--gamma', type=float, default=0.5, help='Biased token percentage (default: 50)')
    parser.add_argument('--bias-weight', type=float, default=2.0, help='Bias weight (informational)')
    parser.add_argument('--z-threshold', type=float, default=4.0, help='z-score threshold')
    args = parser.parse_args()
    text = args.text
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
    if not text:
        parser.error('Provide --text or --text-file')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = WatermarkConfig(seed=args.seed, gamma=args.gamma, delta=args.delta)
    detector = WatermarkDetector(config, tokenizer)
    detector.z_threshold = args.z_threshold
    result = detector.detect(text)
    print(result)
if __name__ == '__main__':
    _cli_detect()