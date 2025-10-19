import re
import math
from typing import Tuple

# Lightweight complexity estimator

def _norm_log(x: float, base: float = 1024.0, cap: float = 1e12) -> float:
    x = min(max(x, 0.0), cap)
    return min(1.0, math.log(x + 1, base + 1))


def estimate_complexity(prompt: str) -> float:
    """Return a complexity score in [0,1] quickly.

    Components:
    - token factor (context length)
    - size factor (NxM matrices or data sizes / batch sizes)
    - op factor (keywords like train/render/fft)
    """
    if not prompt or not prompt.strip():
        return 0.0
    s = prompt.lower()

    # token factor
    tokens = re.findall(r"\w+", s)
    token_count = len(tokens)
    token_factor = min(1.0, token_count / 2000.0)

    # matrix / data size detection
    matrix_factor = 0.0
    for m in re.finditer(r"(\d{1,6})\s*[x×]\s*(\d{1,6})", s):
        a, b = int(m.group(1)), int(m.group(2))
        area = a * b
        matrix_factor = max(matrix_factor, _norm_log(area, base=4096.0))

    bs = re.search(r"(?:batch\s*size|bs)\s*[:=]?\s*(\d{1,6})", s)
    batch_factor = 0.0
    if bs:
        batch_factor = _norm_log(int(bs.group(1)), base=256.0)

    size_factor = max(matrix_factor, batch_factor)

    # operation keywords
    keywords = {
        r"\btrain\b": 0.9,
        r"\bbackprop\b": 0.95,
        r"\bconvolution\b": 0.8,
        r"\bfft\b": 0.7,
        r"\brender\b": 0.9,
        r"\bray[- ]trace\b": 1.0,
        r"\binference\b": 0.5,
        r"\bsimulat": 0.8,
        r"\boptimi[sz]e\b": 0.6,
        r"\badd\b": 0.05,
        r"\bmean\b": 0.05
    }
    op_score = 0.0
    for patt, score in keywords.items():
        if re.search(patt, s):
            op_score = max(op_score, score)

    final = 0.5 * op_score + 0.35 * size_factor + 0.15 * token_factor
    final = max(0.0, min(1.0, final))
    return float(final)
