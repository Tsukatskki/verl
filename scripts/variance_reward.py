def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """A lightweight reward with enough variation for GRPO smoke tests.

    It only depends on the sampled response text, so it preserves the
    project's existing reward interface while making it much less likely that
    all rollouts in a group receive the exact same scalar reward.
    """
    text = (solution_str or "").strip()
    if not text:
        return -1.0

    char_len = len(text)
    unique_ratio = len(set(text)) / max(char_len, 1)
    line_breaks = text.count("\n")
    digit_count = sum(ch.isdigit() for ch in text)
    punct_count = sum(ch in ",.;:!?，。；：！？" for ch in text)

    score = 0.0
    score += min(char_len / 320.0, 0.8)
    score += unique_ratio * 0.6
    score += min(line_breaks, 4) * 0.05
    score += min(digit_count, 8) * 0.02
    score += min(punct_count, 8) * 0.015

    return float(max(-1.0, min(1.5, score)))
