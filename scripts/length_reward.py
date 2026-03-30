def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Simple non-constant reward for short end-to-end validation runs.

    Rewards concise but non-empty responses so PPO gets a usable signal during
    connectivity checks without depending on task-specific reward routing.
    """
    text = (solution_str or "").strip()
    if not text:
        return -1.0

    token_count = len(text.split())
    # Prefer responses in a moderate length band for a stable toy signal.
    if token_count < 8:
        return -0.25
    if token_count <= 64:
        return 1.0
    if token_count <= 128:
        return 0.25
    return -0.25
