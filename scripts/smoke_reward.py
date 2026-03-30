def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Minimal reward hook for environment smoke tests.

    The smoke test only needs the full PPO pipeline to execute one step, so we
    return a constant scalar reward regardless of dataset semantics.
    """
    return 0.0
