"""Importable spawn target: math_verify signal timeouts run in a CPU main thread."""
import threading


def score_in_process(solution_str, ground_truth, mode='grpo'):
    assert threading.current_thread() is threading.main_thread()
    from local.math_m2.official_scoring import score_grpo_response, score_eval_response
    assert mode in ('grpo', 'eval')
    scorer = score_grpo_response if mode == 'grpo' else score_eval_response
    return scorer(solution_str, ground_truth)
