"""OPSD official reward/evaluation semantics, with observational diagnostics.

Adapted from https://github.com/siyan-zhao/OPSD at
ae7d2519e94920c4eb6206c0c26de46d9c50abae:
grpo_train.py (training) and eval/evaluate_math.py (evaluation).
Their extraction and fallback rules intentionally differ. Run in a process
main thread: math_verify's default POSIX timeouts require SIGALRM.
"""
import re

from math_verify import parse, verify

UPSTREAM_COMMIT = 'ae7d2519e94920c4eb6206c0c26de46d9c50abae'
TRAIN_RULE = 'opsd_official_grpo_' + UPSTREAM_COMMIT
EVAL_RULE = 'opsd_official_eval_' + UPSTREAM_COMMIT


def extract_grpo_answer(text):
    """First boxed answer after the last </think>, matching official GRPO."""
    think_end = text.rfind('</think>')
    search_text = text[think_end + len('</think>'):] if think_end != -1 else text
    idx = search_text.find(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth, i = 1, start
    while i < len(search_text) and depth > 0:
        if search_text[i] == '{':
            depth += 1
        elif search_text[i] == '}':
            depth -= 1
        i += 1
    return search_text[start:i - 1].strip() if depth == 0 else None


def preprocess_grpo_answer(answer):
    if answer is None:
        return None
    ratio = re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*', answer)
    if ratio:
        return rf'\frac{{{ratio.group(1)}}}{{{ratio.group(2)}}}'
    return answer


def _record(score, answer, parse_failed, verify_exception=0., string_match_recovered=0.):
    return dict(score=float(score), acc=float(score), format_valid=float(answer is not None),
                no_box=float(answer is None), parse_failed=float(parse_failed),
                verify_exception=float(verify_exception),
                string_match_recovered=float(string_match_recovered))


def score_grpo_response(text, ground_truth):
    """Scalar equivalent of official reward_correctness; no new heuristics."""
    answer = extract_grpo_answer(text)
    score, verify_exception, recovered = 0., 0., 0.
    # Preserve official default extraction, fallback and timeouts. Unexpected
    # parse errors propagate, rather than being silently relabelled as wrong.
    gold = parse(ground_truth)
    pred = parse(preprocess_grpo_answer(answer))
    if gold is not None and pred is not None:
        try:
            score = 1. if verify(gold, pred) else 0.
        except Exception:
            verify_exception = 1.
    if score == 0.:
        pred_norm = re.sub(r'\s+', '', answer or '').lower()
        gold_norm = re.sub(r'\s+', '', ground_truth or '').lower()
        if pred_norm and pred_norm == gold_norm:
            score, recovered = 1., 1.
    return _record(score, answer, not (bool(gold) and bool(pred)), verify_exception, recovered)


def extract_eval_answer(text):
    """Official evaluation's last-box extraction (including its strict syntax)."""
    idx = text.rfind('\\boxed')
    if idx < 0:
        return None
    i, depth, end = idx, 0, None
    while i < len(text):
        if text[i] == '{':
            depth += 1
        if text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end is None:
        return None
    boxed = text[idx:end + 1]
    if boxed.startswith('\\boxed{') and boxed.endswith('}'):
        return boxed[7:-1].strip()
    return None


def score_eval_response(text, ground_truth):
    """Official grade_answer, plus diagnostics that do not change the score."""
    answer = extract_eval_answer(text)
    if answer is None:
        return _record(0., answer, True)
    predicted, gold_text = answer, ground_truth
    parsed_ok = False
    try:
        if '$' not in predicted:
            predicted = f'${predicted}$'
        if '$' not in gold_text:
            gold_text = f'${gold_text}$'
        pred = parse(predicted, fallback_mode='no_fallback')
        gold = parse(gold_text, fallback_mode='no_fallback')
        parsed_ok = bool(pred) and bool(gold)
        score = verify(gold, pred, timeout_seconds=5)
        return _record(score, answer, not parsed_ok)
    except Exception:
        # Official evaluation falls back only on exception, unlike GRPO.
        pred_norm = predicted.replace('$', '').replace(' ', '').lower().strip()
        gold_norm = gold_text.replace('$', '').replace(' ', '').lower().strip()
        score = pred_norm == gold_norm
        return _record(score, answer, not parsed_ok, 1., float(score))
