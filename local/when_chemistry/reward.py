"""Binary option reward shared by the four SciKnowEval science tasks."""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source not in ("sciknoweval_chemistry", "sciknoweval_biology",
                           "sciknoweval_physics", "sciknoweval_materials"):
        raise ValueError(f"Unexpected data source: {data_source}")
    if ground_truth not in ("A", "B", "C", "D"):
        raise ValueError(f"Invalid ground truth: {ground_truth!r}")
    answers = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.S)
    answer = answers[-1].strip() if answers else ""
    valid = answer in ("A", "B", "C", "D")
    score = float(valid and answer == ground_truth)
    return {"score": score, "acc": score, "format_valid": float(valid)}
