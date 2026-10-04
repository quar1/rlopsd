"""Math prompts, dataset validation and thread-safe official OPSD scoring."""
import re
import threading
import unicodedata
from functools import lru_cache
from math_verify import LatexExtractionConfig, parse
from local.math_m2.official_scoring import score_grpo_response, score_eval_response

_reward_pool = None
_reward_pool_lock = threading.Lock()

TEACHER_CONTEXT = (
    '\n\nHere is a reference solution to this problem:\n'
    '=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n\n'
    'After reading the reference solution above, make sure you truly understand '
    'the reasoning behind each step — do not copy or paraphrase it. Now, using your '
    'own words and independent reasoning, derive the same final answer to the problem above. '
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
    'Please reason step by step, and put your final answer within \\boxed{{}}.'
)
# Used by both RL dataset teacher tokenization and student AgentLoop.
PLAIN_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}{{ '\\n\\nSolution:\\n' }}{% endif %}"


def template_kwargs(tokenizer, prompt_format='plain', thinking=False):
    if prompt_format == 'plain':
        return {'chat_template': PLAIN_TEMPLATE}
    if prompt_format != 'native' or not tokenizer.chat_template:
        raise ValueError('Native prompting requires the model tokenizer chat template')
    return {'chat_template': tokenizer.chat_template, 'enable_thinking': thinking}


def prompt_text(problem):
    return 'Solve the following mathematics problem. Show your reasoning and put your final answer inside \\boxed{}.\n\nProblem:\n' + problem.strip()


def normalize_problem(problem):
    text = unicodedata.normalize('NFKC', problem).casefold()
    for s in ('\\left', '\\right', '\\(', '\\)', '\\[', '\\]', '$'):
        text = text.replace(s, '')
    return re.sub(r'\s+', '', text)


def last_boxed(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    level = 1
    begin = start + len('\\boxed{')
    for i in range(begin, len(text)):
        if text[i] == '{': level += 1
        elif text[i] == '}': level -= 1
        if level == 0:
            return text[begin:i].strip() or None
    return None


@lru_cache(maxsize=65536)
def parse_answer(answer):
    return parse('\\boxed{' + answer + '}', extraction_config=[LatexExtractionConfig()],
                 fallback_mode='no_fallback', extraction_mode='first_match', parsing_timeout=2)


def _score_with_main_thread(solution_str, ground_truth, mode):
    # Ray's async reward manager calls synchronous scorers in executor threads.
    # math_verify's POSIX timeouts use SIGALRM and require a process main thread.
    # Keep bounded parsing/verification in a dedicated spawned CPU worker instead
    # of silently turning signal-handler errors into zero rewards.
    if threading.current_thread() is not threading.main_thread():
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from local.math_m2.reward_worker import score_in_process
        global _reward_pool
        with _reward_pool_lock:
            if _reward_pool is None:
                _reward_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        return _reward_pool.submit(score_in_process, solution_str, str(ground_truth), mode).result()
    scorer = score_grpo_response if mode == 'grpo' else score_eval_response
    return scorer(solution_str, str(ground_truth))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return _score_with_main_thread(solution_str, ground_truth, 'grpo')


def compute_eval_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return _score_with_main_thread(solution_str, ground_truth, 'eval')


def reward_group_metrics(uids, scores):
    """Outer-rollout group composition; scores are final binary sequence rewards."""
    groups = {}
    for uid, score in zip(uids, scores):
        groups.setdefault(str(uid), []).append(float(score))
    total = len(groups)
    if not total:
        return {}
    correct = [sum(values) for values in groups.values()]
    return {
        'math/groups_all_wrong': sum(n == 0 for n in correct) / total,
        'math/groups_all_correct': sum(sum(v) == len(v) for v in groups.values()) / total,
        'math/groups_mixed': sum(0 < sum(v) < len(v) for v in groups.values()) / total,
        'math/group_count': total,
    }
