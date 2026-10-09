"""Human-readable step losses and complete sampled question groups."""
import csv
import json
from pathlib import Path


def save_step_losses(path, step, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(step=step,
        rl_loss=metrics['actor/global_token_rl_loss'],
        opsd_kl=metrics['actor/global_token_kl_distill'],
        opsd_loss_weighted=metrics['actor/global_token_kd_loss_weighted'],
        beta=metrics['actor/beta_effective'])
    row['student_loss'] = row['rl_loss'] + row['opsd_loss_weighted']
    row['opsd_loss_unweighted'] = row['opsd_loss_weighted']/row['beta']
    row['reward_mean'] = metrics['critic/score/mean']
    row['response_length_mean'] = metrics['response_length/mean']
    row['truncation_rate'] = metrics['response_length/clip_ratio']
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_rollout_groups(batch, tokenizer, reward_info, directory, step, count, max_tokens):
    # Select the first K question UIDs in this rollout, independently of rewards.
    # Keep every sample for each selected question; this is not a success filter.
    uids = [str(x) for x in batch.non_tensor_batch['uid']]
    selected = set(uids) if count == 0 else set(list(dict.fromkeys(uids))[:count])
    indices = [i for i, uid in enumerate(uids) if uid in selected]
    responses = batch.batch['responses'].detach().cpu()
    masks = batch.batch['response_mask'].detach().cpu().bool()
    advantages = batch.batch['advantages'].detach().cpu()
    scores = batch.batch['token_level_scores'].sum(-1).detach().cpu().tolist()
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    sample_counts = {}
    with (directory/f'{step}.jsonl').open('w') as stream:
        for i in indices:
            uid = uids[i]; sample = sample_counts.get(uid, 0); sample_counts[uid] = sample+1
            tokens = responses[i][masks[i]].tolist()
            extra = batch.non_tensor_batch.get('extra_info', [{} for _ in uids])[i]
            reward_model = batch.non_tensor_batch['reward_model'][i]
            row = dict(step=step, group_uid=uid, question_id=extra.get('id'), sample_index=sample,
                input=tokenizer.decode(batch.batch['prompts'][i].detach().cpu(), skip_special_tokens=True),
                output=tokenizer.decode(tokens, skip_special_tokens=True),
                output_with_special_tokens=tokenizer.decode(tokens, skip_special_tokens=False),
                response_token_ids=tokens, response_tokens=len(tokens),
                reached_response_limit=len(tokens) >= max_tokens,
                gts=reward_model.get('ground_truth'), score=scores[i],
                advantage=float(advantages[i][masks[i]].mean()) if masks[i].any() else 0.)
            for key, values in reward_info.items():
                if len(values) == len(uids) and key not in row:
                    row[key] = values[i]
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
