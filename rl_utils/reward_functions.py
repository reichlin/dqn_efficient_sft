import torch
from .utils_simulation import make_padded_batch_from_token_lists
from llm_utils.utils_parsing import is_correct


@torch.inference_mode()
def rollout_answer_reward(
    model,
    tokenizer,
    token_lists_after_action,
    prompt_token_lens,
    golds,
    max_rollout_tokens=64,
    rollout_gamma=0.97,
    eos_token_id=None,
    do_sample=False,
    temperature=1.0,
    top_k=None,
    device="cpu",
):
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    B = len(token_lists_after_action)

    if prompt_token_lens is None:
        prompt_token_lens = [
            len(ids) - 1 for ids in token_lists_after_action
        ]

    if len(prompt_token_lens) != B:
        raise ValueError("prompt_lengths_before_action must have length B.")

    rewards = torch.zeros(B, dtype=torch.float32, device=device)
    hit_steps = [None] * B
    done = [False] * B

    generated_suffixes = []

    # Check only the generated suffix after the original prompt.
    # This suffix includes the auxiliary action token(s), but not the prompt.
    for i, (ids, gold, prompt_len) in enumerate(
        zip(token_lists_after_action, golds, prompt_token_lens)
    ):
        if prompt_len < 0 or prompt_len > len(ids):
            raise ValueError(f"Invalid prompt length for example {i}: {prompt_len}")

        suffix_ids = ids[prompt_len:]
        generated_suffixes.append(suffix_ids)

        suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)

        if is_correct(suffix_text, gold):
            rewards[i] = 1.0
            hit_steps[i] = 0
            done[i] = True

    if all(done):
        return rewards, hit_steps

    batch = make_padded_batch_from_token_lists(
        token_lists=token_lists_after_action,
        pad_token_id=pad_token_id,
        device=device,
        max_context_len=None,
    )

    input_width = batch["input_ids"].shape[1]

    gen_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "max_new_tokens": max_rollout_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "do_sample": do_sample,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    outputs = model.generate(**gen_kwargs)

    # Only tokens produced by the rollout LM, excluding the padded prompt/context.
    rollout_tokens = outputs[:, input_width:]

    for i in range(B):
        if done[i]:
            continue

        rollout_prefix = []

        for token_id in rollout_tokens[i].tolist():
            rollout_prefix.append(token_id)
            step = len(rollout_prefix)

            if eos_token_id is not None and token_id == eos_token_id:
                break

            # Evaluate only: action token(s) + rollout prefix.
            # Do NOT prepend the original prompt.
            candidate_ids = generated_suffixes[i] + rollout_prefix
            candidate_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)

            if is_correct(candidate_text, golds[i]):
                rewards[i] = rollout_gamma ** step
                hit_steps[i] = step
                break

    return rewards, hit_steps