import torch
from .utils_simulation import make_padded_batch_from_token_lists, get_state_and_action_set
from .utils_parsing import is_correct, get_eos_ids
from .utils_prompting import build_prompt_continuation


@torch.inference_mode()
def greedy_probe_for_answer(
        model,
        tokenizer,
        token_list,
        prompt_token_len,
        gold,
        max_probe_tokens=64,
        eos_ids=None,
        device='cpu',
):
    """
    From the current context, greedily generate up to max_probe_tokens.
    Return the earliest number of probe tokens needed to see the answer.

    Returns:
        hit_step: int or None
            0 means answer is already present in the generated suffix.
            1 means answer appears after one greedy probe token.
            None means answer did not appear.
        probe_text: str
            Full generated suffix after prompt.
    """

    if eos_ids is None:
        eos_ids = get_eos_ids(tokenizer)
    eos_id_set = set(int(x) for x in eos_ids)

    # Check only the generated suffix, not the original prompt.
    gen_suffix_text = tokenizer.decode(
        token_list[prompt_token_len:],
        skip_special_tokens=True,
    )

    if is_correct(gen_suffix_text, gold):
        return 0, gen_suffix_text

    batch = make_padded_batch_from_token_lists(
        token_lists=[token_list],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        max_context_len=None,
    )

    input_len = batch["input_ids"].shape[1]

    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=max_probe_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
    )

    new_tokens = out[0, input_len:].tolist()

    # Scan prefixes of the rollout to find earliest answer appearance.
    for step in range(1, len(new_tokens) + 1):
        probe_prefix = new_tokens[:step]

        candidate_ids = token_list + probe_prefix
        candidate_suffix_text = tokenizer.decode(
            candidate_ids[prompt_token_len:],
            skip_special_tokens=True,
        )

        if is_correct(candidate_suffix_text, gold):
            return step, candidate_suffix_text

        if any(tok in eos_id_set for tok in probe_prefix):
            break

    final_suffix_text = tokenizer.decode(
        token_list[prompt_token_len:] + new_tokens,
        skip_special_tokens=True,
    )

    return None, final_suffix_text


@torch.inference_mode()
def eval_one_q_guided_problem(
        model,
        tokenizer,
        q_net,
        question,
        gold,
        horizon=480,
        top_k=10,
        max_probe_tokens=64,
        enable_thinking=False,
        max_context_len=None,
        device='cpu'
):
    """
    Generate with q_net choosing argmax over the frozen LM top-k tokens.

    At each Q-guided token, probe whether the frozen LM can now complete
    to the correct answer within max_probe_tokens.

    Returns a dict with token-efficiency metrics.
    """
    q_net.eval()

    eos_ids = get_eos_ids(tokenizer)
    eos_id_set = set(int(x) for x in eos_ids)

    prompt = build_prompt_continuation(
        [question],
        tokenizer,
        enable_thinking=enable_thinking,
    )[0]

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        padding=False,
    )["input_ids"]

    token_list = prompt_ids[:]
    prompt_token_len = len(prompt_ids)

    # Optional: check whether the model can already answer without Q tokens.
    probe_hit, probe_text = greedy_probe_for_answer(
        model=model,
        tokenizer=tokenizer,
        token_list=token_list,
        prompt_token_len=prompt_token_len,
        gold=gold,
        max_probe_tokens=max_probe_tokens,
        eos_ids=eos_ids,
        device=device
    )

    if probe_hit is not None:
        return {
            "solved": True,
            "q_tokens": 0,
            "probe_tokens": probe_hit,
            "total_new_tokens": probe_hit,
            "answer_text": probe_text,
            "stopped_by": "probe_success_before_q",
        }

    for t in range(1, horizon + 1):
        # Current state and viable next-token set from frozen LM.
        s_t, C_a_t = get_state_and_action_set(
            model=model,
            tokenizer=tokenizer,
            token_lists=[token_list],
            top_k=top_k,
            max_context_len=max_context_len,
        )

        # Q-values over candidate next tokens.
        q_values = q_net(
            s_t.to(device),
            C_a_t.to(device),
            action_mask=torch.ones_like(C_a_t, dtype=torch.bool, device=device),
        )  # [1, K]

        best_col = q_values.argmax(dim=1).item()
        chosen_token = int(C_a_t[0, best_col].item())

        token_list.append(chosen_token)

        # If Q chose EOS, stop.
        if chosen_token in eos_id_set:
            final_text = tokenizer.decode(
                token_list[prompt_token_len:],
                skip_special_tokens=True,
            )

            return {
                "solved": is_correct(final_text, gold),
                "q_tokens": t,
                "probe_tokens": 0 if is_correct(final_text, gold) else None,
                "total_new_tokens": t if is_correct(final_text, gold) else None,
                "answer_text": final_text,
                "stopped_by": "eos",
            }

        # Probe from this new context.
        probe_hit, probe_text = greedy_probe_for_answer(
            model=model,
            tokenizer=tokenizer,
            token_list=token_list,
            prompt_token_len=prompt_token_len,
            gold=gold,
            max_probe_tokens=max_probe_tokens,
            eos_ids=eos_ids,
            device=device
        )

        if probe_hit is not None:
            return {
                "solved": True,
                "q_tokens": t,
                "probe_tokens": probe_hit,
                "total_new_tokens": t + probe_hit,
                "answer_text": probe_text,
                "stopped_by": "probe_success",
            }

    final_text = tokenizer.decode(
        token_list[prompt_token_len:],
        skip_special_tokens=True,
    )

    return {
        "solved": False,
        "q_tokens": horizon,
        "probe_tokens": None,
        "total_new_tokens": None,
        "answer_text": final_text,
        "stopped_by": "horizon",
    }


def evaluate_q_guidance(
    model,
    tokenizer,
    q_net,
    test_dataset,
    horizon=480,
    top_k=10,
    max_probe_tokens=64,
    max_examples=None,
    enable_thinking=False,
        device='cpu'
):
    results = []

    n = len(test_dataset) if max_examples is None else min(max_examples, len(test_dataset))

    for i in range(n):
        ex = test_dataset[i]

        question = ex["question"]
        gold = ex["answer"].split("####")[-1].strip()

        result = eval_one_q_guided_problem(
            model=model,
            tokenizer=tokenizer,
            q_net=q_net,
            question=question,
            gold=gold,
            horizon=horizon,
            top_k=top_k,
            max_probe_tokens=max_probe_tokens,
            enable_thinking=enable_thinking,
            device=device
        )

        result["idx"] = i
        result["gold"] = gold
        results.append(result)

        if (i + 1) % 25 == 0:
            acc = sum(r["solved"] for r in results) / len(results)
            solved_tokens = [
                r["total_new_tokens"]
                for r in results
                if r["total_new_tokens"] is not None
            ]

            avg_tokens = (
                sum(solved_tokens) / len(solved_tokens)
                if len(solved_tokens) > 0
                else None
            )

            print(
                f"eval {i + 1}/{n} | "
                f"acc={acc:.3f} | "
                f"avg_total_new_tokens={avg_tokens}"
            )

    return results


def accuracy_vs_token_budget(results, budgets=None):
    """
    For each total-new-token budget, compute fraction of examples solved
    within that budget.
    """
    if budgets is None:
        solved_tokens = sorted(
            r["total_new_tokens"]
            for r in results
            if r["total_new_tokens"] is not None
        )

        if len(solved_tokens) == 0:
            return []

        budgets = sorted(set(solved_tokens))

    curve = []

    n = len(results)
    for budget in budgets:
        solved_within_budget = sum(
            r["solved"]
            and r["total_new_tokens"] is not None
            and r["total_new_tokens"] <= budget
            for r in results
        )

        curve.append(
            {
                "token_budget": budget,
                "accuracy": solved_within_budget / n,
                "num_solved": solved_within_budget,
                "num_total": n,
            }
        )

    return curve

