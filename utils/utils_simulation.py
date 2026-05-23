import torch


def model_input_device(model):
    return next(model.parameters()).device


def make_padded_batch_from_token_lists(token_lists, pad_token_id, device, max_context_len=None):
    if max_context_len is not None:
        token_lists = [ids[-max_context_len:] for ids in token_lists]

    max_len = max(len(ids) for ids in token_lists)
    input_ids = torch.full(
        (len(token_lists), max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(token_lists), max_len),
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(token_lists):
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, -len(ids_t):] = ids_t
        attention_mask[i, -len(ids_t):] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


@torch.inference_mode()
def get_state_and_action_set(model, tokenizer, token_lists, top_k, max_context_len=None):
    dev = model_input_device(model)

    batch = make_padded_batch_from_token_lists(
        token_lists=token_lists,
        pad_token_id=tokenizer.pad_token_id,
        device=dev,
        max_context_len=max_context_len,
    )

    outputs = model(
        **batch,
        output_hidden_states=True,
        use_cache=False,   # no benefit unless you actually reuse past_key_values
        return_dict=True,
    )

    attention_mask = batch["attention_mask"]
    seq_len = attention_mask.shape[1]
    last_nonpad_idx = seq_len - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    batch_idx = torch.arange(attention_mask.shape[0], device=dev)

    logits = outputs.logits[batch_idx, last_nonpad_idx, :]
    states = outputs.hidden_states[-1][batch_idx, last_nonpad_idx, :].float()

    k = min(top_k, logits.shape[-1])
    action_sets = torch.topk(logits, k=k, dim=-1).indices.long()

    return states, action_sets