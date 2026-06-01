import re
from decimal import Decimal, InvalidOperation



NUM_RE = re.compile(
    r"[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)



def get_eos_ids(tokenizer):
    ids = []

    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        ids.extend([x for x in eos if x is not None])
    elif eos is not None:
        ids.append(eos)

    for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            if tokenizer.unk_token_id is None or tid != tokenizer.unk_token_id:
                ids.append(tid)

    return sorted(set(ids))


def strip_special_tokens_text(text):
    for tok in [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "</s>",
        "<s>",
    ]:
        text = text.replace(tok, "")
    return text.strip()


def visible_after_thinking(text):
    """
    For Qwen thinking mode, evaluate only the visible final answer after </think>.
    If the model never closes </think>, treat it as no visible answer.
    """
    text = strip_special_tokens_text(text)

    if "</think>" in text:
        return text.split("</think>")[-1].strip()

    if "<think>" in text:
        return ""

    return text.strip()


def extract_number_from_text(text):
    """
    Prefer numbers after FINAL:, otherwise fallback to the last visible number.
    """
    text = visible_after_thinking(text)

    final_markers = list(re.finditer(r"FINAL\s*:", text, flags=re.I))
    if final_markers:
        after_final = text[final_markers[-1].end():]
        nums = NUM_RE.findall(after_final)
        if nums:
            return nums[0]

    nums = NUM_RE.findall(text)
    if nums:
        return nums[-1]

    return None


def to_decimal(x):
    if x is None:
        return None

    x = str(x).strip()
    x = x.replace("$", "").replace(",", "")
    x = x.rstrip(".")

    try:
        return Decimal(x)
    except InvalidOperation:
        return None


def is_correct(pred_text, gold_text):
    pred_num = to_decimal(extract_number_from_text(pred_text))
    gold_num = to_decimal(gold_text)
    return pred_num is not None and gold_num is not None and pred_num == gold_num


def is_parseable(pred_text):
    return to_decimal(extract_number_from_text(pred_text)) is not None


def has_final(text):
    return re.search(r"\bFINAL\s*:", text, flags=re.I) is not None


def clean_step(text):
    """
    Keep only the first step-like chunk.
    This is intentionally simple and hackable.
    """
    text = strip_special_tokens_text(text)

    # If a thinking block somehow appears, score/use only visible content.
    if "</think>" in text or "<think>" in text:
        text = visible_after_thinking(text)

    # Stop at first blank paragraph.
    text = re.split(r"\n\s*\n", text)[0].strip()

    # Remove leading "Step k:".
    text = re.sub(r"^Step\s*\d+\s*:\s*", "", text, flags=re.I).strip()

    # If it starts writing another step, cut there.
    text = re.split(r"\n\s*Step\s*\d+\s*:", text, flags=re.I)[0].strip()

    return text
