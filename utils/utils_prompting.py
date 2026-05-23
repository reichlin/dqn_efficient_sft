

def render_chat(tokenizer, messages, enable_thinking=None):
    """
    Robust wrapper around apply_chat_template.

    For original hybrid Qwen3 checkpoints, enable_thinking should be True/False.
    For Thinking-2507/Instruct-2507, pass None.
    """
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )

    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    else:
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some tokenizers/templates do not accept enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt_continuation(questions, tokenizer, enable_thinking=None):
    prompts = []

    for question in questions:
        user_text = f"""{question}""".strip()

        messages = [
            {
                "role": "system",
                "content": "You are a careful math solver. Follow the user's requested format exactly.",
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        prompts.append(render_chat(tokenizer, messages, enable_thinking=enable_thinking))

    return prompts


def build_prompt_only_answer(questions, tokenizer, enable_thinking=None):
    prompts = []

    for question in questions:
        user_text = f"""
        Solve this GSM8K math problem.

        Problem:
        {question}

        Rules:
        - Do not output any intermediate step.
        - Give only the final answer in the exact format: FINAL: <answer>
        - The answer should be a single number.
        """.strip()

        messages = [
            {
                "role": "system",
                "content": "You are a careful math solver. Follow the user's requested format exactly.",
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        prompts.append(render_chat(tokenizer, messages, enable_thinking=enable_thinking))

    return prompts


def build_prompt_oneshot(questions, tokenizer, enable_thinking=None):
    prompts = []

    for question in questions:
        user_text = f"""
        Solve this GSM8K math problem.
        
        Problem:
        {question}
        
        Rules:
        - Give the final answer in the exact format: FINAL: <answer>
        - The answer should be a single number.
        """.strip()

        messages = [
            {
                "role": "system",
                "content": "You are a careful math solver. Follow the user's requested format exactly.",
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        prompts.append(render_chat(tokenizer, messages, enable_thinking=enable_thinking))

    return prompts


def build_prompt_reasoning(questions, thoughts_for_questions, tokenizer, enable_thinking=None):
    prompts = []

    for question, thought_list in zip(questions, thoughts_for_questions):
        reasoning_so_far = "".join(
            f"Step {i}: {t}\n" for i, t in enumerate(thought_list, start=1)
        )

        user_text = f"""
        Solve this GSM8K math problem.
        
        Problem:
        {question}
        
        We are studying the model's intermediate reasoning trajectory.
        Write exactly ONE next reasoning step.
        
        Rules:
        - If more reasoning is needed, write only the next short step.
        - If you know the answer, write: FINAL: <answer>
        - Do not write multiple steps at once.
        - The final answer should be a single number.
        
        Reasoning so far:
        {reasoning_so_far}
        Next:
        """.strip()

        messages = [
            {
                "role": "system",
                "content": "You are a careful math reasoner. Follow the user's requested format exactly.",
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        prompts.append(render_chat(tokenizer, messages, enable_thinking=enable_thinking))

    return prompts