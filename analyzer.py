import tiktoken
import re

ENCODINGS = {
"gpt-4o":        "o200k_base",
"gpt-4o-mini":   "o200k_base",
"gpt-4-turbo":   "cl100k_base",
"gpt-3.5-turbo": "cl100k_base",
}

PRICING = {
    # model: cost per 1M input tokens in USD
    "gpt-4o":            2.50,
    "gpt-4o-mini":       0.15,
    "gpt-4-turbo":      10.00,
    "gpt-3.5-turbo":     0.50,
    "claude-3-5-sonnet": 3.00,
    "claude-3-haiku":    0.25,
}

CONTEXT_WINDOWS = {
    "gpt-4o":            128_000,
    "gpt-4o-mini":       128_000,
    "gpt-4-turbo":       128_000,
    "gpt-3.5-turbo":      16_385,
    "claude-3-5-sonnet": 200_000,
    "claude-3-haiku":    200_000,
}

EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "text-embedding-ada-002": 0.100,
}

def count_tokens(text, model):

    encoding = ENCODINGS.get(model, "cl100k_base")

    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)

    return len(tokens)

def estimate_cost(token, model):

    cost = PRICING.get(model, 0.0)

    return (cost * token) / 1_000_000

def has_system_instruction(text: str) -> bool:
    patterns = [r"(?i)^(you are|act as|your role|system:)"]
    has_instruction = any(re.search(p, text) for p in patterns)
    
    return "Yes" if has_instruction else "No"

def count_few_shot_examples(text: str) -> int:
    patterns = [r"(?i)input\s*:[\s\S]+?output\s*:", r"(?i)(q\s*:\s*.+\n.*a\s*:)", r"(?i)example\s*\d+"]

    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text)
        count += len(matches)
    
    return count

def has_output_format(text: str) -> bool:
    patterns = [r"(?i)(json|markdown|bullet|table|list|yaml)"]
    has_format = any(re.search(p, text) for p in patterns)

    return "Yes" if has_format else "No"

def has_delimiters(text: str) -> bool:
    patterns = [
        r"<\w+>[\s\S]*?</\w+>",  # ✅ [\s\S] matches newlines too
        r'"""',
        r"---+",
    ]
    has_delim = any(re.search(p, text) for p in patterns)
    return "Yes" if has_delim else "No"

def context_window_usage(token_count: int, model: str) -> tuple:
    window = CONTEXT_WINDOWS.get(model, 128_000)
    percentage = (token_count / window) * 100
    return percentage, window

def fmt_bar(token_count: int, window: int, width: int = 20) -> str:
    filled = int((token_count / window) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def estimate_embedding_costs(token_count: int) -> dict:
    return {
        model: (price * token_count) / 1_000_000
        for model, price in EMBEDDING_PRICING.items()
    }

def generate_suggestions(text, token_count, model, sys_instr, few_shot, out_format, delimiter):
    suggestions = []

    if token_count > 128000:
        suggestions.append("🚨 Token limit exceeded. Please shorten your prompt.")
    
    if token_count > 10000:
        suggestions.append("⚠️  Prompt is very long, consider using RAG instead.")

    if sys_instr == "No":
        suggestions.append("📌 Add a system instruction (e.g. 'You are an expert...')")

    if out_format == "No":
        suggestions.append("📋 Specify an output format (e.g. JSON, bullet list)")

    if delimiter == "No":
        suggestions.append("🔖 Use delimiters to separate sections (e.g. <context></context>)")

    if few_shot == 0 and token_count < 500:
        suggestions.append("💡 Add 2-3 few-shot examples to improve output quality")

    if not suggestions:
        suggestions.append("✅ Prompt looks good! No issues detected.")

    return suggestions

