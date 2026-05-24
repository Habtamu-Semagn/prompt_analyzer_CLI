import tiktoken

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

def count_tokens(text, model):

    encoding = ENCODINGS.get(model, "cl100k_base")

    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)

    return len(tokens)

def estimate_cost(token, model):

    cost = PRICING.get(model)

    return (cost * token) / 1_000_000