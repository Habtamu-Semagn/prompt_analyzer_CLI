import argparse
import sys
from analyzer import count_tokens
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str)
    group.add_argument("--file", type=str)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    if args.prompt:
        text = args.prompt
    else:
        try:
            with open(args.file, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: file '{args.file}' not found.")
            sys.exit()

    token_count = count_tokens(text, args.model)
    print(f"Token count: {token_count}")
    print(f"Text: {text} | Model: {args.model}")

if __name__ == "__main__":
    main()