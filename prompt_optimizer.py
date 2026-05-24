import argparse
import sys
from analyzer import count_tokens, estimate_cost,count_few_shot_examples, has_system_instruction, has_delimiters, has_output_format, generate_suggestions, context_window_usage, fmt_bar

def print_report(text, model, token_count, estimated_cost, suggestions):
    few_shot = count_few_shot_examples(text)
    
    print("═" * 40)
    print(f"{'PROMPT ANALYSIS REPORT':^40}")
    print("═" * 40)

    print("\n📊 TOKEN & COST")
    print("─" * 40)
    print(f"  Token Count     : {token_count}")
    print(f"  Estimated Cost  : ${estimated_cost:.6f}")
    print(f"  Model           : {model}")
    percentage, window = context_window_usage(token_count, model)
    bar = fmt_bar(token_count, window)

    print(f"  Context Window  : {token_count:,} / {window:,} tokens")
    print(f"  Context Usage   : {bar} {percentage:.2f}%")

    print("\n🔬 STRUCTURE ANALYSIS")
    print("─" * 40)
    print(f"  System instruction  : {has_system_instruction(text)}")
    print(f"  Few-shot examples   : {'Yes' if few_shot > 0 else 'No'} (count: {few_shot})")
    print(f"  Output format       : {has_output_format(text)}")
    print(f"  Delimiters          : {has_delimiters(text)}")

    print("\n💬 SUGGESTIONS")
    print("─" * 40)
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")
    print("\n" + "═" * 40)
    
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
    estimated_cost = estimate_cost(token_count, args.model)
    suggestions = generate_suggestions(
        text, 
        token_count, 
        args.model, 
        has_system_instruction(text), 
        count_few_shot_examples(text), 
        has_output_format(text), 
        has_delimiters(text)
        )

    print_report(text, args.model, token_count, estimated_cost, suggestions)

if __name__ == "__main__":
    main()