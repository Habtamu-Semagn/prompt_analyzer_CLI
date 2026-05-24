import argparse
import sys
from analyzer import count_tokens, estimate_cost,count_few_shot_examples, has_system_instruction, has_delimiters, has_output_format, generate_suggestions
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
    
    print("═" * 40)
    print(f"{'PROMPT ANALYSIS REPORT':^40}")
    print("═" * 40)

    print("\n📊 Token & Cost")
    print("─" * 40)

    print("  Token Count" + " "*5 + f": {token_count}")
    print("  Estimated Cost" + " "*2 + f": {estimated_cost:.6f}$")
    print("  Model" + " "*11 + f": {args.model}")

    print("\n🔬 STRUCTURE ANALYSIS")
    print("─" * 40)

    print("  System instruction" + " "*2 + f": {has_system_instruction(text)}")
    if(count_few_shot_examples(text) > 0):
        print("  Few-shot examples" + " "*3 + f": Yes (count: {count_few_shot_examples(text)})")
    else:
        print("  Few-shot examples" + " "*3 + ": No (count: 0)")
    print("  Output format" + " "*7 + f": {has_output_format(text)}")
    print("  Delimiters" + " "*10 + f": {has_delimiters(text)}")

    
    print("💬 Suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

if __name__ == "__main__":
    main()