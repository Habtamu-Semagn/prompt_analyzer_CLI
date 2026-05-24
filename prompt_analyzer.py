import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    print(f"Prompt: {args.prompt} | Model: {args.model}")

if __name__ == "__main__":
    main()