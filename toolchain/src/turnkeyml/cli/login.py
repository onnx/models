import os
import argparse


def login(key: str):

    toke_path = os.path.expanduser("~/.huggingface/token")
    hf_path = os.path.expanduser("~/.huggingface")
    if not os.path.exists(hf_path):
        os.makedirs(hf_path)

    if not os.path.exists(toke_path):
        with open(toke_path, "w", encoding="utf-8") as f:
            f.write(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update huggingface credentials")
    parser.add_argument("--key", help="API key", required=True)
    args = vars(parser.parse_args())
    login(args["key"])
