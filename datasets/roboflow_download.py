"""Download the dataset from Roboflow (the data is NOT stored in GitHub).

Usage (Colab):
    import os; os.environ["ROBOFLOW_API_KEY"] = "<your key>"
    !python -m datasets.roboflow_download --workspace WS --project PROJ --version 1 --out /content/dataset
The API key is read from the environment so it never ends up in the repo.
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--format", default="png-mask-semantic")
    p.add_argument("--out", default="dataset")
    args = p.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first.")

    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)
    project.version(args.version).download(args.format, location=args.out)
    print(f"Dataset downloaded to: {args.out}")


if __name__ == "__main__":
    main()
