import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from supervised_learning import clean_seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Small inference demo for the TCR binding case study. "
            "Pass a single TCR-epitope pair or a CSV file with tcr/epitope columns."
        )
    )
    parser.add_argument(
        "--model",
        default="artifacts/tcr_binding_baseline.pkl",
        help="Path to a trained model created by train_baseline.py.",
    )
    parser.add_argument("--tcr", default=None, help="Single TCR sequence to score.")
    parser.add_argument("--epitope", default=None, help="Single epitope sequence to score.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional CSV file with columns 'tcr' and 'epitope'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probabilities into labels.",
    )
    return parser.parse_args()


def build_examples(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if not {"tcr", "epitope"}.issubset(df.columns):
            raise SystemExit("Input CSV must contain 'tcr' and 'epitope' columns.")
        examples = df[["tcr", "epitope"]].copy()
    elif args.tcr and args.epitope:
        examples = pd.DataFrame([{"tcr": args.tcr, "epitope": args.epitope}])
    else:
        raise SystemExit("Provide either --tcr with --epitope, or --input-csv.")

    examples["tcr"] = examples["tcr"].map(clean_seq)
    examples["epitope"] = examples["epitope"].map(clean_seq)
    invalid = examples["tcr"].isna() | examples["epitope"].isna()
    if invalid.any():
        raise SystemExit("One or more sequences were empty or invalid after cleaning.")
    return examples.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    examples = build_examples(args)

    probs = pipeline.predict_proba(examples)[:, 1]
    labels = (probs >= args.threshold).astype(int)

    rows = []
    for idx, row in examples.iterrows():
        rows.append(
            {
                "index": int(idx),
                "tcr": row["tcr"],
                "epitope": row["epitope"],
                "binding_probability": round(float(probs[idx]), 4),
                "predicted_label": int(labels[idx]),
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
