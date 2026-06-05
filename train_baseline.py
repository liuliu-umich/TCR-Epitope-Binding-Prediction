import argparse
from pathlib import Path

from supervised_learning import defaults, train_vdjdb, validate_trait


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the recommended baseline model for the TCR binding case study. "
            "This entrypoint focuses on a lightweight, reproducible benchmark run."
        )
    )
    parser.add_argument(
        "--vdjdb-path",
        default="Datasets/VDJdb_clean.xlsx",
        help="Path to the curated VDJdb dataset used for baseline training.",
    )
    parser.add_argument(
        "--vdjdb-sheet",
        default=None,
        help="Optional Excel sheet name for the VDJdb file.",
    )
    parser.add_argument(
        "--model-out",
        default="artifacts/tcr_binding_baseline.pkl",
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--maxlen-tcr",
        type=int,
        default=defaults["maxlen_tcr"],
        help="Maximum TCR sequence length for one-hot encoding.",
    )
    parser.add_argument(
        "--maxlen-epi",
        type=int,
        default=defaults["maxlen_epi"],
        help="Maximum epitope sequence length for one-hot encoding.",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=defaults["neg_ratio"],
        help="Negative-to-positive sampling ratio used during training.",
    )
    parser.add_argument(
        "--validate-trait",
        action="store_true",
        help="Run the optional external validation pass on TRAITdb after training.",
    )
    parser.add_argument(
        "--trait-path",
        default="Datasets/TRAITdb_clean.xlsx",
        help="Path to the TRAITdb validation dataset.",
    )
    parser.add_argument(
        "--trait-sheet",
        default=None,
        help="Optional Excel sheet name for the TRAITdb file.",
    )
    return parser.parse_args()


def ensure_parent(path_str: str) -> None:
    path = Path(path_str)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_parent(args.model_out)

    train_vdjdb(
        vdj_path=args.vdjdb_path,
        sheet=args.vdjdb_sheet,
        tcr_col=defaults["vdj_tcr"],
        epi_col=defaults["vdj_epi"],
        score_col=defaults["vdj_score"],
        pos_score=defaults["pos_score"],
        neg_scores=defaults["neg_scores"],
        neg_ratio=args.neg_ratio,
        maxlen_tcr=args.maxlen_tcr,
        maxlen_epi=args.maxlen_epi,
        model_out=args.model_out,
    )

    if args.validate_trait:
        validate_trait(
            model_path=args.model_out,
            trait_path=args.trait_path,
            sheet=args.trait_sheet,
            tcr_col=defaults["trait_tcr"],
            epi_col=defaults["trait_epi"],
            bind_col=defaults["trait_bind"],
        )


if __name__ == "__main__":
    main()
