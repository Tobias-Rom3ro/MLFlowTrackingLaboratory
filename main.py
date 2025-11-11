import argparse
from src.qsar_biodeg.pipeline import main as pipeline_main


def main():
    parser = argparse.ArgumentParser(description="QSAR Biodegradation MLflow Pipeline")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline including nested experiments"
    )
    args = parser.parse_args()

    pipeline_main(full=args.full)


if __name__ == "__main__":
    main()