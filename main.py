# main.py
import argparse
from pipelines.data_pipeline import load_and_prepare_data as data_pipeline
from pipelines.training_pipeline import main as training_pipeline
from pipelines.evaluation_pipeline import main as evaluation_pipeline


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Orchestrator")
    parser.add_argument(
        "--task",
        type=str,
        choices=["data_pipeline", "training_pipeline", "evaluation_pipeline", "app"],
        required=True,
        help="Specify the pipeline task",
    )

    args = parser.parse_args()

    if args.task == "data_pipeline":
        data_pipeline()
    elif args.task == "training_pipeline":
        training_pipeline()
    elif args.task == "evaluation_pipeline":
        evaluation_pipeline()
    elif args.task == "app":
        from app import app  # <-- Import conditionnel
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
