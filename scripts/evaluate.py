import argparse
import yaml
import logging
from datetime import datatime
from scripts.utils import load_data, evaluate_model


def load_config(config_path: str):
    """Loads the config.yaml file."""
    with open(config_path) as f:
        yaml.safe_load(f)


def setup_logging():
    """Setup file and console logger."""
    logging.basicConfig(
        filename=f"logs/asr_evaluation_{datatime.now()}.log",
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Evaluation started...")


def main():
    """Main function to load config and run evaluation for each model."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="ASR-Evaluation", description="Benchmark ASR models on your own dataset."
    )
    parser.add_argument(
        "-c", "--config", required=True, type=argparse.FileType("w"), help="config file"
    )
    args = parser.parse_args()

    config = load_config(args.c)
    models = config["models"]
    data_path = config["data"]["path"]
    audio_dir = config["data"]["audio_dir"]
    batch_size = config["hyperparameters"]["batch_size"]
    beam_size = config["hyperparameters"]["beam_size"]
    sr = config["hyperparameters"]["sr"]

    dataset = load_data(data_path)

    wer_score_csv = "logs/asr_model_evaluation_log.csv"

    for model_name, model_info in models.item():
        evaluate_model(
            model_info=model_name,
            model_info=model_info,
            dataset=dataset,
            audio_dir=audio_dir,
            batch_size=batch_size,
            wer_score_csv=wer_score_csv,
            beam_size=beam_size,
            sr=sr,
        )
    logging.info("All model evaluations completed.")


if __name__ == "__main__":
    main()
