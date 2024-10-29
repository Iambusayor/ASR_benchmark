import argparse
import yaml
import logging
from datetime import datetime
from utils import load_data, evaluate_model


def load_config(config_path: str):
    """Loads the config.yaml file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging():
    """Setup file and console logger."""

    # Create a custom logger
    logger = logging.getLogger()

    # Set the global log level
    logger.setLevel(logging.INFO)

    # Create a file handler and set level to INFO
    file_handler = logging.FileHandler(
        f"asr_evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    file_handler.setLevel(logging.INFO)

    # Create a console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter for both handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial message
    logging.info("Evaluation started...")


def main():
    """Main function to load config and run evaluation for each model."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="ASR-Evaluation", description="Benchmark ASR models on your own dataset."
    )
    parser.add_argument("-c", "--config", required=True, help="A YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    models = config["models"]
    data_path = config["data"]["path"]
    audio_dir = config["data"]["audio_dir"]
    batch_size = config["hyperparameters"]["batch_size"]
    beam_size = config["hyperparameters"]["beam_size"]
    sr = config["hyperparameters"]["sr"]

    dataset = load_data(data_path)

    wer_score_csv = "asr_model_evaluation_log.csv"

    for model_name, model_info in models.items():
        evaluate_model(
            model_name=model_name,
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
