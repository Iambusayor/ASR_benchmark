import os
import csv
import logging
import time
import librosa
import torch
from transformers import pipeline
import jiwer
from datetime import datetime
from typing import List, Tuple
import nemo.collections.asr as nemo_asr
import datasets
from datasets import load_dataset
from tqdm import tqdm

def load_data(data_path) -> datasets.Dataset:
    """load evaluation dataset from a CSV file."""
    logging.info("Loading dataset...")
    dataset = load_dataset("csv", data_files={"test": data_path})
    logging.info(f"Dataset loaded with {len(dataset["test"])} samples.")
    return dataset["test"]

def write_predictions_to_csv(model_name, predictions, references, csv_file) -> csv:
    """saves the predictions and references to a CSV file"""
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["references", "predictions"])
        for ref, pred in zip(references, predictions):
            writer.writerow([ref, pred])
    logging.info(f"Predictions for {model_name} saved to {csv_file}")

def log_metrics_to_csv(model_name, wer, cer, timestamp, inference_time, max_memory_allocated, wer_score_csv) -> csv:
    """Logs the WER, CER and timestamp to WER_CSV file."""
    csv_header = ["model_name", "WER", "CER", "timestamp", "inference_time", "max_memory_allocated"]

    # CHeck if CSV file exisat, else create new and write header
    file_exists = os.path.isfile(wer_score_csv)

    with open(wer_score_csv, mode="a", newline="", encoding="utf-8") as file:
        writer  = csv.DictWriter(file, fieldnames=csv_header)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "model_name": model_name,
            "WER": wer,
            "CER": cer,
            "timestamp": timestamp,
            inference_time: inference_time,
            max_memory_allocated: max_memory_allocated
        })
    logging.info(f"Model {model_name} WER and CER logged to {wer_score_csv}")

def evaluate_huggingface_model(model_name, model_id, dataset, audio_dir, batch_size, sr=16000) -> Tuple[List[str]]:
    """Evaluates a Hugging Face ASR model."""
    logging.info(f"Evaluating Hugging Face model: {model_name}")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    references = []
    predictions = []
    num_samples = len(dataset)
    
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Processing {model_name}"):
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        audio_files = batch['audio_files']
        transcriptions = batch['transcription']
        
        inputs = []
        for file_path in audio_files:
            full_file_path = os.path.join(audio_dir, file_path)
            waveform, _ = librosa.load(full_file_path, sr=sr)
            inputs.append(waveform)
        
        with torch.no_grad():
            preds = asr_pipeline(inputs)
        predictions.extend([p['text'] for p in preds])
        references.extend(transcriptions)

    return predictions, references

def evaluate_nemo_model(model_name, model_info, dataset, audio_dir, batch_size, beam_size=1):
    """Evaluates a NeMo ASR model."""
    model_id = model_info['model_id']
    model_class_name = model_info.get('model_class', 'EncDecMultiTaskModel')  # Default to ASRModel if not specified
    model_class = getattr(nemo_asr.models, model_class_name)

    logging.info(f"Evaluating NeMo model: {model_name} using class {model_class_name}")
    
    asr_model = model_class.from_pretrained(model_name=model_id)
    if hasattr(asr_model, 'cfg') and hasattr(asr_model.cfg, 'decoding'):
        decode_cfg = asr_model.cfg.decoding
        decode_cfg.beam.beam_size = beam_size
        asr_model.change_decoding_strategy(decode_cfg)

    wav_files = _get_wav_files_from_dataset(dataset, audio_dir)
    predicted_texts = asr_model.transcribe(paths2audio_files=wav_files, batch_size=batch_size)
    
    return predicted_texts, dataset['transcription']

def _get_wav_files_from_dataset(dataset, audio_dir):
    audio_files = dataset['audio_files']
    wav_files = [os.path.join(audio_dir, file) for file in audio_files]
    return wav_files

def evaluate_model(model_name: str, model_info: str, dataset: datasets.Dataset, audio_dir:str, batch_size:int, wer_score_csv: str, beam_size:int=None, sr:int=None):
    """Evaluates a model (Hugging Face or NeMo) and logs WER, CER."""
    framework = model_info['framework']

    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    if framework == 'huggingface':
        predictions, references = evaluate_huggingface_model(model_name, model_info['model_id'], dataset, audio_dir, batch_size, sr)
    elif framework == 'nemo':
        predictions, references = evaluate_nemo_model(model_name, model_info, dataset, audio_dir, batch_size, beam_size)
    else:
        logging.error(f"Unknown framework {framework} for model {model_name}")
        return
    
    # Compute inference time
    inference_tme = time.time() - start_time
    
    # Compute WER and CER
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)

    # Get GPU memory usage if CUDA is available
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device_index) / (1024 ** 3)  # in GB
    else:
        max_memory_allocated = 0  # Set to zero for CPU inference

    logging.info(f"Model {model_name} - WER: {wer}, CER: {cer}, Time taken: {inference_tme:.2f} seconds, GPU Memory Usage: {max_memory_allocated:.2f} GB")
    
    # Save predictions to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("predictions", exist_ok=True)
    predictions_file = f"predictions/{model_name}--{timestamp}_predictions.csv"
    write_predictions_to_csv(model_name=model_name, predictions=predictions, references=references, predictions_file=predictions_file)

    # Log WER and CER to the general CSV file
    log_metrics_to_csv(model_name=model_name, wer=wer, cer=cer, timestamp=timestamp, inference_time=inference_tme, max_memory_allocated=max_memory_allocated, wer_score_csv=wer_score_csv)
