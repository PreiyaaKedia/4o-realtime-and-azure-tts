import os
import time
import json
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import tempfile
import wave
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from scipy.io.wavfile import write  
import azure.cognitiveservices.speech as speechsdk
from jiwer import wer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_speech_recognizer(subscription_key: str, region: str, speech_language: str = "ta-IN", audio_file_path = None) -> speechsdk.SpeechRecognizer:
    """
    Create an Azure Speech Recognizer for Tamil language.
    
    Args:
        subscription_key: Azure Speech Service subscription key
        region: Azure region
        speech_language: Speech language code (default is Tamil - ta-IN)
        
    Returns:
        A configured SpeechRecognizer object
    """
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_recognition_language = speech_language
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path) if audio_file_path else None
    # Create a speech recognizer without audio config (we'll set this per audio file)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        audio_config=audio_config  # We'll set this per audio file
    )
    
    return speech_recognizer

def save_audio_to_wav(audio_array: np.ndarray, file_path: str, sample_rate: int = 16000) -> None:
    """
    Save audio data as a WAV file.
    
    Args:
        audio_array: Audio data as numpy array
        file_path: Path to save the WAV file
        sample_rate: Sample rate of the audio (default: 16000 Hz)
    """
    # Ensure the audio data is in the correct format (float32 between -1 and 1)
    if audio_array.dtype != np.float32:
        # Convert to float32 if not already
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        else:
            audio_array = audio_array.astype(np.float32)
    
    # Normalize if needed
    if np.max(np.abs(audio_array)) > 1.0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    
    # Convert to int16 for WAV file
    audio_data = (audio_array * 32767).astype(np.int16)
    
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def speech_to_text(subscription_key : str, region : str,  audio_file_path: str) -> str:
    """
    Transcribe speech from an audio file using Azure Speech-to-Text with continuous recognition.
    
    Args:
        speech_recognizer: Configured speech recognizer
        audio_file_path: Path to the audio file
        
    Returns:
        Transcription of the audio file
    """
    # Create an audio configuration for the specific file
    # speech_config = speech_recognizer._config
    
    # # Create an audio configuration for the specific file
    # audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    
    # # Create a new speech recognizer with the audio config
    # speech_recognizer = speechsdk.SpeechRecognizer(
    #     speech_config=speech_config, 
    #     audio_config=audio_config
    # )
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, value = "AtStart")
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["te-IN", "ml-IN", "ta-IN"])
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config, auto_detect_source_language_config=auto_detect_source_language_config)
    
    
    # speech_recognizer = create_speech_recognizer(subscription_key, region, audio_file_path=audio_file_path)
    
    # Store the transcribed text
    all_results = []
    done = False
    
    # Set up the event handlers for continuous recognition
    def recognized_cb(evt):
        if evt.result.text:
            all_results.append(evt.result.text)
    
    def stop_cb(evt):
        nonlocal done
        done = True
    
    def canceled_cb(evt):
        nonlocal done
        speech_recognizer.stop_continuous_recognition()
        cancellation = speechsdk.CancellationDetails(evt.result)
        logger.error(f"Speech recognition canceled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            logger.error(f"Error details: {cancellation.error_details}")
        done = True
    
    # Connect callbacks to the events
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(canceled_cb)
    
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
    speech_recognizer.stop_continuous_recognition()
    
    # Disconnect event handlers
    speech_recognizer.recognized.disconnect_all()
    speech_recognizer.session_stopped.disconnect_all()
    speech_recognizer.canceled.disconnect_all()
    
    # Return the combined transcription
    return " ".join(all_results)

def load_dataset_sample(
    dataset_name: str, 
    split: str = "validation", 
    sample_count: Optional[int] = None, 
    start_index: int = 0, 
    use_streaming: bool = False
):
    """
    Load a sample of the dataset instead of the entire dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use (train, validation, test)
        sample_count: Number of samples to load (None for all)
        start_index: Index to start loading samples from
        use_streaming: Whether to use streaming mode to avoid downloading the entire dataset
        
    Returns:
        A dataset object with the requested samples
    """
    logger.info(f"Loading {dataset_name} dataset ({split} split)...")
    
    try:
        if use_streaming:
            # Load the dataset in streaming mode to avoid downloading everything
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            if sample_count:
                # For streaming datasets, we need to manually take samples
                dataset = dataset.skip(start_index).take(sample_count)
                logger.info(f"Streaming {sample_count} samples starting from index {start_index}")
            else:
                # If no sample_count is specified but we still want to stream from a specific index
                if start_index > 0:
                    dataset = dataset.skip(start_index)
                    logger.info(f"Streaming all samples starting from index {start_index}")
                else:
                    logger.info("Streaming all samples")
        else:
            # Load the dataset in regular mode (downloads all data)
            dataset = load_dataset(dataset_name, split=split)
            
            if sample_count:
                # Select a range of samples
                end_index = min(start_index + sample_count, len(dataset))
                dataset = dataset.select(range(start_index, end_index))
                logger.info(f"Selected {end_index - start_index} samples from index {start_index}")
            else:
                if start_index > 0:
                    # If no sample_count is specified but we still want to start from a specific index
                    dataset = dataset.select(range(start_index, len(dataset)))
                    logger.info(f"Selected all samples starting from index {start_index}")
                else:
                    logger.info(f"Using full dataset with {len(dataset)} samples")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return None

def calculate_wer_for_dataset(
    dataset_name: str, 
    split: str = "validation", 
    sample_count: Optional[int] = None, 
    start_index: int = 0, 
    use_streaming: bool = False
) -> Dict:
    """
    Calculate Word Error Rate for the IndicTTS_Tamil dataset using Azure Speech-to-Text.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use (train, validation, test)
        sample_count: Number of samples to process (None for all)
        start_index: Index to start loading samples from
        use_streaming: Whether to use streaming mode to avoid downloading the entire dataset
        
    Returns:
        Dictionary with WER results
    """
    # Check if Azure Speech credentials are available
    subscription_key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    
    if not subscription_key or not region:
        logger.error("Azure Speech Service credentials not found. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables.")
        return {"error": "Azure Speech Service credentials not found"}
    
    # Create speech recognizer
    speech_recognizer = create_speech_recognizer(subscription_key, region)
    
    # Load the dataset sample
    dataset = load_dataset_sample(dataset_name, split, sample_count, start_index, use_streaming)
    # dataset = load_dataset_sample("SPRINGLab/IndicTTS_Tamil", split = "train", sample_count = 2, start_index = 10, use_streaming=True)
    # sample_count = 2
    # iter(next(dataset))
    if dataset is None:
        return {"error": "Failed to load dataset"}
    
    # Process each sample
    results = []
    reference_texts = []
    hypothesis_texts = []
    
    # Count samples for progress reporting
    sample_counter = 0
    max_samples = sample_count if sample_count is not None else float('inf')
    
    for sample in dataset:
        sample_counter += 1
        logger.info(f"Processing sample {sample_counter}")
        
        if sample_counter > max_samples:
            break
        
        try:
            # Get the audio data and text - adapt for IndicTTS_Tamil dataset structure
            if 'audio' in sample and isinstance(sample['audio'], dict) and 'array' in sample['audio']:
                audio_data = sample['audio']['array']
                sample_rate = sample['audio'].get('sampling_rate', 16000)
            else:
                # For datasets where 'audio' is the actual numpy array
                audio_data = sample.get('audio')
                sample_rate = 16000  # Default sample rate if not specified
            
            reference_text = sample.get('text', '').replace(",", "").strip()
            
            if audio_data is None or not reference_text:
                logger.warning(f"Sample {sample_counter} is missing audio data or text")
                continue
            
            # Save audio to a temporary file
            # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            #     temp_path = temp_file.name
            
            temp_file = "abc.wav"

            audio_int16 = np.int16(audio_data * 32767)
            write(temp_file, sample_rate, audio_int16)  
            # Transcribe the audio
            transcribed_text = speech_to_text(subscription_key, region, temp_file)
            
            # Skip if transcription failed
            if not transcribed_text:
                logger.warning(f"Failed to transcribe sample {sample_counter}")
                continue
            
            # Store results
            reference_texts.append(reference_text)
            hypothesis_texts.append(transcribed_text.strip())
            
            # Calculate WER for this sample
            sample_wer = wer(reference_text, transcribed_text.strip())
            
            results.append({
                "sample_id": sample_counter - 1,
                "reference": reference_text,
                "hypothesis": transcribed_text.strip(),
                "wer": sample_wer
            })
            
            # Log progress
            if sample_counter % 10 == 0:
                logger.info(f"Processed {sample_counter} samples")
                
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_counter}: {e}")
            logger.error(traceback.format_exc())
    
    # Calculate overall WER
    if reference_texts and hypothesis_texts:
        overall_wer = wer(reference_texts, hypothesis_texts)
    else:
        overall_wer = 1.0  # Worst case if no transcriptions were successful
    
    # Prepare final results
    wer_results = {
        "dataset": dataset_name,
        "split": split,
        "samples_processed": len(results),
        "overall_wer": overall_wer,
        "sample_results": results
    }
    
    # Save results to a JSON file
    output_file = f"wer_results_{dataset_name.replace('/', '_')}_{split}_{len(results)}_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wer_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"WER calculation completed. Overall WER: {overall_wer:.4f}")
    logger.info(f"Results saved to {output_file}")
    
    return wer_results

def main():
    """Main function to calculate WER for the IndicTTS_Tamil dataset"""
    dataset_name = "SPRINGLab/IndicTTS_Telugu"
    split = "train"  # or "test", "train" depending on what you want to evaluate
    sample_count = 20     # Set to None to process all samples, or a specific number for testing
    start_index = 0       # Start from the beginning of the dataset
    use_streaming = True  # Use streaming mode to avoid downloading the entire dataset
    
    logger.info(f"Starting WER calculation for {dataset_name}")
    wer_results = calculate_wer_for_dataset(
        dataset_name=dataset_name,
        split=split,
        sample_count=sample_count,
        start_index=start_index,
        use_streaming=use_streaming
    )
    
    if "error" in wer_results:
        logger.error(f"Failed to calculate WER: {wer_results['error']}")
    else:
        logger.info(f"WER calculation completed successfully.")
        logger.info(f"Overall WER: {wer_results['overall_wer']:.4f}")
        logger.info(f"Processed {wer_results['samples_processed']} samples.")

if __name__ == "__main__":
    main()
