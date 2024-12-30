
# Audio Captioning with Whisper: A Comprehensive Guide to Fine-tuning

This repository provides a detailed guide on how to fine-tune OpenAI's Whisper model for audio captioning using the Hugging Face Transformers library. This README caters to both beginners and experienced users, covering single GPU training, multi-GPU training on a single node (pod), data preparation, and inference.

## 1. Installation

Before starting, ensure you have a working Python environment (preferably Python 3.8 or higher). Create a virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs essential libraries including:

-   **PyTorch:** Deep learning framework.
-   **Transformers:** Hugging Face library for using and fine-tuning pre-trained models.
-   **Datasets:** Hugging Face library for easily accessing and processing datasets.
-   **Webdataset:** For efficient data loading, especially for large datasets.
-   **Accelerate:** Hugging Face library for simplifying multi-GPU and distributed training.
-   **einops:** For convenient tensor manipulation.
-   **hydra-core:** For managing configuration files.
-   **tensorboardX:** For visualizing training progress.
-   **librosa:** For audio resampling.

## 2. Data Preparation

This project supports datasets containing audio in various formats, including MP3, WAV, and FLAC. The provided `WhisperAudioCaptionProcessor` class can handle these formats seamlessly, thanks to its use of `librosa` for audio loading and resampling.

**Crucially, your dataset must have a text or JSON file associated with each audio file.** These text files contain the target captions that the model will learn to predict.

### 2.1 Dataset Structure

The script assumes your data is in a format accessible by the Hugging Face `datasets` library. You can use datasets directly from the Hugging Face Hub, or you can use local datasets.

**Example Dataset Structure (Hugging Face Hub):**
The `krishnakalyan3/emo_webds_2` dataset has the following structure, which can be seen when you print the dataset:
```
Dataset emo_webds_2({
    features: ['flac', 'json'],
    num_rows: 39592
})
```
The `json` field is another dictionary that contains the `text` field, which is what we will use as target. The `flac` field is a dictionary that contains the audio data. The `path` subfield contains the path to the audio file, and the `array` subfield is a 1-D array that contains the audio data already resampled to 16kHz.

**Example Dataset Structure (Local):**

```
├── your_dataset_folder
│   ├── train
│   │   ├── audio_001.mp3
│   │   ├── audio_001.json
│   │   ├── audio_002.wav
│   │   ├── audio_002.txt
│   │   └── ...
│   └── validation
│       ├── audio_101.flac
│       ├── audio_101.json
│       ├── audio_102.mp3
│       ├── audio_102.txt
│       └── ...
```

Each audio file should have a corresponding text (.txt) or JSON (.json) file that contains the caption. The naming convention should be consistent (e.g., `audio_001.mp3` with `audio_001.json` or `audio_001.txt`).

### 2.2. Handling Different Audio Formats (MP3, WAV, FLAC) and Connecting Audio to Text

The current code uses `librosa.resample` within the `WhisperAudioCaptionProcessor` to handle audio files. `librosa` can automatically load and decode various audio formats, including MP3 and WAV.

**Key Code Snippet (data.py):**

```python
import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer

class WhisperAudioCaptionProcessor:
    def __init__(self, model_name: str):
        # use translate task as captioning task now
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language="en", task="translate")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    def __call__(self, item):
        # For HuggingFace datasets:
        caption = item["json"]["text"]
        audio = item["flac"]["array"]
        sr = item["flac"]["sampling_rate"]

        # For local datasets:
        # caption = open(item["caption_path"], "r").read() # Assuming your dataset provides a "caption_path" field
        # audio_path = item["audio_path"] # Assuming your dataset provides an "audio_path" field
        # audio, sr = librosa.load(audio_path, sr=None)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        input_features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = self.tokenizer(caption).input_ids
        return {"input_features": input_features, "labels": input_ids}
```

**Explanation:**

-   The `__call__` method is where each audio file is linked to its caption.
-   In the original example, `item["json"]["text"]` fetches the caption text, and `item["flac"]["array"]` fetches the audio data.
-   You might need to modify these lines to `caption = open(item["caption_path"], "r").read()` and `audio, sr = librosa.load(audio_path, sr=None)` if your dataset structure is different, as shown in the code snippet above.
-   `librosa.load()` loads the audio, and `librosa.resample()` resamples it to 16kHz if necessary.
-   The feature extractor and tokenizer then process the audio and text, respectively, into the format required by the Whisper model.

### 2.3. Example: Adapting to Different Column Names

Let's say you have a dataset where:

-   The audio file path is in a column named "audio\_file."
-   The caption text is in a column named "caption\_text."

You would modify the `WhisperAudioCaptionProcessor` like this:

```python
class WhisperAudioCaptionProcessor:
    # ... (rest of the class definition)

    def __call__(self, item):
        # Adapt this part based on your dataset's structure
        caption = item["caption_text"] 
        audio_path = item["audio_file"]

        audio, sr = librosa.load(audio_path, sr=None)
        
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        input_features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = self.tokenizer(caption).input_ids
        return {"input_features": input_features, "labels": input_ids}
```

### 2.4. Loading local datasets

If your dataset is not hosted on Hugging Face Hub, you need to modify the `load_dataset` function in `train.py` to load your local dataset. For example:

```python
# In train.py

# For local datasets, you might need to write a custom loading script
# Or use the datasets library to load from local files
dataset = load_dataset("audiofolder", data_dir="path/to/your/dataset", split="train").map(processor, batched=False)
```
Replace `"path/to/your/dataset"` with the actual path to your local dataset. You might need to specify additional arguments to `load_dataset` depending on the format of your local dataset.

## 3. Configuration with Hydra

Hydra is used for managing configurations, making it easy to change parameters without modifying the code. The configurations are in the `configs` directory.

-   `data.yaml`: Specifies the dataset.
-   `hydra.yaml`: Configures Hydra's logging and output directory structure.
-   `main.yaml`: Main configuration file, importing defaults from others.
-   `model.yaml`: Specifies the Whisper model name.
-   `paths.yaml`: Defines paths for data, logs, and outputs.
-   `trainer.yaml`: Contains arguments for the Hugging Face `Trainer`.

### 3.1. Hydra Configuration Examples

Here are examples of how to set configurations using Hydra, either by modifying the YAML files or through command-line overrides.

**Example 1: Changing the dataset and model in `main.yaml`**

```yaml
# configs/main.yaml

defaults:
  - paths
  - hydra
  - data
  - model
  - trainer
  - _self_

task_name: tiny-test

data_config:
  name: your_username/your_custom_dataset # or your local dataset

model_config:
  name: openai/whisper-small
  attn_implementation: sdpa

# ... other configurations
```

**Example 2: Overriding parameters on the command line**

To train with a different learning rate and batch size, run:

```bash
accelerate launch train.py trainer_config.per_device_train_batch_size=16 ++trainer_config.learning_rate=1e-5
```

**Example 3: Using a different model and enabling mixed-precision training**

```bash
accelerate launch train.py model_config.name=openai/whisper-large ++trainer_config.bf16=true
```

**Example 4: Changing output directory and logging directory.**
```yaml
# configs/main.yaml

trainer_config:
  # ...other configs
  output_dir: ${paths.output_dir}/my_experiment
  logging_dir: ${trainer_config.output_dir}/tb_logs
```

## 4. Training

### 4.1. Single GPU Training

To train on a single GPU:

```bash
accelerate launch train.py
```

Accelerate automatically detects the available GPU. Customize parameters in `configs/trainer.yaml` or override them on the command line.

### 4.2. Multi-GPU Training on a Single Node

To train on all available GPUs:

```bash
accelerate launch train.py trainer_config.per_device_train_batch_size=128 ++trainer_config.bf16=true trainer_config.max_steps=25000 ++trainer_config.push_to_hub=true ++trainer_config.hub_model_id='your-username/your-model-name' ++trainer_config.split_batches=true ++trainer_config.save_total_limit=5 ++trainer_config.save_steps=5000 ++trainer_config.dataloader_prefetch_factor=4 ++trainer_config.dataloader_num_workers=16
```
**Note:** `trainer_config.split_batches=true` is sometimes necessary for multi-GPU training.

### 4.3. Specifying GPU IDs

To train on specific GPUs (e.g., GPUs 1 and 2):

```bash
CUDA_VISIBLE_DEVICES=1,2 accelerate launch train.py
```

## 5. Inference

The `test.py` script is used for running inference with your fine-tuned model.

### 5.1. Inference Script Explanation

The `test.py` script performs the following steps:

1. **Loads the Fine-tuned Model and Processor:**
    
    -   It loads your fine-tuned Whisper model using `WhisperForConditionalGeneration.from_pretrained()`. You need to provide the path to your saved model directory.
    
    -   It loads the corresponding `WhisperAudioCaptionProcessor`, which handles both the feature extractor and the tokenizer, used for preparing data for the model and decoding the model's output.
    
2. **Loads and Preprocesses the Dataset:**
    
    -   It uses the `load_dataset` function from the Hugging Face `datasets` library to load your test dataset. It will need to be adapted to use your data.
    
    -   It applies the `WhisperAudioCaptionTestProcessor` to each item in the dataset. If your data is structured differently, you will need to modify the `WhisperAudioCaptionTestProcessor`'s `__call__` function to match your dataset's structure.
    
    -   It creates a `DataLoader` to efficiently iterate through the dataset in batches, using the `DataCollatorSpeechSeq2SeqWithPadding` to handle padding within each batch.
    
3. **Generates Captions:**
    
    -   It iterates through the `DataLoader`, processing each batch of data.
    
    -   For each batch, it generates captions using `model.generate()`.
    
    -   The inference loop in `test.py` uses greedy decoding by default (`do_sample=False`).
    
4. **Decodes and Stores Results:**
    
    -   It decodes the generated token IDs back into text captions using `tokenizer.batch_decode()`.
    
    -   It stores the generated captions along with the ground truth captions (if available) in a list called `results`.

### 5.2. Inputs and Outputs

**Inputs:**

-   `--model_name`:  **(Required)** The path to your fine-tuned model directory or the name of a model on the Hugging Face Hub.
    
-   `--num_dataloader_workers`: (Optional) The number of worker processes for data loading (default: 4).
    
-   `--batch_size`: (Optional) The batch size for inference (default: 16).
    
-   You also need to adapt the `load_dataset` function to load your test data.

**Outputs:**

-   The script will print a list of dictionaries to the console. Each dictionary contains:
    
    -   `gt_caption`: The ground truth caption (if available in your dataset).
    
    -   `pred_caption`: The caption generated by the model.

**Example Usage:**

```bash
python test.py --model_name /path/to/your/fine-tuned/model --batch_size 8
```

## 6. Additional Notes

-   **Experimentation:** The provided parameters are starting points. Experiment to optimize performance.
    
-   **Data Augmentation:** Consider data augmentation to improve the model's robustness.
    
-   **Monitoring:** Use TensorBoard to monitor training and identify issues.
    
-   **Advanced Configurations:** Explore Hydra and Trainer documentation for more advanced options.

This revised README provides a more comprehensive and detailed guide to fine-tuning Whisper for audio captioning, addressing the use of different audio formats, providing more elaborate Hydra configuration examples, and thoroughly explaining the inference script. Remember to adapt the code and configurations to your specific needs and dataset.
