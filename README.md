# A-Z---LLM

## Fine-tuning and Inference with LLaMA Model

This repository contains code for fine-tuning a LLaMA model using PEFT (Parameter-Efficient Fine-Tuning) and Hugging Face's Transformers library. The model is trained on a medical terms dataset and can generate text based on user prompts.

### Requirements

Ensure you have the following Python packages installed:

```bash
pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 huggingface_hub
```

### Model and Dataset

- **Model**: `aboonaji/llama2finetune-v2`
- **Dataset**: `aboonaji/wiki_medical_terms_llam2_format`

### Setup

1. **Install Required Packages**:

   ```bash
   pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 huggingface_hub
   ```

2. **Import Libraries**:

   ```python
   import torch
   from trl import SFTTrainer
   from peft import LoraConfig
   from datasets import load_dataset
   from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
   ```

### Fine-tuning the Model

1. **Load the Model with Quantization**:

   ```python
   llama_model = AutoModelForCausalLM.from_pretrained(
       pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
       quantization_config=BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_compute_dtype=getattr(torch, "float16"),
           bnb_4bit_quant_type="nf4"
       )
   )
   llama_model.config.use_cache = False
   llama_model.config.pretraining_tp = 1
   ```

2. **Load the Tokenizer**:

   ```python
   llama_tokenizer = AutoTokenizer.from_pretrained(
       pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
       trust_remote_code=True
   )
   llama_tokenizer.pad_token = llama_tokenizer.eos_token
   llama_tokenizer.padding_side = "right"
   ```

3. **Set Training Arguments**:

   ```python
   training_arguments = TrainingArguments(
       output_dir="./results",
       per_device_train_batch_size=4,
       max_steps=100
   )
   ```

4. **Prepare the Trainer**:

   ```python
   llama_sft_trainer = SFTTrainer(
       model=llama_model,
       args=training_arguments,
       train_dataset=load_dataset(path="aboonaji/wiki_medical_terms_llam2_format", split="train"),
       tokenizer=llama_tokenizer,
       peft_config=LoraConfig(
           task_type="CAUSAL_LM",
           r=64,
           lora_alpha=16,
           lora_dropout=0.1
       ),
       dataset_text_field="text"
   )
   ```

5. **Train the Model**:

   ```python
   llama_sft_trainer.train()
   ```

### Inference

After fine-tuning, you can generate text using the model:

1. **Create a Text Generation Pipeline**:

   ```python
   user_prompt = "Please tell me about Bursitis"
   text_generation_pipeline = pipeline(
       task="text-generation",
       model=llama_model,
       tokenizer=llama_tokenizer,
       max_length=300
   )
   model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
   print(model_answer[0]['generated_text'])
   ```

### Usage

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Run the Training Script**:

   Ensure you have saved the training code into a Python script (e.g., `train.py`) and then execute:

   ```bash
   python train.py
   ```

3. **Run Inference**:

   Save the inference code into a script (e.g., `inference.py`) and execute:

   ```bash
   python inference.py
   ```

### Notes

- The training process is configured for a quick demonstration with `max_steps=100`. For a more robust model, consider increasing the number of training steps.
- The model uses 4-bit quantization to reduce memory usage and speed up training.
- Ensure your environment supports the required GPU resources for efficient training and inference.

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Acknowledgments

- Hugging Face for providing the Transformers and datasets libraries.
- `aboonaji` for providing the pretrained model and dataset.
