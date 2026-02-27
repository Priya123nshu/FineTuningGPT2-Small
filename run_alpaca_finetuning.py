import urllib.request
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
import tiktoken

# IMPORTANT: This script assumes you have the helper functions from earlier chapters
# in your local directory (e.g., previous_chapters.py, gpt_download.py).
try:
    from previous_chapters import (
        GPTModel,
        calc_loss_loader,
        generate_text_simple,
        load_weights_into_gpt,
        text_to_token_ids,
        token_ids_to_text,
        train_model_simple,
    )
except ImportError:
    print("Warning: Ensure 'previous_chapters.py' is in the same directory.")
    print("You can download it from the book's Chapter 7 GitHub repository.")

try:
    from gpt_download import download_and_load_gpt2
except ImportError:
    print("Warning: Ensure 'gpt_download.py' is in the same directory.")


# ---------------------------------------------------------
# 1. Formatting and Data Loading Functions
# ---------------------------------------------------------

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # Pad sequences to match the longest sequence
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Replace pad_token_id in targets with ignore_index
        # so gradients are not computed for padding
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def main():
    # ---------------------------------------------------------
    # 2. Setup Configuration
    # ---------------------------------------------------------
    
    # Selected GPT-2 Small
    CHOOSE_MODEL = "gpt2-small (124M)" 
    
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    # Verify GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 3. Load Dataset
    # ---------------------------------------------------------
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

    # NOTE: The script automatically downloads the JSON dataset if it isn't there!
    if not os.path.exists(file_path):
        print(f"Downloading instruction data from {url}...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # ---------------------------------------------------------
    # 4. Tokenizer & DataLoaders
    # ---------------------------------------------------------
    tokenizer = tiktoken.get_encoding("gpt2")
    
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 2 # Crucial for 5GB limit

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=True, drop_last=True, num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=False, drop_last=False, num_workers=num_workers
    )

    # ---------------------------------------------------------
    # 5. Load Pretrained Model
    # ---------------------------------------------------------
    print(f"Loading {CHOOSE_MODEL} weights...")
    # Fixed model_size extraction and added models_dir argument
    model_size = CHOOSE_MODEL.split(" ")[-1].strip("()")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    if torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)
        print("Enabled bfloat16 precision for memory savings on A100.")
        
    model.to(device)
    
    # ---------------------------------------------------------
    # 6. Training Configuration & Loop
    # ---------------------------------------------------------
    print("Starting instruction fine-tuning...")
    import time
    start_time = time.time()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    
    # Number of epochs set to 2 for a standard fine-tuning experiment
    num_epochs = 2 
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.2f} seconds.")

    # ---------------------------------------------------------
    # 7. Save the Fine-Tuned Model (The "Artifact")
    # ---------------------------------------------------------
    save_path = f"gpt2-small-alpaca-finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"*** JOB FINISHED ***")
    print(f"Model saved to {save_path}. PLEASE DOWNLOAD THIS FILE BEFORE SESSION ENDS!")

if __name__ == "__main__":
    main()
