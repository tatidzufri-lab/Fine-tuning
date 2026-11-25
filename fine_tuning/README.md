# Training the adapter: fine_tuning/train.py

The script trains a LoRA adapter on top of a base model from Hugging Face. It works both with instruct models that have a chat template and with models that don't.

## What the script does

1. On first run it downloads the base model from Hugging Face into the `~/.cache/huggingface/` cache.
2. Picks the device. With `--device auto` the order is CUDA, then MPS, then CPU.
3. Attaches LoRA. About 0.9% of the model's parameters are trained.
4. Turns every dataset example into a dialog through the model's chat template.
5. Computes loss only on the assistant's answers, so the model learns to answer and doesn't memorize prompts.
6. Trains, optionally with validation, and logs the loss.
7. Saves the adapter to the folder given in `--output_dir`.

## Dataset format

The script reads `.json` and `.jsonl` and detects the format from the example's fields.

| Fields in the example | How the script reads them |
|---|---|
| `instruction` + `input` + `output` | system = instruction, user = input, assistant = output |
| `instruction` + `output` | user = instruction, assistant = output |
| `prompt` + `completion` | user = prompt, assistant = completion |
| `input` + `output` | user = input, assistant = output |
| `text` | plain text with no roles |

The retreat dataset uses the first format (the data is in Russian):

```json
{
  "instruction": "Напиши маркетинговый текст для ретрита «Познай себя» в фирменном тоне бренда...",
  "input": "Напиши текст для сторис с опросом",
  "output": "Честный вопрос..."
}
```

In English: the instruction asks for a marketing text for the "Know Yourself" retreat in the brand's tone, the input asks for a stories text with a poll, and the output is the finished text.

## Parameters

Required:

| Parameter | What it sets |
|---|---|
| `--model_name` | Model name on Hugging Face, for example `Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct` |
| `--dataset_path` | Path to the training dataset, `.json` or `.jsonl` |

Training:

| Parameter | Default | What it sets |
|---|---|---|
| `--eval_dataset_path` | none | Validation dataset; it shows overfitting |
| `--output_dir` | `./lora_model` | Where to save the adapter |
| `--num_train_epochs` | `3` | Number of epochs |
| `--per_device_train_batch_size` | `4` | Batch size |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation. The effective batch is the batch size times this number |
| `--learning_rate` | `2e-4` | Learning rate |
| `--max_length` | `512` | Maximum example length in tokens |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `--use_4bit` | off | 4-bit quantization. Works only on NVIDIA; on other devices the script turns it off by itself |

LoRA:

| Parameter | Default | What it sets |
|---|---|---|
| `--lora_r` | `16` | Adapter rank. The higher it is, the more trainable parameters |
| `--lora_alpha` | `32` | Update scale, usually twice the rank |
| `--lora_dropout` | `0.05` | Regularization |

Steps and logging:

| Parameter | Default | What it sets |
|---|---|---|
| `--save_steps` | `100` | A checkpoint every N steps |
| `--logging_steps` | `5` | Loss in the log every N steps |
| `--warmup_steps` | `20` | Learning rate warm-up steps |

## Commands

Apple Silicon (MPS):

```bash
python fine_tuning/train.py \
  --model_name "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct" \
  --dataset_path "dataset_retrit_poznaisebia/train.jsonl" \
  --eval_dataset_path "dataset_retrit_poznaisebia/valid.jsonl" \
  --output_dir "./fine_tuning/lora_retrit" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --device auto
```

NVIDIA with 4-bit quantization:

```bash
python fine_tuning/train.py \
  --model_name "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct" \
  --dataset_path "dataset_retrit_poznaisebia/train.jsonl" \
  --eval_dataset_path "dataset_retrit_poznaisebia/valid.jsonl" \
  --output_dir "./fine_tuning/lora_retrit" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --use_4bit \
  --device cuda
```

CPU, for a quick check that everything starts:

```bash
python fine_tuning/train.py \
  --model_name "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct" \
  --dataset_path "dataset_retrit_poznaisebia/train.jsonl" \
  --output_dir "./fine_tuning/lora_retrit" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_length 256 \
  --device cpu
```

## Output

```text
lora_retrit/
├── adapter_config.json        # LoRA settings
├── adapter_model.safetensors  # adapter weights
├── tokenizer.json             # tokenizer
├── tokenizer_config.json
└── chat_template.jinja        # the model's chat template
```

The adapter folder is in `.gitignore` and isn't part of the repository. You can attach the adapter to the chat with `inference/chat.py`; examples are in `inference/README.md`.
