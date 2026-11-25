# Chatting with the model: inference/chat.py

A terminal chat. It works with the base model on its own or together with the trained LoRA adapter. The script picks the device (MPS, CUDA or CPU) by itself.

## Running it

Base model only, no fine-tuning:

```bash
python inference/chat.py \
  --base_model "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
```

With the retreat adapter:

```bash
python inference/chat.py \
  --base_model "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct" \
  --lora_model "./fine_tuning/lora_retrit"
```

With the adapter and a system prompt (the model writes in Russian; the prompt says "You are an assistant for the Know Yourself retreat brand. Write warmly, honestly, without manipulation."):

```bash
python inference/chat.py \
  --base_model "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct" \
  --lora_model "./fine_tuning/lora_retrit" \
  --system_prompt "Ты помощник бренда ретрита «Познай себя». Пиши тепло, честно, без манипуляций." \
  --temperature 0.3
```

## Parameters

| Parameter | Default | What it sets |
|---|---|---|
| `--base_model` | required | Base model name on Hugging Face |
| `--lora_model` | none | Path to the adapter folder |
| `--system_prompt` | empty | System prompt, the model's role |
| `--max_new_tokens` | `512` | Maximum answer length in tokens |
| `--temperature` | `0.3` | Generation temperature |

For Vikhr a temperature of 0.1–0.3 works well: answers are precise and predictable. At 0.5–0.7 the wording varies more. From 0.8 up the answers get creative but less stable.

## Chat commands

| Command | What it does |
|---|---|
| `quit`, `exit`, `q` | End the session |
| `clear` | Clear the conversation history |

The chat remembers the last 20 messages. The system prompt isn't lost: if it's set, it stays the first message of the history.
