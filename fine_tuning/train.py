"""
Скрипт для fine-tuning модели с использованием LoRA
"""
import os
import json
import time
import sys
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import torch
from transformers import BitsAndBytesConfig

def format_metric(value, fmt):
    """Безопасно форматирует метрику, даже если она пришла строкой."""
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)

class DetailedLoggingCallback(TrainerCallback):
    """Callback для детального логирования процесса обучения"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Вызывается в начале обучения"""
        self.start_time = time.time()
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ | Эпох: {args.num_train_epochs} | Батч: {effective_batch} | LR: {args.learning_rate}")
        print(f"{'='*60}\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Вызывается в начале каждой эпохи"""
        self.epoch_start_time = time.time()
        steps = state.max_steps // int(args.num_train_epochs) if state.max_steps else '?'
        print(f"\nЭпоха {state.epoch}/{int(args.num_train_epochs)} | Шагов: {steps}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при каждом логировании"""
        if logs is None:
            return
        
        step = state.global_step
        loss = logs.get('loss', 'N/A')
        lr = logs.get('learning_rate', 'N/A')
        loss_str = format_metric(loss, ".4f")
        lr_str = format_metric(lr, ".2e")
        
        # Компактный вывод
        if state.max_steps:
            progress = (step / state.max_steps) * 100
            print(f"Шаг {step}/{state.max_steps} ({progress:.1f}%) | Loss: {loss_str} | LR: {lr_str}", end='')
        else:
            print(f"Шаг {step} | Loss: {loss_str} | LR: {lr_str}", end='')
        
        # Память GPU (CUDA или MPS)
        if torch.cuda.is_available() and not getattr(args, "use_cpu", False):
            mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f" | GPU: {mem:.1f}GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "current_allocated_memory"):
                mem = torch.mps.current_allocated_memory() / 1024**3
                print(f" | MPS: {mem:.1f}GB")
            else:
                print()
        else:
            print()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце каждой эпохи"""
        epoch_time = time.time() - self.epoch_start_time
        loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
        print(f"Эпоха {state.epoch} завершена | Время: {epoch_time/60:.1f}мин | Loss: {format_metric(loss, '.4f')}\n")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Вызывается в конце обучения"""
        total_time = time.time() - self.start_time
        loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Время: {total_time/60:.1f}мин | Шагов: {state.global_step} | Loss: {format_metric(loss, '.4f')}")
        print(f"{'='*60}\n")

def _resolve_device(device_arg):
    """
    Определяет устройство для обучения.
    Возвращает (resolved_device, use_gpu, backend) где backend = 'cuda' | 'mps' | 'cpu'.
    """
    if device_arg == "cpu":
        return "cpu", False, "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен device=cuda, но CUDA недоступна")
        return "cuda:0", True, "cuda"

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Запрошен device=mps, но MPS недоступен")
        return "mps", True, "mps"

    # device == "auto": выбираем лучшее доступное
    if torch.cuda.is_available():
        return "cuda:0", True, "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", True, "mps"
    return "cpu", False, "cpu"


def print_system_info(backend):
    """Выводит информацию о системе"""
    print("\n" + "="*60)
    print("СИСТЕМА")
    print("="*60)
    if backend == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"CUDA: {cuda_version}")
        print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        if compute_capability[0] >= 12:
            print("[WARN] RTX 5060 (sm_120) требует PyTorch 2.5+ или nightly build")
    elif backend == "mps":
        import platform
        print(f"GPU: Apple Silicon ({platform.processor() or 'M-series'})")
        print(f"Backend: MPS (Metal Performance Shaders)")
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / 1024**3
            print(f"Unified Memory: {mem_gb:.0f}GB")
        except ImportError:
            pass
    else:
        print("[WARN] ВНИМАНИЕ: GPU недоступен!")
        print("Обучение будет выполняться на CPU")
    print("="*60)

def load_model_and_tokenizer(model_name, use_4bit=True, cache_dir=None, device="auto"):
    """
    Загружает модель и токенизатор
    
    Args:
        model_name: имя модели с HuggingFace
        use_4bit: использовать ли 4-bit quantization для экономии памяти
        cache_dir: директория для сохранения модели (если None, используется локальная директория)
    """
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА МОДЕЛИ: {model_name}")
    print(f"{'='*60}")
    use_cuda = device == "cuda" and torch.cuda.is_available()
    use_mps = device == "mps"
    use_gpu = use_cuda or use_mps

    if use_4bit and not use_cuda:
        print("[WARN] 4-bit quantization доступна только на NVIDIA GPU. Отключаем --use_4bit.")
        use_4bit = False

    if use_4bit:
        print("Quantization: 4-bit")
    
    # Определяем директорию для сохранения модели
    if cache_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "models", model_name.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
    else:
        os.makedirs(cache_dir, exist_ok=True)
    
    load_start = time.time()
    
    # Настройка quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Загрузка токенизатора
    if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        print("Токенизатор: локальный")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.save_pretrained(cache_dir)
        print("Токенизатор: скачан")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Pad token: установлен как eos_token ({tokenizer.pad_token})")
    else:
        print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    # Загрузка модели
    model_start = time.time()
    if use_cuda:
        device_map = "auto"
        torch_dtype = torch.float16 if not use_4bit else None
        torch.cuda.set_device(0)

        compute_cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Устройство: CUDA GPU ({gpu_name})")

        if compute_cap[0] >= 12:
            print(f"[WARN] Обнаружена архитектура sm_{compute_cap[0]}{compute_cap[1]} (Blackwell)")
            print("  Для RTX 5060 требуется PyTorch 2.5+ или newer build")
            if use_4bit:
                print("  [WARN] Quantization может не работать с sm_120")
                print("  Если возникнут ошибки, запустите БЕЗ --use_4bit")
    elif use_mps:
        device_map = None
        torch_dtype = torch.float32
        print("Устройство: MPS (Apple Silicon GPU)")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("Устройство: CPU")
    
    # Проверяем, есть ли уже сохраненная модель локально
    config_path = os.path.join(cache_dir, "config.json")
    model_files = [
        os.path.join(cache_dir, "model.safetensors"),
        os.path.join(cache_dir, "pytorch_model.bin"),
        os.path.join(cache_dir, "model.safetensors.index.json"),
    ]
    
    has_local_model = os.path.exists(config_path) and any(os.path.exists(f) for f in model_files)
    
    # Попытка загрузки с обработкой ошибок quantization
    try:
        if has_local_model:
            if use_4bit:
                quantization_config_path = os.path.join(cache_dir, "quantization_config.json")
                if not os.path.exists(quantization_config_path):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        trust_remote_code=True
                    )
                    if hasattr(model, 'config'):
                        model.config.save_pretrained(cache_dir)
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        cache_dir,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        trust_remote_code=True
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cache_dir,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
        else:
            # Попытка загрузки с обработкой ошибок quantization для sm_120
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config if use_4bit else None,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                if not use_4bit:
                    model.save_pretrained(cache_dir)
                else:
                    if hasattr(model, 'config'):
                        model.config.save_pretrained(cache_dir)
            except RuntimeError as e:
                error_str = str(e)
                if "no kernel image is available" in error_str or "CUDA capability" in error_str or "sm_120" in error_str:
                    print("\n" + "="*60)
                    print("ОШИБКА: Несовместимость с RTX 5060 (sm_120)")
                    print("="*60)
                    print("Проблема: bitsandbytes не поддерживает sm_120")
                    print("\nРешение 1: Установите PyTorch Nightly (рекомендуется)")
                    print("  pip uninstall torch torchvision torchaudio bitsandbytes")
                    print("  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                    print("  pip install bitsandbytes")
                    print("\nРешение 2: Запустите БЕЗ quantization")
                    print("  python train.py --model_name ... --dataset_path ...")
                    print("  (уберите флаг --use_4bit)")
                    print("="*60)
                    sys.exit(1)
                else:
                    raise
    except RuntimeError as e:
        if use_gpu and ("no kernel image is available" in str(e) or "CUDA capability" in str(e)):
            print("\n" + "="*60)
            print("ОШИБКА: Несовместимость CUDA capability")
            print("="*60)
            print("RTX 5060 (sm_120) требует PyTorch 2.5+ или nightly build")
            print("\nРешение 1: Установите PyTorch Nightly")
            print("pip uninstall torch torchvision torchaudio bitsandbytes")
            print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
            print("pip install bitsandbytes")
            print("\nРешение 2: Запустите БЕЗ quantization (медленнее, но работает)")
            print("python train.py --model_name ... --dataset_path ...")
            print("(уберите флаг --use_4bit)")
            print("="*60)
            sys.exit(1)
        else:
            raise
    
    model_time = time.time() - model_start
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры: {total_params/1e6:.1f}M всего, {trainable_params/1e6:.1f}M обучаемых")
    
    # Память GPU
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU память: {mem:.1f}/{total_mem:.1f}GB")
    
    # Подготовка модели для обучения с quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    total_load_time = time.time() - load_start
    print(f"Загружено за {total_load_time:.1f}с\n")
    
    return model, tokenizer

def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Настраивает LoRA для модели
    
    Args:
        model: модель для настройки
        r: rank LoRA
        lora_alpha: alpha параметр LoRA
        lora_dropout: dropout для LoRA
    """
    print(f"{'='*60}")
    print(f"НАСТРОЙКА LoRA | r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"{'='*60}")
    
    # Определяем target_modules в зависимости от архитектуры модели
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else 'unknown'
    
    # Определяем target_modules в зависимости от архитектуры
    if model_type in ['gpt2', 'gpt_neo', 'gpt_neo_x']:
        # GPT-2, DialoGPT, GPT-Neo используют c_attn и c_proj
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif model_type in ['llama', 'mistral', 'mixtral']:
        # LLaMA модели используют q_proj, v_proj и т.д.
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif model_type in ['bloom', 'bloomz']:
        # BLOOM модели
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif model_type in ['opt']:
        # OPT модели
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    else:
        # Пытаемся автоматически найти подходящие модули
        print("  Автоматическое определение модулей...")
        all_module_names = set()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Листовые модули
                module_name = name.split('.')[-1]
                all_module_names.add(module_name)
        
        # Ищем распространенные паттерны
        target_modules = []
        common_patterns = ['attn', 'proj', 'fc', 'dense', 'query', 'key', 'value']
        for pattern in common_patterns:
            matching = [name for name in all_module_names if pattern.lower() in name.lower()]
            target_modules.extend(matching)
        
        if not target_modules:
            # Если ничего не найдено, используем все линейные слои
            target_modules = [name for name in all_module_names if 'linear' in name.lower() or 'weight' in name.lower()]
        
        if not target_modules:
            # Последняя попытка - используем стандартные для GPT-2
            print("  [WARN] Не удалось определить модули, используем стандартные для GPT-2")
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            target_modules = list(set(target_modules))  # Убираем дубликаты
    
    print(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Информация о параметрах
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = (trainable_params / total_params) * 100
    print(f"Обучаемых параметров: {trainable_params/1e6:.2f}M ({trainable_percent:.2f}%)\n")
    
    return model

def load_dataset_from_file(dataset_path):
    """
    Загружает датасет из файла
    
    Поддерживаемые форматы:
    - JSON файл с полем 'text' или 'instruction'/'output'
    - JSONL файл (каждая строка - JSON объект)
    """
    print(f"{'='*60}")
    print(f"ЗАГРУЗКА ДАТАСЕТА: {os.path.basename(dataset_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")
    
    load_start = time.time()
    
    if dataset_path.endswith('.jsonl'):
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())[0] if data else []
    else:
        raise ValueError("Поддерживаются только .json и .jsonl файлы")
    
    load_time = time.time() - load_start
    file_size = os.path.getsize(dataset_path) / 1024**2
    print(f"Примеров: {len(data)} | Размер: {file_size:.1f}MB | Время: {load_time:.1f}с\n")
    
    return data

def _detect_chat_template(tokenizer):
    """Проверяет, есть ли у токенизатора chat template."""
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def _example_to_messages(example):
    """
    Конвертирует пример датасета в список сообщений для chat template.

    Поддерживаемые форматы:
    - instruction/input/output  -> system + user + assistant
    - instruction/output        -> user + assistant
    - prompt/completion         -> user + assistant
    - input/output              -> user + assistant
    - text                      -> user
    """
    if 'instruction' in example and 'input' in example and 'output' in example:
        return [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    if 'instruction' in example and 'output' in example:
        return [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    if 'prompt' in example and 'completion' in example:
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    if 'input' in example and 'output' in example:
        return [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    if 'text' in example:
        return [{"role": "user", "content": example["text"]}]
    text_keys = [k for k in example.keys() if 'text' in k.lower() or 'content' in k.lower()]
    if text_keys:
        return [{"role": "user", "content": example[text_keys[0]]}]
    return [{"role": "user", "content": str(example)}]


def _format_prompt_legacy(example):
    """Форматирование для моделей без chat template (обратная совместимость)."""
    if 'text' in example:
        return example['text']
    if 'instruction' in example and 'input' in example and 'output' in example:
        return (
            f"### System:\n{example['instruction']}\n\n"
            f"### Instruction:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    if 'instruction' in example and 'output' in example:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    if 'prompt' in example and 'completion' in example:
        return f"{example['prompt']}\n\n{example['completion']}"
    if 'input' in example and 'output' in example:
        return f"Input: {example['input']}\nOutput: {example['output']}"
    text_keys = [k for k in example.keys() if 'text' in k.lower() or 'content' in k.lower()]
    if text_keys:
        return example[text_keys[0]]
    return str(example)


def preprocess_dataset(data, tokenizer, max_length=512):
    """
    Предобрабатывает датасет для обучения.

    Для instruct-моделей с chat template:
      - форматирует данные через apply_chat_template
      - маскирует loss на промпте (system + user), оставляя только ответ assistant

    Для моделей без chat template:
      - форматирует как плоский текст (обратная совместимость)
    """
    print(f"{'='*60}")
    print(f"ПРЕДОБРАБОТКА | max_length={max_length}")
    print(f"{'='*60}")

    preprocess_start = time.time()
    use_chat_template = _detect_chat_template(tokenizer)

    if use_chat_template:
        print("Режим: chat template (instruct)")
    else:
        print("Режим: плоский текст (legacy)")

    # --- Статистика длины ---
    text_lengths = []
    for example in data[:min(100, len(data))]:
        if use_chat_template:
            messages = _example_to_messages(example)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        else:
            text = _format_prompt_legacy(example)
            ids = tokenizer.encode(text, add_special_tokens=True)
        text_lengths.append(len(ids))

    if text_lengths:
        avg_tokens = sum(text_lengths) / len(text_lengths)
        max_tokens = max(text_lengths)
        truncated = sum(1 for t in text_lengths if t > max_length)
        print(f"Средняя длина: {avg_tokens:.0f} токенов | Макс: {max_tokens} | Обрезано: {truncated}/{len(text_lengths)}")

    # --- Токенизация ---
    from datasets import Dataset
    dataset = Dataset.from_list(data)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if use_chat_template:
        def tokenize_chat(examples):
            batch_size = len(list(examples.values())[0])
            all_input_ids = []
            all_attention_mask = []
            all_labels = []

            for i in range(batch_size):
                ex = {key: examples[key][i] for key in examples.keys()}
                messages = _example_to_messages(ex)

                full_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    return_dict=False,
                )

                prompt_messages = [m for m in messages if m["role"] != "assistant"]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True,
                    return_dict=False,
                )
                prompt_len = len(prompt_ids)

                full_ids = full_ids[:max_length]
                seq_len = len(full_ids)
                padding_length = max_length - seq_len

                input_ids = full_ids + [pad_id] * padding_length
                attention_mask = [1] * seq_len + [0] * padding_length

                labels = (
                    [-100] * min(prompt_len, seq_len)
                    + full_ids[prompt_len:]
                    + [-100] * padding_length
                )
                labels = labels[:max_length]

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
            }

        tokenized_dataset = dataset.map(
            tokenize_chat,
            batched=True,
            batch_size=256,
            remove_columns=dataset.column_names,
            desc="Токенизация (chat template)",
        )
    else:
        def tokenize_legacy(examples):
            batch_size = len(list(examples.values())[0])
            examples_list = []
            for i in range(batch_size):
                example_dict = {key: examples[key][i] for key in examples.keys()}
                examples_list.append(example_dict)

            texts = [_format_prompt_legacy(ex) for ex in examples_list]
            tokenized = tokenizer(
                texts, truncation=True, max_length=max_length, padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_legacy,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Токенизация (legacy)",
        )

    preprocess_time = time.time() - preprocess_start
    total_tokens = sum(len(ids) for ids in tokenized_dataset['input_ids'])
    non_masked = sum(
        sum(1 for l in labels if l != -100)
        for labels in tokenized_dataset['labels']
    )
    print(f"Токенизировано: {len(tokenized_dataset)} примеров, {total_tokens/1e3:.0f}K токенов")
    if use_chat_template:
        print(f"Токенов для обучения (assistant): {non_masked/1e3:.0f}K ({non_masked*100//total_tokens}% от общего)")
    print(f"Время: {preprocess_time:.1f}с\n")

    return tokenized_dataset, use_chat_template

def train(
    model_name,
    dataset_path,
    output_dir="./lora_model",
    eval_dataset_path=None,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,
    use_4bit=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    save_steps=100,
    logging_steps=5,
    warmup_steps=20,
    device="auto"
):
    """
    Основная функция обучения

    Args:
        model_name: имя модели с HuggingFace
        dataset_path: путь к датасету
        output_dir: директория для сохранения модели
        eval_dataset_path: путь к валидационному датасету (опционально)
        num_train_epochs: количество эпох
        per_device_train_batch_size: размер батча на устройство
        gradient_accumulation_steps: шаги накопления градиента
        learning_rate: скорость обучения
        max_length: максимальная длина последовательности
        use_4bit: использовать ли 4-bit quantization
        lora_r: rank LoRA
        lora_alpha: alpha параметр LoRA
        lora_dropout: dropout для LoRA
        save_steps: шаги сохранения
        logging_steps: шаги логирования
        warmup_steps: шаги warmup
    """
    resolved_device, use_gpu, backend = _resolve_device(device)

    if use_4bit and backend != "cuda":
        print("[WARN] 4-bit quantization доступна только на NVIDIA GPU. Отключаем --use_4bit.")
        use_4bit = False

    # Вывод информации о системе
    print_system_info(backend)
    print(f"Выбранное устройство: {resolved_device} (backend: {backend})")
    
    # Определяем директорию для сохранения обученной модели (в проекте)
    if not os.path.isabs(output_dir):
        # Если путь относительный, делаем его относительно корня проекта
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, output_dir)
    
    # Создаем директорию для сохранения модели
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nДиректория для сохранения обученной модели: {os.path.abspath(output_dir)}")
    
    # Определяем директорию для базовой модели (в проекте)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(project_root, "models", model_name.replace("/", "_"))
    
    # Загрузка модели и токенизатора
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        use_4bit=use_4bit,
        cache_dir=base_model_cache_dir,
        device=backend,
    )
    
    # Настройка LoRA
    model = setup_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    try:
        model_device = next(model.parameters()).device
        if str(model_device) != resolved_device and not (model_device.type == resolved_device):
            print(f"Перемещение модели с {model_device} на {resolved_device}...")
            model = model.to(resolved_device)
        print(f"[OK] Модель на устройстве: {next(model.parameters()).device}")
    except Exception:
        model = model.to(resolved_device)
        print(f"[OK] Модель перемещена на {resolved_device}")
    
    # Загрузка датасета
    data = load_dataset_from_file(dataset_path)
    train_dataset, use_chat_tpl = preprocess_dataset(data, tokenizer, max_length=max_length)

    # Загрузка eval датасета
    eval_dataset = None
    if eval_dataset_path:
        eval_data = load_dataset_from_file(eval_dataset_path)
        eval_dataset, _ = preprocess_dataset(eval_data, tokenizer, max_length=max_length)
        print(f"Eval датасет: {len(eval_dataset)} примеров")

    # Data collator: для chat template labels уже содержат маскировку,
    # используем default_data_collator чтобы не перезаписать их;
    # для legacy формата — DataCollatorForLanguageModeling маскирует pad сам
    if use_chat_tpl:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    
    # Вычисляем общее количество шагов
    total_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps) * num_train_epochs
    steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    
    print("\n" + "="*80)
    print("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    print("="*80)
    print(f"Размер датасета: {len(train_dataset)} примеров")
    print(f"Размер батча на устройство: {per_device_train_batch_size}")
    print(f"Шаги накопления градиента: {gradient_accumulation_steps}")
    print(f"Эффективный размер батча: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"Шагов в эпохе: {steps_per_epoch}")
    print(f"Всего шагов: {total_steps}")
    print(f"Количество эпох: {num_train_epochs}")
    print(f"Скорость обучения: {learning_rate}")
    print(f"Warmup шагов: {warmup_steps}")
    print(f"Шаги логирования: {logging_steps}")
    print(f"Шаги сохранения: {save_steps}")
    use_gradient_checkpointing = False  # Инициализация
    
    # Проверка конкретной GPU модели
    if backend == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if "RTX 5060" in gpu_name or gpu_memory_gb < 10:
            use_gradient_checkpointing = True
            if per_device_train_batch_size > 4:
                print(f"[WARN] Батч {per_device_train_batch_size} может быть слишком большим для 8GB GPU")

    if use_4bit and backend == "cuda":
        optim_name = "paged_adamw_8bit"
    else:
        optim_name = "adamw_torch"

    # Настройки precision и dataloader по backend
    if backend == "cuda":
        fp16 = True
        bf16 = False
        dataloader_pin_memory = True
        dataloader_num_workers = 4
    elif backend == "mps":
        fp16 = False
        bf16 = False
        dataloader_pin_memory = False
        dataloader_num_workers = 0
        use_gradient_checkpointing = False
        print("[OK] MPS: float32, gradient checkpointing выключен")
    else:
        fp16 = False
        bf16 = False
        dataloader_pin_memory = False
        dataloader_num_workers = 0
        use_gradient_checkpointing = False
        print("[WARN] Обучение на CPU будет очень медленным!\n")

    if use_gpu and use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Eval strategy
    eval_strategy = "no"
    eval_steps = None
    if eval_dataset is not None:
        eval_strategy = "steps"
        eval_steps = max(save_steps, 1)
        print(f"Evaluation: каждые {eval_steps} шагов")

    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_num_workers=dataloader_num_workers,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        save_total_limit=3,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        report_to="none",
        use_cpu=(backend == "cpu"),
        optim=optim_name,
        logging_first_step=True,
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if use_gpu else None,
        gradient_checkpointing=use_gradient_checkpointing if use_gpu else False,
    )
    
    # Создаем директорию для логов
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Trainer с callback для детального логирования
    def _make_trainer():
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[DetailedLoggingCallback()],
        )

    trainer = _make_trainer()

    # Убеждаемся что модель на нужном устройстве перед обучением
    try:
        model_device = next(model.parameters()).device
        if model_device.type != backend and not (backend == "cpu" and model_device.type == "cpu"):
            print(f"Перемещение модели на {resolved_device} перед обучением...")
            model = model.to(resolved_device)
            trainer = _make_trainer()
    except Exception:
        model = model.to(resolved_device)
        trainer = _make_trainer()
    
    # Информация о памяти перед обучением
    if backend == "cuda":
        mem = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - mem
        print(f"GPU память: {mem:.1f}/{total:.1f}GB (свободно: {free:.1f}GB)")
        if free < 1.0:
            print("[WARN] Мало памяти! Уменьшите batch_size")
        torch.cuda.empty_cache()
    elif backend == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            mem_mb = torch.mps.current_allocated_memory() / 1024**2
            print(f"MPS память: {mem_mb:.0f}MB выделено")
        print("Обучение на Apple Silicon GPU (MPS)")
    else:
        print("Обучение запускается на CPU. Это может быть заметно медленнее.")
    print()
    
    # Обучение
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    # Финальная статистика
    loss = trainer.state.log_history[-1].get('loss', 'N/A') if trainer.state.log_history else 'N/A'
    speed = len(train_dataset) * num_train_epochs / train_time if train_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Время: {train_time/60:.1f}мин | Скорость: {speed:.1f} примеров/сек | Loss: {format_metric(loss, '.4f')}")
    print(f"{'='*60}\n")
    
    # Сохранение модели
    print(f"Сохранение модели в {output_dir}...")
    save_start = time.time()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_time = time.time() - save_start
    print(f"[OK] Сохранено за {save_time:.1f}с | Путь: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning модели с LoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Имя модели с HuggingFace")
    parser.add_argument("--dataset_path", type=str, required=True, help="Путь к датасету (.json или .jsonl)")
    parser.add_argument("--eval_dataset_path", type=str, default=None, help="Путь к валидационному датасету (.json или .jsonl)")
    parser.add_argument("--output_dir", type=str, default="./lora_model", help="Директория для сохранения")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Шаги накопления градиента")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Скорость обучения")
    parser.add_argument("--max_length", type=int, default=512, help="Максимальная длина последовательности")
    parser.add_argument("--use_4bit", action="store_true", help="Использовать 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout LoRA")
    parser.add_argument("--save_steps", type=int, default=100, help="Шаги сохранения")
    parser.add_argument("--logging_steps", type=int, default=5, help="Шаги логирования")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Шаги warmup")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Устройство: auto, cpu, cuda, mps (Apple Silicon)")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        eval_dataset_path=args.eval_dataset_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        device=args.device,
    )

