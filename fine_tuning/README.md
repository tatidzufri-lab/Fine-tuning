# Дообучение модели — fine_tuning/train.py

Скрипт обучает LoRA-адаптер поверх базовой модели с HuggingFace. Поддерживает instruct-модели с chat template и модели без него.

## Как работает

1. Скачивает базовую модель с HuggingFace (при первом запуске)
2. Определяет устройство — MPS, CUDA или CPU
3. Применяет LoRA-конфигурацию (обучается ~0.9% параметров)
4. Форматирует датасет через chat template модели
5. Маскирует loss на промпте — модель учится только на ответах ассистента
6. Обучает с опциональной валидацией и логированием
7. Сохраняет адаптер в указанную директорию

## Форматы датасета

Поддерживаются файлы `.json` и `.jsonl`. Скрипт автоматически определяет формат:

| Поля в примере | Как интерпретируется |
|---|---|
| `instruction` + `input` + `output` | system=instruction, user=input, assistant=output |
| `instruction` + `output` | user=instruction, assistant=output |
| `prompt` + `completion` | user=prompt, assistant=completion |
| `input` + `output` | user=input, assistant=output |
| `text` | плоский текст |

Датасет проекта использует формат `instruction/input/output`:

```json
{
  "instruction": "Напиши маркетинговый текст для ретрита «Познай себя»...",
  "input": "Напиши текст для сторис с опросом",
  "output": "Честный вопрос 🌿\n\nКогда ты в последний раз чувствовала себя собой?..."
}
```

## Параметры запуска

### Обязательные

| Параметр | Описание |
|---|---|
| `--model_name` | имя модели с HuggingFace, например `Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct` |
| `--dataset_path` | путь к тренировочному датасету `.json` или `.jsonl` |

### Часто используемые

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--eval_dataset_path` | нет | валидационный датасет для отслеживания переобучения |
| `--output_dir` | `./lora_model` | куда сохранить обученный адаптер |
| `--num_train_epochs` | `3` | количество эпох |
| `--per_device_train_batch_size` | `4` | размер батча |
| `--gradient_accumulation_steps` | `4` | накопление градиента (эффективный батч = batch × steps) |
| `--learning_rate` | `2e-4` | скорость обучения |
| `--max_length` | `512` | максимальная длина последовательности в токенах |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

### LoRA-параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--lora_r` | `16` | rank адаптера — чем выше, тем больше параметров |
| `--lora_alpha` | `32` | масштаб обновлений (обычно = 2 × r) |
| `--lora_dropout` | `0.05` | регуляризация |

### Шаги и логирование

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--save_steps` | `100` | каждые N шагов сохраняется чекпоинт |
| `--logging_steps` | `5` | каждые N шагов выводится loss |
| `--warmup_steps` | `20` | шаги прогрева learning rate |

### Дополнительные

| Параметр | Описание |
|---|---|
| `--use_4bit` | 4-bit квантование (только CUDA, экономит видеопамять) |

## Команды запуска

### Apple Silicon (MPS)

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

### NVIDIA GPU

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

### CPU (для проверки без GPU)

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

## Результат

После завершения в `output_dir` появятся:

```text
lora_retrit/
├── adapter_config.json       # конфигурация LoRA
├── adapter_model.safetensors # веса адаптера
├── tokenizer.json            # токенизатор
├── tokenizer_config.json
└── chat_template.jinja       # chat-шаблон модели
```

## Примечания

- На MPS (Apple Silicon) 4-bit квантование не поддерживается — скрипт отключит его автоматически
- Базовая модель скачивается в кеш HuggingFace (`~/.cache/huggingface/`) при первом запуске
- `--device auto` выбирает CUDA → MPS → CPU в порядке приоритета
