# Обучение адаптера: fine_tuning/train.py

Скрипт обучает LoRA-адаптер поверх базовой модели с Hugging Face. Он работает и с instruct-моделями, у которых есть chat template, и с моделями без него.

## Что делает скрипт

1. При первом запуске скачивает базовую модель с Hugging Face в кеш `~/.cache/huggingface/`.
2. Выбирает устройство. С `--device auto` порядок такой: CUDA, потом MPS, потом CPU.
3. Подключает LoRA. Обучается около 0,9% параметров модели.
4. Превращает каждый пример датасета в диалог через chat template модели.
5. Считает loss только на ответах ассистента, чтобы модель училась отвечать и не заучивала промпты.
6. Обучает, по желанию с валидацией, и пишет loss в лог.
7. Сохраняет адаптер в папку из `--output_dir`.

## Формат датасета

Скрипт читает `.json` и `.jsonl` и сам определяет формат по полям примера.

| Поля в примере | Как скрипт их понимает |
|---|---|
| `instruction` + `input` + `output` | system = instruction, user = input, assistant = output |
| `instruction` + `output` | user = instruction, assistant = output |
| `prompt` + `completion` | user = prompt, assistant = completion |
| `input` + `output` | user = input, assistant = output |
| `text` | обычный текст без ролей |

В датасете ретрита используется первый вариант:

```json
{
  "instruction": "Напиши маркетинговый текст для ретрита «Познай себя» в фирменном тоне бренда...",
  "input": "Напиши текст для сторис с опросом",
  "output": "Честный вопрос..."
}
```

## Параметры

Обязательные:

| Параметр | Что задает |
|---|---|
| `--model_name` | Имя модели на Hugging Face, например `Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct` |
| `--dataset_path` | Путь к обучающему датасету `.json` или `.jsonl` |

Обучение:

| Параметр | По умолчанию | Что задает |
|---|---|---|
| `--eval_dataset_path` | нет | Валидационный датасет, по нему видно переобучение |
| `--output_dir` | `./lora_model` | Куда сохранить адаптер |
| `--num_train_epochs` | `3` | Число эпох |
| `--per_device_train_batch_size` | `4` | Размер батча |
| `--gradient_accumulation_steps` | `4` | Накопление градиента. Эффективный батч равен батчу, умноженному на это число |
| `--learning_rate` | `2e-4` | Скорость обучения |
| `--max_length` | `512` | Максимальная длина примера в токенах |
| `--device` | `auto` | `auto`, `cpu`, `cuda` или `mps` |
| `--use_4bit` | выключено | Сжатие модели до 4 бит. Работает только на NVIDIA, на других устройствах скрипт его отключит сам |

LoRA:

| Параметр | По умолчанию | Что задает |
|---|---|---|
| `--lora_r` | `16` | Ранг адаптера. Чем он выше, тем больше обучаемых параметров |
| `--lora_alpha` | `32` | Масштаб обновлений, обычно вдвое больше ранга |
| `--lora_dropout` | `0.05` | Регуляризация |

Шаги и лог:

| Параметр | По умолчанию | Что задает |
|---|---|---|
| `--save_steps` | `100` | Чекпоинт каждые N шагов |
| `--logging_steps` | `5` | Loss в логе каждые N шагов |
| `--warmup_steps` | `20` | Шаги разогрева learning rate |

## Команды запуска

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

NVIDIA с 4-битным сжатием:

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

CPU, чтобы быстро проверить, что все запускается:

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

## Что получается на выходе

```text
lora_retrit/
├── adapter_config.json        # настройки LoRA
├── adapter_model.safetensors  # веса адаптера
├── tokenizer.json             # токенизатор
├── tokenizer_config.json
└── chat_template.jinja        # chat template модели
```

Папка адаптера в `.gitignore`, в репозиторий она не входит. Подключить адаптер к чату можно через `inference/chat.py`, примеры в `inference/README.md`.
