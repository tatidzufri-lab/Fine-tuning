"""
Скрипт для запуска модели в терминале для общения
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse

def load_model_and_tokenizer(base_model_name, lora_model_path=None):
    """
    Загружает модель и токенизатор
    
    Args:
        base_model_name: имя базовой модели с HuggingFace
        lora_model_path: путь к дообученной LoRA модели (опционально)
    """
    print(f"Загрузка модели {base_model_name}...")
    
    # Выбор устройства: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        device_map = "auto"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{'='*60}")
        print(f"Устройство: CUDA GPU ({gpu_name}, {gpu_mem:.1f}GB)")
        print(f"{'='*60}\n")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
        device_map = None
        print(f"\n{'='*60}")
        print(f"Устройство: MPS (Apple Silicon GPU)")
        print(f"{'='*60}\n")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        device_map = None
        print(f"\n[WARN] GPU не обнаружен, используется CPU")
        print(f"Генерация на CPU будет медленной!\n")
    
    # Определяем директорию для базовой модели (в проекте)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(project_root, "models", base_model_name.replace("/", "_"))
    os.makedirs(base_model_cache_dir, exist_ok=True)
    
    config_path = os.path.join(base_model_cache_dir, "config.json")
    has_local_model = os.path.exists(config_path)
    
    # Проверяем, есть ли локальная копия модели
    if has_local_model:
        print(f"Использование локальной копии модели из {base_model_cache_dir}")
        model_path = base_model_cache_dir
    else:
        print(f"Модель будет скачана и сохранена в {base_model_cache_dir}")
        model_path = base_model_name
    
    # Загрузка токенизатора
    tokenizer_config_path = os.path.join(base_model_cache_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"Загрузка токенизатора из локальной директории...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_cache_dir)
    else:
        print(f"Загрузка токенизатора из HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(base_model_cache_dir)
        print(f"Токенизатор сохранен в {base_model_cache_dir}")
    
    # Установка pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Загрузка базовой модели
    print(f"Загрузка модели из {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Сохраняем модель локально, если она была скачана
    if not has_local_model:
        print(f"Сохранение модели в {base_model_cache_dir}...")
        model.save_pretrained(base_model_cache_dir)
        print(f"Модель сохранена!")
    
    # Загрузка LoRA весов если указан путь
    if lora_model_path and os.path.exists(lora_model_path):
        print(f"Загрузка LoRA весов из {lora_model_path}...")
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()
        print("LoRA веса успешно загружены и объединены!")
    elif lora_model_path:
        print(f"Предупреждение: Путь {lora_model_path} не найден. Используется базовая модель.")

    # Перемещаем на MPS если доступен (device_map=None загружает на CPU)
    if device == "mps":
        print("Перемещение модели на MPS...")
        model = model.to("mps")

    model.eval()
    return model, tokenizer

def _has_chat_template(tokenizer):
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.3, top_p=0.9, top_k=50):
    """
    Генерирует ответ модели.

    Для instruct-моделей с chat template принимает список messages и
    форматирует их через apply_chat_template.
    Для legacy-моделей принимает messages[0]["content"] как плоский промпт.
    """
    if _has_chat_template(tokenizer):
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=False,
        )
    else:
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

    device = model.device if hasattr(model, 'device') else ("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    generated_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

def chat_loop(model, tokenizer, system_prompt="", max_new_tokens=512, temperature=0.3):
    """
    Основной цикл чата.
    Формирует messages в формате chat template и передаёт в generate_response.
    """
    use_chat_tpl = _has_chat_template(tokenizer)

    print("\n" + "="*50)
    print("Чат с моделью запущен!")
    if use_chat_tpl:
        print("Режим: chat template (instruct)")
    else:
        print("Режим: legacy prompt")
    print("Введите 'quit', 'exit' или 'q' для выхода")
    print("Введите 'clear' для очистки истории")
    print("="*50 + "\n")

    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("Вы: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("До свидания!")
                break

            if user_input.lower() == 'clear':
                conversation_history = []
                if system_prompt:
                    conversation_history.append({"role": "system", "content": system_prompt})
                print("История очищена.\n")
                continue

            messages = list(conversation_history)
            messages.append({"role": "user", "content": user_input})

            print("Модель генерирует ответ...")
            response = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            print(f"\nМодель: {response}\n")

            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

            if len(conversation_history) > 20:
                if system_prompt:
                    conversation_history = [conversation_history[0]] + conversation_history[-19:]
                else:
                    conversation_history = conversation_history[-20:]

        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            continue

def main():
    parser = argparse.ArgumentParser(description="Чат с моделью в терминале")
    parser.add_argument("--base_model", type=str, required=True, help="Имя базовой модели с HuggingFace")
    parser.add_argument("--lora_model", type=str, default=None, help="Путь к дообученной LoRA модели")
    parser.add_argument("--system_prompt", type=str, default="", help="Системный промпт")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Максимальное количество новых токенов")
    parser.add_argument("--temperature", type=float, default=0.3, help="Температура генерации (рекомендуется 0.3 для Vikhr)")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.lora_model)

    chat_loop(
        model,
        tokenizer,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

if __name__ == "__main__":
    main()

