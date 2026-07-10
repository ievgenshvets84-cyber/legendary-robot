"""Опциональное дообучение весов через LoRA (QLoRA).

В отличие от памяти-RAG (обучение без изменения весов), этот модуль реально
адаптирует открытую модель под ваши данные методом LoRA. Это тяжёлый путь:
нужны GPU, transformers/peft/datasets и конвертация результата обратно в
формат Ollama (GGUF). Поэтому зависимости импортируются лениво, и модуль не
мешает работе ядра, если библиотеки не установлены.

Формат датасета — JSONL, по одному объекту на строку:
    {"instruction": "...", "input": "", "output": "..."}

Запуск:
    python -m localmind.train --data state/train.jsonl --base Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def format_example(row: dict) -> str:
    instruction = row.get("instruction", "").strip()
    context = row.get("input", "").strip()
    output = row.get("output", "").strip()
    user = instruction if not context else f"{instruction}\n\n{context}"
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>")


def train(data: str, base: str, out_dir: str, epochs: int, lr: float) -> None:
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   DataCollatorForLanguageModeling, Trainer,
                                   TrainingArguments)
    except ImportError as exc:
        raise SystemExit(
            "Для дообучения нужны пакеты ML. Установите:\n"
            "    pip install 'localmind[train]'\n"
            "или напрямую: pip install torch transformers peft datasets accelerate bitsandbytes\n"
            f"Причина: {exc}"
        )

    rows = load_jsonl(data)
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [format_example(r) for r in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir, num_train_epochs=epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=lr, logging_steps=5, save_strategy="epoch",
            bf16=True, report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nLoRA-адаптер сохранён в {out_dir}.")
    print("Дальше: слейте адаптер с базой и сконвертируйте в GGUF "
          "(llama.cpp: convert + quantize), затем создайте Ollama-модель через Modelfile.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LoRA-дообучение открытой модели для LocalMind.")
    p.add_argument("--data", required=True, help="JSONL с примерами обучения")
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="базовая модель HF")
    p.add_argument("--out", default="state/lora-adapter", help="куда сохранить адаптер")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args(argv)
    train(args.data, args.base, args.out, args.epochs, args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
