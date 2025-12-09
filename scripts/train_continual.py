#!/usr/bin/env python3
"""
AGI Trinity - Continual Learning Training Script
지속학습 훈련 스크립트

LFM2-VL 모델의 지속적인 학습을 위한 훈련 파이프라인
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# LoRA for efficient fine-tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA fine-tuning disabled.")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ExperienceDataset(Dataset):
    """경험 데이터셋"""

    def __init__(
        self,
        experiences: List[Dict[str, Any]],
        processor,
        max_length: int = 2048
    ):
        self.experiences = experiences
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        exp = self.experiences[idx]

        prompt = exp.get("prompt", "")
        response = exp.get("correction") or exp.get("response", "")

        # 대화 형식으로 구성
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        # 토큰화
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        encoding = self.processor(
            text=text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def load_experiences(data_dir: str, min_quality: float = 0.7) -> List[Dict[str, Any]]:
    """경험 데이터 로드"""
    experiences = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return experiences

    # JSON 파일들 로드
    for json_file in data_path.glob("experiences_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for exp in data:
                # 품질 필터링
                quality = exp.get("quality_score")
                if quality is not None and quality >= min_quality:
                    experiences.append(exp)
                elif quality is None and exp.get("correction"):
                    # 수정이 있으면 학습 대상
                    experiences.append(exp)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Loaded {len(experiences)} high-quality experiences")
    return experiences


def setup_lora(model, config: Dict[str, Any]):
    """LoRA 설정"""
    if not PEFT_AVAILABLE:
        return model

    lora_config = LoraConfig(
        r=config.get("r", 8),
        lora_alpha=config.get("alpha", 16),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train(
    model_id: str = "LiquidAI/LFM2-VL-1.6B",
    data_dir: str = "~/.trinity/lfm2_memory",
    output_dir: str = "~/.trinity/lfm2_memory/model_checkpoint",
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    num_epochs: int = 3,
    use_lora: bool = True,
    min_quality: float = 0.7,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    fp16: bool = False,
    bf16: bool = True,
    resume_from_checkpoint: Optional[str] = None
):
    """메인 훈련 함수"""

    data_dir = os.path.expanduser(data_dir)
    output_dir = os.path.expanduser(output_dir)

    print(f"=" * 60)
    print("AGI Trinity - Continual Learning Training")
    print(f"=" * 60)
    print(f"Model: {model_id}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"LoRA: {use_lora}")
    print(f"=" * 60)

    # 경험 데이터 로드
    experiences = load_experiences(data_dir, min_quality)

    if not experiences:
        print("No training data found. Please interact with the model first.")
        return

    # 모델 및 프로세서 로드
    print("\nLoading model and processor...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # dtype 설정
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA 적용
    if use_lora:
        print("\nApplying LoRA...")
        model = setup_lora(model, {
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "dropout": 0.05
        })

    # 데이터셋 생성
    print("\nPreparing dataset...")
    dataset = ExperienceDataset(experiences, processor)

    # 훈련/검증 분할
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        fp16=fp16 and torch.cuda.is_available(),
        bf16=bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="adamw_torch"
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 훈련 시작
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    try:
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        # 결과 저장
        print("\nSaving model...")
        trainer.save_model()
        trainer.save_state()

        # 메트릭 저장
        metrics_file = Path(output_dir) / "training_metrics.json"
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_id": model_id,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "use_lora": use_lora,
                "num_experiences": len(experiences)
            }
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nTraining completed!")
        print(f"Model saved to: {output_dir}")
        print(f"Final loss: {train_result.training_loss:.4f}")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise

    finally:
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="AGI Trinity Continual Learning Training"
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="LiquidAI/LFM2-VL-1.6B",
        help="Model ID or path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/.trinity/lfm2_memory",
        help="Directory containing experience data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.trinity/lfm2_memory/model_checkpoint",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum quality score for training data"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training"
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    args = parser.parse_args()

    train(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_lora=not args.no_lora,
        min_quality=args.min_quality,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=args.fp16,
        bf16=not args.no_bf16,
        resume_from_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()
