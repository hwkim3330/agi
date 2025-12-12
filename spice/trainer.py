"""
SPICE Trainer - Self-Play Training Loop with Online Learning

Implements the SPICE training loop:
1. Collect documents from corpus (browser)
2. Generate tasks (Challenger)
3. Solve tasks (Reasoner)
4. Update weights on successful attempts (LoRA)
5. Prevent forgetting (EWC)

Based on Meta's SPICE paper (arXiv:2510.24684)
"""

import asyncio
import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .corpus import BrowserCorpus
from .challenger import Challenger, Task
from .reasoner import Reasoner, Attempt


@dataclass
class TrainingConfig:
    """SPICE training configuration"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # EWC settings
    ewc_lambda: float = 1000.0  # Importance of old tasks
    ewc_samples: int = 200  # Samples for Fisher computation

    # Self-play settings
    episodes_per_round: int = 10
    min_score_threshold: float = 0.5
    difficulty_increment: float = 0.1

    # Data settings
    data_dir: str = "data/spice"
    checkpoint_dir: str = "checkpoints/spice"


class EWC:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    Key insight: Protect important weights for previous tasks by
    adding a penalty term based on Fisher Information.

    L_total = L_current + (lambda/2) * sum(F_i * (theta_i - theta*_i)^2)
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, dataloader: DataLoader, num_samples: int = 200):
        """
        Compute Fisher Information Matrix diagonal.

        Uses empirical Fisher: F_i = E[(d log p(y|x) / d theta_i)^2]
        """
        self.model.eval()
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break

            inputs = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_dict[n] += p.grad.data.pow(2)

            sample_count += inputs.size(0)

        # Normalize
        for n in fisher_dict:
            fisher_dict[n] /= sample_count

        self.fisher_dict = fisher_dict

        # Store optimal parameters
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}

        print(f"[EWC] Computed Fisher for {len(fisher_dict)} parameters")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term.

        Returns sum of Fisher-weighted squared differences
        from optimal parameters.
        """
        if not self.fisher_dict:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for n, p in model.named_parameters():
            if n in self.fisher_dict and n in self.optimal_params:
                loss += (self.fisher_dict[n] * (p - self.optimal_params[n]).pow(2)).sum()

        return loss

    def save(self, path: str):
        """Save EWC state"""
        state = {
            "fisher_dict": {k: v.cpu() for k, v in self.fisher_dict.items()},
            "optimal_params": {k: v.cpu() for k, v in self.optimal_params.items()}
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load EWC state"""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.fisher_dict = {k: v.to(self.device) for k, v in state["fisher_dict"].items()}
            self.optimal_params = {k: v.to(self.device) for k, v in state["optimal_params"].items()}


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for efficient fine-tuning.

    Instead of full weight update W' = W + ΔW,
    use low-rank decomposition: W' = W + BA
    where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d, k)
    """

    def __init__(self, original_layer: nn.Linear, r: int = 16,
                 alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for param in original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original_layer(x)

        # Add LoRA contribution
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling

        return result

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into original layer"""
        merged = copy.deepcopy(self.original_layer)
        merged.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        return merged


class SPICEDataset(Dataset):
    """Dataset for SPICE training"""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format as instruction-response pair
        text = f"Question: {sample['question']}\n\nAnswer: {sample['response']}"

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()  # Causal LM
        }


class SPICETrainer:
    """
    SPICE Self-Play Training Loop.

    Orchestrates:
    1. Corpus collection (browser)
    2. Task generation (Challenger)
    3. Task solving (Reasoner)
    4. Online weight updates (LoRA + EWC)
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Components
        self.corpus = BrowserCorpus(data_dir=self.config.data_dir)
        self.challenger = Challenger(data_dir=self.config.data_dir)
        self.reasoner = Reasoner(data_dir=self.config.data_dir)

        # Training state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.ewc = None
        self.lora_layers: Dict[str, LoRALayer] = {}

        # Metrics
        self.training_history: List[Dict] = []
        self.current_difficulty = 0.5

        self._load_state()

    async def initialize(self, headless: bool = True):
        """Initialize all components"""

        # Initialize corpus browser
        await self.corpus.initialize(headless=headless)

        # Load model with LoRA
        await self._load_model()

        print("[SPICE] Trainer initialized")

    async def _load_model(self):
        """Load base model and apply LoRA"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[SPICE] Loading model: {self.config.base_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            # Apply LoRA to target modules
            self._apply_lora()

            # Setup optimizer (only LoRA params)
            lora_params = []
            for name, module in self.model.named_modules():
                if isinstance(module, LoRALayer):
                    lora_params.extend([module.lora_A, module.lora_B])

            if lora_params:
                self.optimizer = torch.optim.AdamW(
                    lora_params,
                    lr=self.config.learning_rate
                )

            # Setup EWC
            self.ewc = EWC(self.model)
            ewc_path = os.path.join(self.config.checkpoint_dir, "ewc_state.pt")
            self.ewc.load(ewc_path)

            print(f"[SPICE] Model loaded with {len(self.lora_layers)} LoRA layers")

        except Exception as e:
            print(f"[SPICE] Model loading failed: {e}")
            self.model = None

    def _apply_lora(self):
        """Apply LoRA to target modules"""

        for name, module in self.model.named_modules():
            # Check if this is a target module
            for target in self.config.target_modules:
                if target in name and isinstance(module, nn.Linear):
                    # Replace with LoRA layer
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]

                    parent = self.model.get_submodule(parent_name) if parent_name else self.model

                    lora_layer = LoRALayer(
                        module,
                        r=self.config.lora_r,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )

                    setattr(parent, child_name, lora_layer)
                    self.lora_layers[name] = lora_layer
                    break

    async def train_round(self, corpus_category: str = "tech",
                         collect_count: int = 5,
                         task_count: int = 10) -> Dict:
        """
        Run one round of self-play training.

        Returns:
            Dict with round statistics
        """
        round_start = datetime.now()

        # 1. Collect documents
        print(f"[SPICE] Collecting {collect_count} documents...")
        documents = await self.corpus.collect(
            category=corpus_category,
            count=collect_count
        )

        if not documents:
            # Use existing corpus
            documents = self.corpus.sample(n=collect_count)

        if not documents:
            return {"error": "No documents available"}

        # 2. Generate tasks
        print(f"[SPICE] Generating {task_count} tasks...")
        tasks = await self.challenger.generate_tasks(
            documents=documents,
            count=task_count
        )

        # 3. Solve tasks and collect training data
        print("[SPICE] Solving tasks...")
        successful_samples = []

        for task in tasks:
            attempt = await self.reasoner.solve(task)

            # Calibrate difficulty based on performance
            self.challenger.calibrate_difficulty(task, attempt.score)

            # Collect successful attempts for training
            if attempt.score >= self.config.min_score_threshold:
                successful_samples.append({
                    "question": task.question,
                    "response": attempt.response,
                    "reasoning": attempt.reasoning_trace,
                    "score": attempt.score
                })

        # 4. Update weights
        train_loss = 0.0
        if successful_samples and self.model is not None:
            print(f"[SPICE] Training on {len(successful_samples)} samples...")
            train_loss = await self._train_step(successful_samples)

        # 5. Update difficulty for next round
        reasoner_level = self.reasoner.get_current_level()
        self.current_difficulty = min(1.0, reasoner_level + self.config.difficulty_increment)

        # Record metrics
        round_stats = {
            "timestamp": round_start.isoformat(),
            "documents_collected": len(documents),
            "tasks_generated": len(tasks),
            "successful_attempts": len(successful_samples),
            "success_rate": len(successful_samples) / len(tasks) if tasks else 0,
            "train_loss": train_loss,
            "current_difficulty": self.current_difficulty,
            "reasoner_level": reasoner_level,
            "duration_seconds": (datetime.now() - round_start).total_seconds()
        }

        self.training_history.append(round_stats)
        self._save_state()

        print(f"[SPICE] Round complete: {round_stats}")
        return round_stats

    async def _train_step(self, samples: List[Dict]) -> float:
        """Run one training step with LoRA + EWC"""

        if not self.model or not self.optimizer:
            return 0.0

        # Create dataset
        dataset = SPICEDataset(samples, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.model.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Loss = LM loss + EWC penalty
            loss = outputs.loss
            if self.ewc:
                ewc_loss = self.ewc.penalty(self.model)
                loss = loss + (self.config.ewc_lambda * ewc_loss)

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (steps + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1

        # Update EWC after training
        if self.ewc and steps > 0:
            self.ewc.compute_fisher(dataloader, self.config.ewc_samples)

        return total_loss / steps if steps > 0 else 0.0

    async def continuous_train(self, rounds: int = 100,
                               collect_per_round: int = 3,
                               tasks_per_round: int = 10):
        """
        Run continuous self-play training.

        SPICE insight: Continuous self-play allows the model
        to generate its own curriculum and improve iteratively.
        """
        print(f"[SPICE] Starting continuous training for {rounds} rounds")

        categories = ["tech", "science", "news", "wiki"]

        for round_num in range(rounds):
            print(f"\n{'='*50}")
            print(f"[SPICE] Round {round_num + 1}/{rounds}")
            print(f"{'='*50}")

            # Rotate categories
            category = categories[round_num % len(categories)]

            # Run training round
            stats = await self.train_round(
                corpus_category=category,
                collect_count=collect_per_round,
                task_count=tasks_per_round
            )

            # Print progress
            if "error" not in stats:
                print(f"  Success rate: {stats['success_rate']:.2%}")
                print(f"  Train loss: {stats['train_loss']:.4f}")
                print(f"  Difficulty: {stats['current_difficulty']:.2f}")

            # Checkpoint every 10 rounds
            if (round_num + 1) % 10 == 0:
                await self.save_checkpoint(f"round_{round_num + 1}")

        print("\n[SPICE] Continuous training complete!")
        return self.get_training_stats()

    def get_training_stats(self) -> Dict:
        """Get overall training statistics"""

        if not self.training_history:
            return {"rounds": 0}

        success_rates = [h["success_rate"] for h in self.training_history]
        losses = [h["train_loss"] for h in self.training_history if h.get("train_loss")]

        return {
            "total_rounds": len(self.training_history),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "best_success_rate": max(success_rates),
            "avg_train_loss": sum(losses) / len(losses) if losses else 0,
            "current_difficulty": self.current_difficulty,
            "reasoner_stats": self.reasoner.get_performance_stats(),
            "corpus_size": self.corpus.size(),
            "total_tasks": len(self.challenger.generated_tasks)
        }

    async def save_checkpoint(self, name: str):
        """Save training checkpoint"""

        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save LoRA weights
        if self.lora_layers:
            lora_state = {
                name: {
                    "lora_A": layer.lora_A.data.cpu(),
                    "lora_B": layer.lora_B.data.cpu()
                }
                for name, layer in self.lora_layers.items()
            }
            torch.save(lora_state, os.path.join(checkpoint_path, "lora_weights.pt"))

        # Save EWC state
        if self.ewc:
            self.ewc.save(os.path.join(checkpoint_path, "ewc_state.pt"))

        # Save training state
        state = {
            "training_history": self.training_history,
            "current_difficulty": self.current_difficulty,
            "config": {
                "base_model": self.config.base_model,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha
            }
        }
        with open(os.path.join(checkpoint_path, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(f"[SPICE] Checkpoint saved: {checkpoint_path}")

    async def load_checkpoint(self, name: str):
        """Load training checkpoint"""

        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)

        if not os.path.exists(checkpoint_path):
            print(f"[SPICE] Checkpoint not found: {checkpoint_path}")
            return False

        # Load LoRA weights
        lora_path = os.path.join(checkpoint_path, "lora_weights.pt")
        if os.path.exists(lora_path) and self.lora_layers:
            lora_state = torch.load(lora_path)
            for name, state in lora_state.items():
                if name in self.lora_layers:
                    self.lora_layers[name].lora_A.data = state["lora_A"].to(self.model.device)
                    self.lora_layers[name].lora_B.data = state["lora_B"].to(self.model.device)

        # Load EWC state
        ewc_path = os.path.join(checkpoint_path, "ewc_state.pt")
        if self.ewc:
            self.ewc.load(ewc_path)

        # Load training state
        state_path = os.path.join(checkpoint_path, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.training_history = state.get("training_history", [])
            self.current_difficulty = state.get("current_difficulty", 0.5)

        print(f"[SPICE] Checkpoint loaded: {checkpoint_path}")
        return True

    def _save_state(self):
        """Save current training state"""
        state_file = os.path.join(self.config.data_dir, "trainer_state.json")
        state = {
            "training_history": self.training_history[-100:],  # Keep last 100
            "current_difficulty": self.current_difficulty
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load training state"""
        state_file = os.path.join(self.config.data_dir, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            self.training_history = state.get("training_history", [])
            self.current_difficulty = state.get("current_difficulty", 0.5)

    async def close(self):
        """Close all components"""
        await self.corpus.close()
        self._save_state()

    def __repr__(self) -> str:
        stats = self.get_training_stats()
        return f"SPICETrainer(rounds={stats['total_rounds']}, corpus={stats['corpus_size']})"
