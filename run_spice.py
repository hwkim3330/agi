#!/usr/bin/env python3
"""
SPICE Runner - Self-Play In Corpus Environments

Run continuous self-play training with:
- Browser-based corpus collection
- Challenger/Reasoner adversarial loop
- LoRA online weight updates
- EWC catastrophic forgetting prevention

Usage:
    python run_spice.py                    # Run default training
    python run_spice.py --rounds 50        # Run 50 rounds
    python run_spice.py --visible          # Show browser
    python run_spice.py --demo             # Quick demo mode
"""

import argparse
import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def run_demo():
    """Run a quick demonstration of SPICE components"""
    print("\n" + "="*60)
    print("SPICE Demo - Self-Play In Corpus Environments")
    print("="*60)

    from spice import BrowserCorpus, Challenger, Reasoner

    # 1. Corpus Demo
    print("\n[1] Corpus Component")
    print("-" * 40)

    corpus = BrowserCorpus(data_dir="data/spice")
    print(f"  Corpus initialized: {corpus}")
    print(f"  Documents in corpus: {len(corpus)}")

    # Sample if available
    if len(corpus) > 0:
        samples = corpus.sample(n=2)
        for doc in samples:
            print(f"  - {doc.title[:50]}... ({doc.domain})")

    # 2. Challenger Demo
    print("\n[2] Challenger Component")
    print("-" * 40)

    challenger = Challenger(data_dir="data/spice")
    print(f"  Challenger: {challenger}")

    # Show task types
    print("  Task types: qa_factual, qa_inference, summarization, reasoning")

    # 3. Reasoner Demo
    print("\n[3] Reasoner Component")
    print("-" * 40)

    reasoner = Reasoner(data_dir="data/spice")
    print(f"  Reasoner: {reasoner}")

    stats = reasoner.get_performance_stats()
    if stats["total_attempts"] > 0:
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Average score: {stats['average_score']:.2%}")

    # 4. Training Config
    print("\n[4] Training Configuration")
    print("-" * 40)

    from spice import TrainingConfig
    config = TrainingConfig()
    print(f"  Base model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  EWC lambda: {config.ewc_lambda}")
    print(f"  Learning rate: {config.learning_rate}")

    print("\n" + "="*60)
    print("Demo complete! Run with --rounds N for actual training.")
    print("="*60 + "\n")


async def run_training(args):
    """Run full SPICE training"""
    from spice import SPICETrainer, TrainingConfig

    print("\n" + "="*60)
    print("SPICE Training - Self-Play In Corpus Environments")
    print("="*60)

    # Configure
    config = TrainingConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.lr,
        ewc_lambda=args.ewc_lambda,
        lora_r=args.lora_rank
    )

    # Create trainer
    trainer = SPICETrainer(config)

    try:
        # Initialize
        print("\n[*] Initializing trainer...")
        await trainer.initialize(headless=not args.visible)

        # Load checkpoint if specified
        if args.checkpoint:
            await trainer.load_checkpoint(args.checkpoint)

        # Run training
        print(f"\n[*] Starting training for {args.rounds} rounds...")
        print(f"    Category: {args.category}")
        print(f"    Documents per round: {args.docs}")
        print(f"    Tasks per round: {args.tasks}")

        if args.continuous:
            # Continuous training
            stats = await trainer.continuous_train(
                rounds=args.rounds,
                collect_per_round=args.docs,
                tasks_per_round=args.tasks
            )
        else:
            # Single round
            for i in range(args.rounds):
                print(f"\n--- Round {i+1}/{args.rounds} ---")
                stats = await trainer.train_round(
                    corpus_category=args.category,
                    collect_count=args.docs,
                    task_count=args.tasks
                )
                print(f"Success rate: {stats.get('success_rate', 0):.2%}")

        # Final stats
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        final_stats = trainer.get_training_stats()
        print(f"  Total rounds: {final_stats['total_rounds']}")
        print(f"  Average success rate: {final_stats['avg_success_rate']:.2%}")
        print(f"  Corpus size: {final_stats['corpus_size']}")
        print(f"  Total tasks: {final_stats['total_tasks']}")

        # Save final checkpoint
        await trainer.save_checkpoint("final")
        print("\n  Checkpoint saved to: checkpoints/spice/final")

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        await trainer.save_checkpoint("interrupted")

    finally:
        await trainer.close()


async def run_corpus_only(args):
    """Just collect corpus without training"""
    from spice import BrowserCorpus

    print("\n[*] Corpus Collection Mode")
    print("-" * 40)

    corpus = BrowserCorpus(data_dir=args.data_dir)

    try:
        await corpus.initialize(headless=not args.visible)

        print(f"Collecting {args.docs} documents from '{args.category}'...")

        documents = await corpus.collect(
            category=args.category,
            count=args.docs
        )

        print(f"\nCollected {len(documents)} documents:")
        for doc in documents:
            print(f"  - {doc.title[:60]}...")
            print(f"    URL: {doc.url}")
            print(f"    Content: {len(doc.content)} chars")
            print()

        print(f"Total corpus size: {len(corpus)}")

    finally:
        await corpus.close()


def main():
    parser = argparse.ArgumentParser(
        description="SPICE - Self-Play In Corpus Environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_spice.py --demo                  # Quick demo
    python run_spice.py --rounds 10             # 10 training rounds
    python run_spice.py --rounds 100 --continuous  # Continuous training
    python run_spice.py --corpus-only --docs 20    # Just collect corpus
    python run_spice.py --visible              # Show browser window
        """
    )

    # Mode
    parser.add_argument("--demo", action="store_true",
                       help="Run quick demonstration")
    parser.add_argument("--corpus-only", action="store_true",
                       help="Only collect corpus, no training")

    # Training settings
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of training rounds (default: 10)")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous training mode")
    parser.add_argument("--category", default="tech",
                       choices=["tech", "science", "news", "wiki"],
                       help="Corpus category (default: tech)")

    # Per-round settings
    parser.add_argument("--docs", type=int, default=3,
                       help="Documents to collect per round (default: 3)")
    parser.add_argument("--tasks", type=int, default=10,
                       help="Tasks to generate per round (default: 10)")

    # Model settings
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank (default: 16)")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0,
                       help="EWC importance (default: 1000)")

    # Paths
    parser.add_argument("--data-dir", default="data/spice",
                       help="Data directory (default: data/spice)")
    parser.add_argument("--checkpoint-dir", default="checkpoints/spice",
                       help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str,
                       help="Load from checkpoint")

    # Display
    parser.add_argument("--visible", action="store_true",
                       help="Show browser window")

    args = parser.parse_args()

    # Run
    if args.demo:
        asyncio.run(run_demo())
    elif args.corpus_only:
        asyncio.run(run_corpus_only(args))
    else:
        asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
