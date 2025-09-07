#!/usr/bin/env python3
"""
Modified experiment runner that handles long-running experiments better
"""

import sys
import os
from datetime import datetime
import time
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Experiment timed out")

def run_single_experiment(experiment_name, config_name, timeout_minutes=10):
    """Run a single experiment with timeout"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {experiment_name}")
    print(f"CONFIG: {config_name}")
    print(f"TIMEOUT: {timeout_minutes} minutes")
    print('='*60)

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_minutes * 60)  # Convert to seconds

    try:
        # Import here to avoid issues
        from config import ConfigPresets
        from run_experiments import run_experiment

        # Get the config
        config_method = getattr(ConfigPresets, config_name)
        config = config_method()

        # Run the experiment
        start_time = time.time()
        result = run_experiment(config, experiment_name)
        end_time = time.time()

        print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
        return result

    except TimeoutError:
        print(f"⏰ TIMEOUT: {experiment_name} exceeded {timeout_minutes} minutes")
        return None
    except Exception as e:
        print(f"❌ ERROR in {experiment_name}: {e}")
        return None
    finally:
        signal.alarm(0)  # Cancel the alarm

def main():
    """Run experiments with better control"""
    print("🚀 CONTROLLED EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Define experiments with their config methods and timeouts
    experiments = [
        ("Basic Parser + Fast Training", "fast_training", 5),
        ("NLTK Parser + Balanced", "balanced", 8),
        ("NLTK Parser + High Accuracy", "high_accuracy", 10),
        ("spaCy Parser + Optimized", "spacy_optimized", 15),  # Give spaCy more time
        ("BERT Parser + Optimized", "bert_optimized", 20),
        ("RoBERTa Parser + Optimized", "roberta_optimized", 20),
        ("DistilBERT Parser + Fast", "distilbert_fast", 10),
        ("LayoutLMv2 Parser + Optimized", "layoutlmv2_optimized", 20),
        ("DONUT Parser + Optimized", "donut_optimized", 20),
    ]

    results = {}
    completed_experiments = 0

    for experiment_name, config_name, timeout in experiments:
        print(f"\n🎯 Starting experiment {completed_experiments + 1}/{len(experiments)}")
        result = run_single_experiment(experiment_name, config_name, timeout)

        if result:
            results[experiment_name] = result
            completed_experiments += 1
            print(f"✅ {experiment_name} completed successfully")
        else:
            print(f"⚠️  {experiment_name} failed or timed out")

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print('='*80)
    print(f"Total Experiments: {len(experiments)}")
    print(f"Completed: {completed_experiments}")
    print(f"Failed/Timeouts: {len(experiments) - completed_experiments}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if results:
        print(f"\n📊 RESULTS:")
        for experiment_name, result in results.items():
            print(f"{experiment_name}: {result['accuracy']:.2%}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiments_controlled_output_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write("CONTROLLED EXPERIMENT RUNNER RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(experiments)}\n")
        f.write(f"Completed: {completed_experiments}\n")
        f.write(f"Failed/Timeouts: {len(experiments) - completed_experiments}\n\n")

        if results:
            f.write("RESULTS:\n")
            for experiment_name, result in results.items():
                f.write(f"{experiment_name}: {result['accuracy']:.2%}\n")

    print(f"\n📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
