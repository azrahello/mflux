#!/usr/bin/env python3
"""
Test Patch 1: Vision Attention Optimization
Confronta performance prima e dopo l'ottimizzazione
"""

import time

import mlx.core as mx

from src.mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention
from src.mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention_optimized import (
    VisionAttention as VisionAttentionOptimized,
)


def benchmark_attention(attention_class, name, num_runs=10):
    """Benchmark attention module"""

    # Setup
    embed_dim = 1280
    num_heads = 16
    seq_len = 1024

    attention = attention_class(embed_dim=embed_dim, num_heads=num_heads)

    # Create dummy input
    x = mx.random.normal((seq_len, embed_dim))

    # Warmup
    print(f"\n🔥 Warmup {name}...")
    for _ in range(3):
        _ = attention(x)
        mx.eval(mx.array([0]))  # Sync GPU

    # Benchmark
    print(f"⏱️  Benchmarking {name}...")
    times = []

    for i in range(num_runs):
        mx.metal.clear_cache()
        start = time.time()

        output = attention(x)
        mx.eval(output)  # Force evaluation

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i + 1}/{num_runs}: {elapsed:.4f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {"name": name, "avg": avg_time, "min": min_time, "max": max_time, "runs": times}


def main():
    print("=" * 60)
    print("  Vision Attention Optimization Test")
    print("=" * 60)

    # Test original
    original_results = benchmark_attention(VisionAttention, "Original")

    # Test optimized
    optimized_results = benchmark_attention(VisionAttentionOptimized, "Optimized")

    # Results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print("\nOriginal:")
    print(f"  Average: {original_results['avg']:.4f}s")
    print(f"  Min:     {original_results['min']:.4f}s")
    print(f"  Max:     {original_results['max']:.4f}s")

    print("\nOptimized:")
    print(f"  Average: {optimized_results['avg']:.4f}s")
    print(f"  Min:     {optimized_results['min']:.4f}s")
    print(f"  Max:     {optimized_results['max']:.4f}s")

    # Speedup
    speedup = original_results["avg"] / optimized_results["avg"]
    improvement = ((original_results["avg"] - optimized_results["avg"]) / original_results["avg"]) * 100

    print(f"\n{'=' * 60}")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    print(f"{'=' * 60}")

    if speedup > 1.05:
        print("\n✅ OPTIMIZATION SUCCESSFUL!")
    elif speedup > 0.95:
        print("\n⚠️  Marginal improvement")
    else:
        print("\n❌ REGRESSION - optimized is slower!")


if __name__ == "__main__":
    main()
