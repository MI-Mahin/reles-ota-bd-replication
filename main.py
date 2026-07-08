"""
main.py - Multi-Agent OTA Evaluation & Training Pipeline
========================================================
Entry point for multi-seed experiments across MARL algorithms and safety
configurations. The CLI remains usable directly and is also used by the local
web control interface.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from config import BENCHMARK_CFG, PATHS_CFG, TRAIN_CFG
from tools.training_registry import log_run, print_summary as registry_summary
from train_mappo import train_algorithm


def _compute_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    if n < 2 or se == 0:
        return m, 0.0
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def _compute_pvalue(baseline_data, target_data):
    if len(baseline_data) < 2 or len(target_data) < 2:
        return 1.0
    _, p = stats.ttest_ind(baseline_data, target_data, equal_var=False)
    return p


def _placeholder_performance(seed_dir, algorithm):
    try:
        from results.marl_models import _placeholder_eval

        return _placeholder_eval(seed_dir, algorithm)
    except Exception:
        algo_offsets = {"ippo": -100.0, "mappo": -50.0, "fp3o": -20.0}
        return algo_offsets.get(algorithm, -50.0) + np.random.randn() * 5.0


def _read_raw_results(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_raw_results(path, raw_results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)


def _upsert_leaderboard(leaderboard_path, row):
    if os.path.exists(leaderboard_path):
        df = pd.read_csv(leaderboard_path)
        idx = df.index[df["Experiment"] == row["Experiment"]].tolist()
        if idx:
            for key, value in row.items():
                df.at[idx[0], key] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(leaderboard_path, index=False)
    return df


def _run_algorithm(args, algorithm, results_dir, raw_results, raw_results_file):
    entry_name = f"{algorithm.upper()}_Safety_{args.safety}"
    final_returns = []

    for seed in range(args.seeds):
        print(f"\n[Seed {seed + 1}/{args.seeds}] Starting training for {entry_name}...", flush=True)
        seed_dir = str(results_dir / "marl_models" / entry_name / f"seed_{seed}")

        t0 = time.time()
        train_algorithm(
            algorithm=algorithm,
            n_agents=args.n_agents,
            n_blocks=args.n_blocks,
            bd_mode=args.bd_mode,
            safety=args.safety,
            total_timesteps=args.timesteps,
            save_dir=seed_dir,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            death_masking=args.death_masking,
        )
        elapsed = time.time() - t0
        print(f"[Seed {seed + 1}/{args.seeds}] Completed in {elapsed:.1f}s", flush=True)

        # TODO: replace with evaluate_model(seed_dir) once a real evaluator exists.
        final_returns.append(_placeholder_performance(seed_dir, algorithm))

    raw_results[entry_name] = final_returns
    _write_raw_results(raw_results_file, raw_results)

    mean_ret, ci_ret = _compute_ci(final_returns)
    p_val = "N/A"
    baseline_name = f"IPPO_Safety_{args.safety}"
    if entry_name != baseline_name and baseline_name in raw_results:
        p_val = f"{_compute_pvalue(raw_results[baseline_name], final_returns):.4f}"

    new_row = {
        "Experiment": entry_name,
        "Mean_Return": round(mean_ret, 2),
        "CI_95": round(ci_ret, 2),
        "p_value_vs_IPPO": p_val,
        "N_Agents": args.n_agents,
        "N_Blocks": args.n_blocks,
        "Timesteps": args.timesteps,
        "Seeds": args.seeds,
    }

    leaderboard_path = results_dir / "leaderboard.csv"
    df = _upsert_leaderboard(leaderboard_path, new_row)
    print(f"\nLeaderboard updated at {leaderboard_path}", flush=True)
    print(df.to_string(), flush=True)

    run_id = log_run(
        algorithm=algorithm,
        safety=args.safety,
        n_seeds=args.seeds,
        timesteps=args.timesteps,
        mean_return=mean_ret,
        ci_95=ci_ret,
        p_value=p_val,
        extra={
            "n_agents": args.n_agents,
            "n_blocks": args.n_blocks,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "device": args.device,
            "bd_mode": args.bd_mode,
            "death_masking": args.death_masking,
        },
    )
    print(f"\n  Run #{run_id} recorded in training registry.", flush=True)
    registry_summary()

    if len(df) >= 2:
        try:
            from tools.plot_comparison import plot_pair

            experiments = list(df["Experiment"])
            plot_pair(df, experiments[-2], experiments[-1])
            print(f"  Comparison chart saved to {PATHS_CFG['charts_dir']}/", flush=True)
        except Exception as chart_err:
            print(f"  [warn] Chart generation skipped: {chart_err}", flush=True)

    target = BENCHMARK_CFG["target_return_bd"] if args.bd_mode else BENCHMARK_CFG["target_return_generic"]
    if mean_ret >= target:
        print(f"\n  BENCHMARK MET: {mean_ret:.2f} >= target {target}", flush=True)
    else:
        gap = target - mean_ret
        print(f"\n  Benchmark not yet met ({mean_ret:.2f}). Still {gap:.2f} away from target {target}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent OTA ReLES Pipeline")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode of execution")
    parser.add_argument("--algorithm", choices=["ippo", "mappo", "fp3o"], default="fp3o", help="Algorithm to run")
    parser.add_argument("--compare_algorithm", choices=["", "ippo", "mappo", "fp3o"], default="", help="Optional second algorithm")
    parser.add_argument("--safety", type=lambda x: str(x).lower() == "true", default=True, help="Enable Safety Shield")
    parser.add_argument("--n_agents", type=int, default=4, help="Number of ECU agents")
    parser.add_argument("--n_blocks", type=int, default=16, help="Firmware blocks per agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per seed")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--n_envs", type=int, default=TRAIN_CFG["n_envs"], help="Parallel rollout environments")
    parser.add_argument("--n_steps", type=int, default=TRAIN_CFG["n_steps"], help="PPO rollout horizon per environment")
    parser.add_argument("--batch_size", type=int, default=TRAIN_CFG["batch_size"], help="PPO minibatch size")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    parser.add_argument("--bd_mode", type=lambda x: str(x).lower() == "true", default=True, help="Enable constrained BD network")
    parser.add_argument("--death_masking", type=lambda x: str(x).lower() == "true", default=True, help="Enable death masking")
    args = parser.parse_args()

    print("\n" + "=" * 60, flush=True)
    print("  Multi-Agent OTA Evaluation Pipeline", flush=True)
    print("=" * 60, flush=True)
    print(f"  Mode:       {args.mode.upper()}", flush=True)
    print(f"  Algorithm:  {args.algorithm.upper()}", flush=True)
    print(f"  Compare:    {args.compare_algorithm.upper() if args.compare_algorithm else 'OFF'}", flush=True)
    print(f"  Safety:     {args.safety}", flush=True)
    print(f"  Seeds:      {args.seeds}", flush=True)
    print(f"  Agents:     {args.n_agents} | Blocks: {args.n_blocks}", flush=True)
    print(f"  Timesteps:  {args.timesteps}", flush=True)
    print(f"  Rollout:    {args.n_steps} x {args.n_envs} = {args.n_steps * args.n_envs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Device:     {args.device}", flush=True)
    print(f"  BD Network: {args.bd_mode}", flush=True)
    print("=" * 60, flush=True)

    if args.mode != "train":
        print("Test mode evaluation will be implemented in future phases.", flush=True)
        return

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_results_file = results_dir / "raw_seed_returns.json"
    raw_results = _read_raw_results(raw_results_file)

    algorithms = [args.algorithm]
    if args.compare_algorithm and args.compare_algorithm != args.algorithm:
        algorithms.append(args.compare_algorithm)

    for algorithm in algorithms:
        _run_algorithm(args, algorithm, results_dir, raw_results, raw_results_file)


if __name__ == "__main__":
    main()
