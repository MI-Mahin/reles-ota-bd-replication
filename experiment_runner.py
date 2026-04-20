import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ota_env import OTAEnv
from stable_baselines3 import PPO
from baselines import BaselineRunner

Path("results/final").mkdir(parents=True, exist_ok=True)

def evaluate_model(model, env, num_episodes=20):
    payloads = []
    memories = []
    rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
        
        payloads.append(info.get('payload_bytes', 0))
        memories.append(info.get('memory_used', 0))
        rewards.append(ep_reward)
    
    return {
        'mean_payload': np.mean(payloads),
        'std_payload': np.std(payloads),
        'mean_memory': np.mean(memories),
        'mean_reward': np.mean(rewards)
    }

# ====================== MAIN EXPERIMENT ======================
if __name__ == "__main__":
    print(" Starting Generic vs BD Experiment (Checkpoint 5)...\n")
    
    # Load trained models (use best if available)
    try:
        model_generic = PPO.load("results/models/generic_best/best_model.zip")
        print(" Loaded best Generic model")
    except:
        model_generic = PPO.load("results/models/ppo_generic_final.zip")
        print(" Loaded final Generic model")
    
    # === Run Generic Evaluation ===
    print("Evaluating Generic model...")
    env_generic = OTAEnv(n_blocks=24, bd_mode=False)
    generic_rl = evaluate_model(model_generic, env_generic, num_episodes=30)
    
    # === Run BD Evaluation ===
    print("Evaluating BD model...")
    env_bd = OTAEnv(n_blocks=24, bd_mode=True)
    try:
        model_bd = PPO.load("results/models/bd_best/best_model.zip")
    except:
        print(" No BD model yet. Training a quick one or using Generic for comparison.")
        model_bd = model_generic  # fallback
    
    bd_rl = evaluate_model(model_bd, env_bd, num_episodes=30)
    
    # === Baselines (for reference) ===
    baseline_runner = BaselineRunner(n_blocks=24)
    generic_baseline = baseline_runner.run_random_baseline(20)
    
    # ====================== RESULTS TABLE ======================
    results = {
        'Setting': ['Generic RL', 'BD RL', 'Random Baseline'],
        'Mean Payload': [generic_rl['mean_payload'], bd_rl['mean_payload'], generic_baseline['mean_payload']],
        'Payload Std': [generic_rl['std_payload'], bd_rl['std_payload'], generic_baseline['std_payload']],
        'Mean Memory': [generic_rl['mean_memory'], bd_rl['mean_memory'], generic_baseline['mean_memory']],
        'Mean Reward': [generic_rl['mean_reward'], bd_rl['mean_reward'], 'N/A']
    }
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL RESULTS: Generic vs Bangladesh Conditions")
    print("="*60)
    print(df.round(2))
    
    # Save table
    df.to_csv("results/final/results_comparison.csv", index=False)
    print("\n Results saved to results/final/results_comparison.csv")
    
    # ====================== PLOTS ======================
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Generic RL', 'BD RL', 'Random'], 
            [generic_rl['mean_payload'], bd_rl['mean_payload'], generic_baseline['mean_payload']],
            color=['skyblue', 'orange', 'gray'])
    plt.ylabel('Mean Payload Cost')
    plt.title('Payload Size Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(['Generic RL', 'BD RL', 'Random'], 
            [generic_rl['mean_memory'], bd_rl['mean_memory'], generic_baseline['mean_memory']],
            color=['skyblue', 'orange', 'gray'])
    plt.ylabel('Mean Memory Overhead')
    plt.title('Memory Usage Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/final/final_comparison.png', dpi=300)
    plt.show()
    
    print(" Plot saved as results/final/final_comparison.png")
    print("\n Checkpoint 5 Completed!")




