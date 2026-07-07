import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl_ota_env import MultiAgentOTAEnv
import numpy as np

def test_edge_case():
    print("Testing Edge Case: 250ms Monsoon Network Jitter")
    
    # Create env with monsoon BD mode
    env = MultiAgentOTAEnv(n_agents=4, n_blocks=8, bd_mode=True, stochastic_latency=True)
    obs, info = env.reset(seed=1337)
    
    # Force the monsoon multiplier to be high
    env.bd_params["monsoon_multiplier"] = 2.5 # simulates severe jitter
    
    steps = 0
    while env.agents and steps < 10:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        steps += 1
        
        for a in env.possible_agents:
            if not term[a] and not trunc[a]:
                # We can check payload bytes scaling
                pass
                
    print(f"Edge case monsoon simulation completed for {steps} steps.")

if __name__ == "__main__":
    test_edge_case()
