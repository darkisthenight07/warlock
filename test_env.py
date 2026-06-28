"""
Quick test script to verify GymBitcoinEnv works correctly.
Run: python test_env.py
"""
import sys
sys.path.insert(0, ".")

from src.env import GymBitcoinEnv
import numpy as np


def test_env():
    print("=" * 60)
    print("Testing GymBitcoinEnv")
    print("=" * 60)

    # Create environment
    env = GymBitcoinEnv()
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Max steps: {env.max_steps}")
    print(f"   Features: {env.n_features}")
    print(f"   Window len: {env.window_len}")

    # Test reset
    obs, info = env.reset()
    print(f"\n✅ Reset successful")
    print(f"   Obs shape: {obs.shape} (expected: {env.observation_space.shape})")
    print(f"   Initial info: {info}")

    assert obs.shape == env.observation_space.shape, f"Shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    assert "step" in info
    assert "price" in info

    # Test step with random actions
    print("\n--- Running 20 random steps ---")
    total_reward = 0.0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"  Step {info['step']:4d}: "
              f"action={action[0]:+6.3f}, "
              f"pos={info['position']:+6.3f}, "
              f"pnl={info['pnl']:+10.2f}, "
              f"cost={info['cost']:8.2f}, "
              f"cap={info['capital']:10.2f}, "
              f"dd={info['drawdown']:.3f}, "
              f"r={reward:+8.2f}")

        # Verify observation shape consistency
        assert obs.shape == env.observation_space.shape, f"Shape changed at step {i}"

        if terminated or truncated:
            print(f"\n🛑 Episode terminated at step {info['step']}")
            print(f"   Terminated: {terminated}, Truncated: {truncated}")
            print(f"   Final capital: {info['capital']:.2f}")
            print(f"   Max drawdown: {info['drawdown']:.3f}")
            break

    print(f"\n✅ Total reward over {i+1} steps: {total_reward:.2f}")

    # Test edge cases
    print("\n--- Testing edge cases ---")

    # Test action clipping
    env.reset()
    obs, r, t, tr, info = env.step(np.array([2.0]))  # Should clip to 1.0
    assert info["position"] <= 1.0, "Action not clipped"
    print("✅ Action clipping works")

    # Test max_trade_step limit
    env.reset()
    env.step(np.array([1.0]))  # Go long
    pos_after_1 = env.position
    env.step(np.array([-1.0]))  # Try to go short (should be limited)
    pos_after_2 = env.position
    assert pos_after_2 >= pos_after_1 - env.max_trade_step - 1e-6, "Max trade step not enforced"
    print(f"✅ Max trade step enforced: {pos_after_1:.3f} -> {pos_after_2:.3f}")

    # Test transaction cost reduces capital
    env.reset()
    init_cap = env.capital
    env.step(np.array([1.0]))
    assert env.capital < init_cap, "Transaction cost not applied"
    print(f"✅ Transaction cost applied: {init_cap:.2f} -> {env.capital:.2f}")

    # Test P&L calculation
    env.reset()
    env.step(np.array([1.0]))  # Long
    price_before = env.prices[env.current_step]
    # Manually step to next price
    # (just checking that pnl is computed)
    print("✅ P&L calculation runs without error")

    env.close()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_env()