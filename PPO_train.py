from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from REMUSAUVEnv import REMUSAUVEnv


def make_env(seed: int = 0):
    def _init():
        return Monitor(REMUSAUVEnv(seed=seed, current_enabled=True, include_current_in_obs=True))

    return _init


if __name__ == "__main__":
    raw_env = REMUSAUVEnv(seed=0, current_enabled=True, include_current_in_obs=True)
    check_env(raw_env, warn=True)
    raw_env.close()

    vec_env = DummyVecEnv([make_env(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log="./tb_logs/remus_current",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints",
        name_prefix="ppo_remus_current",
    )

    model.learn(total_timesteps=500_000, callback=checkpoint_callback)
    model.save("ppo_remus_current_model")
    vec_env.save("ppo_remus_current_vecnormalize.pkl")
    vec_env.close()
