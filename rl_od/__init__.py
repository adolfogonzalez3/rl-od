from gym.envs.registration import register


# Env registration
# ==========================

register(
    'ObjectDetection-v0',
    entry_point='rl_od.envs.rl_od:rl_od'
)
