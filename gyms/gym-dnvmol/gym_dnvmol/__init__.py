from gym.envs.registration import register

register(
    id='dnvmol-v0',
    entry_point='gyms.gym_dnvmol.envs:DeNovoMolEnv',
)
