from gymnasium.envs.registration import register

register(
    id='oneMSL-v0',
    entry_point='guidance.environments.singleagent.oneTGT_oneMSL:oneMSL'
)