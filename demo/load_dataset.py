import gym

import rrc_2022_datasets  # noqa


if __name__ == "__main__":
    env = gym.make(
        "trifinger-cube-push-sim-expert-v0",
        disable_env_checker=True,
        visualization=True,  # enable visualization
    )

    dataset = env.get_dataset()

    n_transitions = len(dataset["observations"])
    print("Number of transitions: ", n_transitions)

    assert dataset["actions"].shape[0] == n_transitions
    assert dataset["rewards"].shape[0] == n_transitions

    print("First observation: ", dataset["observations"][0])

    obs = env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
