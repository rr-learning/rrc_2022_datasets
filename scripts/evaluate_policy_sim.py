import argparse
import gym
import json

from rrc_2022_datasets import Evaluation, PolicyBase


class RandomPolicy(PolicyBase):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate policy in TriFinger environment."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="trifinger-cube-push-sim-expert-v0",
        help="Name of the gym environment to load.",
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Enable visualization of environment.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=64,
        help="Number of episodes to run.",
    )
    args = parser.parse_args()

    env = gym.make(
        args.env_name,
        disable_env_checker=True,
        # enable visualization,
        visualization=args.visualization,
    )
    random_policy = RandomPolicy(env.action_space)

    evaluation = Evaluation(env)
    eval_res = evaluation.evaluate(
        policy=random_policy,
        n_episodes=args.n_episodes,
    )
    print("Evaluation result: ")
    print(json.dumps(eval_res, indent=4))
