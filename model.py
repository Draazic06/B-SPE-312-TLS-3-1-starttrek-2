#!/usr/bin/env python3
import sys

USAGE = """USAGE
    python model.py ALGO MODEL [EPISODES]

DESCRIPTION
    ALGO        algorithm of the saved model: dqn or ppo
    MODEL       path to the saved model file
    EPISODES    number of episodes to run (default: 1)"""


def error(message):
    print("Error:", message, file=sys.stderr)
    sys.exit(84)


def parse_args(args):
    if len(args) == 1 and args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)
    if len(args) not in (2, 3):
        error("wrong number of arguments (see python model.py -h)")

    algo, model_path = args[0], args[1]
    if algo not in ("dqn", "ppo"):
        error("ALGO must be dqn or ppo")

    episodes = 1
    if len(args) == 3:
        if not args[2].isdigit() or int(args[2]) == 0:
            error("EPISODES must be a positive integer")
        episodes = int(args[2])
    return algo, model_path, episodes


def load_dqn(model_path, state_dim, n_actions):
    import torch
    import torch.nn as nn

    # Même réseau que dans dqn.ipynb
    class QNetwork(nn.Module):
        def __init__(self, state_dim, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    q_net = QNetwork(state_dim, n_actions)
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()

    # Pas d'exploration : on joue toujours l'action qui a la plus grande Q-value
    def policy(state):
        with torch.no_grad():
            return int(q_net(torch.tensor(state, dtype=torch.float32)).argmax())

    return policy


def main():
    algo, model_path, episodes = parse_args(sys.argv[1:])
    if algo == "ppo":
        error("ppo models are not supported yet")

    import gymnasium as gym

    # La taille de l'état et le nombre d'actions sont lus depuis l'environnement
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    try:
        policy = load_dqn(model_path, state_dim, n_actions)
    except FileNotFoundError:
        error(f"model file '{model_path}' not found")
    except Exception:
        error(f"'{model_path}' is not a valid {algo} model")

    scores = []
    try:
        env = gym.make("LunarLander-v3", render_mode="human")
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            total_reward, done = 0.0, False
            while not done:
                state, reward, terminated, truncated, _ = env.step(policy(state))
                total_reward += reward
                done = terminated or truncated
            scores.append(total_reward)
            print(f"Episode {episode}: score = {total_reward:.1f}")
        env.close()
    except Exception as e:
        error(f"cannot run the episode: {e}")

    if episodes > 1:
        print(f"Mean score: {sum(scores) / len(scores):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
