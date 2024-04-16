import gymnasium as gym
from mlp import MLP
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from typing import Union, Tuple, List
from tqdm import tqdm


class TDActorCriticWeightSharing:
    """
    Basic (TD) Actor-Critic implementation, with a single network for the actor and critic.
    This can be viewed as A2C, as we approximate the advantage function with the TD error.
    Updates are performed at every time step.
    The critic loss is the TD error and the actor loss is the policy gradient weighted by the TD error.
    """

    _env: gym.Env
    _env_has_continuous_states: bool
    _env_has_continuous_actions: bool
    _state_dim: int
    _action_dim: int
    _n_actions: int
    _writer: Union[None, SummaryWriter]
    _device: torch.device
    _actor_critic: MLP
    _actor_optimizer: torch.optim.Optimizer
    _critic_optimizer: torch.optim.Optimizer

    def __init__(self, env: gym.Env, writer: Union[None, SummaryWriter] = None):
        self._env = env
        if isinstance(env.observation_space, gym.spaces.Box):
            self._env_has_continuous_states = True
            self._state_dim = env.observation_space.shape[0]
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            self._env_has_continuous_states = False
            self._state_dim = env.observation_space.nvec.shape[0]
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            self._env_has_continuous_states = False
            self._state_dim = 1
        else:
            raise ValueError(
                f"Observation space {type(env.observation_space)} not supported"
            )

        if isinstance(env.action_space, gym.spaces.Discrete):
            self._env_has_continuous_actions = False
            self._action_dim = 1
            self._n_actions = env.action_space.n
        else:
            self._env_has_continuous_actions = True
            raise ValueError(f"Action space {type(env.action_space)} not supported")

        self._writer = writer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_nets()
        self._setup_optimizers()

    def _build_nets(self):
        self._actor_critic = MLP(
            self._state_dim,
            128,
            2,
            self._n_actions + 1,
            activation="relu",
            device=self._device,
        )

    def _setup_optimizers(self):
        self._actor_critic_optimizer = torch.optim.Adam(
            self._actor_critic.parameters(), lr=1e-3
        )

    @torch.no_grad()
    def act(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self._device)
        return torch.distributions.Categorical(
            logits=self._actor_critic(state)[: self._n_actions]
        ).sample()

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> tuple[float, float]:
        state_tensor: torch.Tensor = torch.from_numpy(state).float().to(self._device)
        action_tensor: torch.Tensor = torch.from_numpy(action).float().to(self._device)
        reward_tensor: torch.Tensor = torch.tensor(reward).float().to(self._device)
        next_state_tensor: torch.Tensor = (
            torch.from_numpy(next_state).float().to(self._device)
        )
        done_tensor: torch.Tensor = torch.tensor(done).float().to(self._device)

        # Compute the TD target
        next_value: torch.Tensor = self._actor_critic(next_state_tensor)[-1]
        target: torch.Tensor = reward_tensor + next_value * (1.0 - done_tensor)

        # Compute the TD error
        value: torch.Tensor = self._actor_critic(state_tensor)[-1]
        td_error: torch.Tensor = target - value

        # Compute the policy gradient
        log_prob: torch.Tensor = torch.distributions.Categorical(
            logits=self._actor_critic(state_tensor)[: self._n_actions]
        ).log_prob(action_tensor)
        actor_loss: torch.Tensor = -log_prob * td_error.detach()

        # Compute the value function loss
        critic_loss: torch.Tensor = (
            0.5 * td_error**2
        )  # multiply by 0.5 to avoid critic loss overwhelming actor loss

        total_loss = actor_loss + critic_loss

        # Update the actor critic
        self._actor_critic_optimizer.zero_grad()
        total_loss.backward()
        self._actor_critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self, n_episodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_rewards: List[float] = []
        mean_actor_losses: List[float] = []
        mean_critic_losses: List[float] = []
        print(f"Training for {n_episodes} episodes")
        for episode in tqdm(range(n_episodes)):
            state, _ = self._env.reset()
            done: bool = False
            total_reward: float = 0.0
            actor_losses: List[float] = []
            critic_losses: List[float] = []
            while not done:
                action = self.act(state).cpu().numpy()
                next_state, reward, terminated, truncated, _ = self._env.step(action)
                actor_loss, critic_loss = self.learn(
                    state, action, reward, next_state, done
                )
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                total_reward += reward
                state = next_state
                done = terminated or truncated

            total_rewards.append(total_reward)
            mean_actor_losses.append(float(np.mean(actor_losses)))
            mean_critic_losses.append(float(np.mean(critic_losses)))

            if self._writer is not None:
                self._writer.add_scalar("train/actor_loss", actor_losses[-1], episode)
                self._writer.add_scalar("train/critic_loss", critic_losses[-1], episode)
                self._writer.add_scalar("train/reward", total_reward, episode)

        return (
            np.array(total_rewards),
            np.array(mean_actor_losses),
            np.array(mean_critic_losses),
        )

    def evaluate(self, n_episodes: int) -> np.ndarray:
        total_rewards: List[float] = []
        print(f"Evaluating for {n_episodes} episodes")
        for episode in tqdm(range(n_episodes)):
            state, _ = self._env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.act(state).cpu().numpy()
                next_state, reward, terminated, truncated, _ = self._env.step(action)
                total_reward += reward
                state = next_state
                done = terminated or truncated
            total_rewards.append(total_reward)

            if self._writer is not None:
                self._writer.add_scalar("eval/reward", total_reward, episode)

        return np.array(total_rewards)


env = gym.make("CartPole-v1")
writer = SummaryWriter("logs/cartpole")
ac = TDActorCriticWeightSharing(env, writer)
ac.train(1000)
env = gym.make("CartPole-v1")
ac._env = env
ac.evaluate(10)
