import gymnasium as gym
from actor_critic_pytorch.core.mlp import MLP
import torch as T
import numpy as np
from typing import Union, Tuple, List


class ActorCritic:

    _env: gym.Env
    _actor: MLP
    _critic: MLP

    def __init__(
        self,
        env: gym.Env,
        actor_kwargs: Union[dict, None] = None,
        critic_kwargs: Union[dict, None] = None,
    ):

        self._env = env

        # Determine state dim
        if isinstance(env.observation_space, gym.spaces.Box):
            state_dim: int = env.observation_space.shape[0]
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            state_dim: int = env.observation_space.n
        else:
            raise NotImplementedError(
                f"Observation space {env.observation_space} not supported"
            )

        # Determine action dim
        if isinstance(env.action_space, gym.spaces.Box):
            action_dim: int = env.action_space.shape[0]
        elif isinstance(env.action_space, gym.spaces.Discrete):
            action_dim: int = env.action_space.n
        else:
            raise NotImplementedError(f"Action space {env.action_space} not supported")

        # Create actor network
        if actor_kwargs is None:
            self._actor = MLP(
                name="actor",
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_dims=[32, 32],
                dropout_probs=[0.0, 0.0],
                activation=T.nn.ReLU,
                optimizer=T.optim.Adam,
                optimizer_kwargs={"lr": 1e-3},
                lr_scheduler=T.optim.lr_scheduler.ExponentialLR,
                lr_scheduler_kwargs={"gamma": 0.99},
            )
        else:
            self._actor = MLP(**actor_kwargs)

        # Create critic network
        if critic_kwargs is None:
            self._critic = MLP(
                name="critic",
                input_dim=state_dim,
                output_dim=1,
                hidden_dims=[32, 32],
                dropout_probs=[0.0, 0.0],
                activation=T.nn.ReLU,
                optimizer=T.optim.Adam,
                optimizer_kwargs={"lr": 1e-3},
                lr_scheduler=T.optim.lr_scheduler.ExponentialLR,
                lr_scheduler_kwargs={"gamma": 0.99},
            )
        else:
            self._critic = MLP(**critic_kwargs)

    def _wrap_to_tensors(
        self, *args: Tuple[np.ndarray, T.dtype]
    ) -> Tuple[T.Tensor, ...]:
        return tuple(T.tensor(arr, dtype=dtype) for arr, dtype in args)

    def _learn(
        self,
        state: T.Tensor,
        action: T.Tensor,
        reward: T.Tensor,
        next_state: T.Tensor,
        done: T.Tensor,
    ):
        raise NotImplementedError("Learn method not implemented")

    def act(self, state: T.Tensor) -> T.Tensor:
        return self._actor(state)

    def _run_episode(self, train: bool = True) -> Tuple[float, int]:
        total_reward: float = 0.0
        state, _ = self.env.reset()
        state = self._wrap_to_tensors((state, T.float32))[0]
        done: bool = False
        steps: int = 0
        while not done:
            action: np.ndarray = self.act(state).numpy()
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated

            if train:
                self._learn(
                    *self._wrap_to_tensors(
                        (state, T.float32),
                        (action, T.float32),
                        (np.array(reward), T.float32),
                        (next_state, T.float32),
                        (np.array(done), T.bool),
                    )
                )

            total_reward += reward
            state = next_state
            steps += 1
            done = terminated or truncated
        return total_reward, steps

    def train(
        self,
        num_episodes: Union[int, None] = None,
        max_timesteps: Union[int, None] = None,
        checkpoint_freq: Union[int, None] = None,
        train_freq: int = 1,
        eval_freq: Union[int, None] = None,
    ):
        assert (
            num_episodes is not None or max_timesteps is not None
        ), "Either num_episodes or max_timesteps must be provided"
        if checkpoint_freq is None:
            print("Checkpoint frequency not provided, no checkpoints will be saved")
        if eval_freq is None:
            print("Evaluation frequency not provided, no evaluations will be done")
        assert train_freq > 0, "train_freq must be a positive integer"

        if num_episodes is not None:
            for episode in range(num_episodes):
                total_reward = self._run_episode()
                # print(f"Episode {episode + 1} | Total Reward: {total_reward}")
                # if checkpoint_freq is not None and (episode + 1) % checkpoint_freq == 0:
                #     self._actor.save_checkpoint()
                #     self._critic.save_checkpoint()
                # if eval_freq is not None and (episode + 1) % eval_freq == 0:
                #     self.evaluate()

        if max_timesteps is not None:
            steps: int = 0
            state, _ = self.env.reset()
            while steps < max_timesteps:
                result: Tuple[float, int] = self._run_episode()
                total_reward: float = result[0]
                steps = steps + result[1]
