import torch as T

from typing import Union, List, Type


class MLP(T.nn.Module):

    name: str
    network: T.nn.Module
    optimizer: T.optim.Optimizer
    lr_scheduler: Union[T.optim.lr_scheduler._LRScheduler, None]
    lr_scheduler_kwargs: Union[dict, None]

    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[List[int], None] = None,
        dropout_probs: Union[List[float], None] = None,
        activation: Type[T.nn.Module] = T.nn.ReLU,
        optimizer: Type[T.optim.Optimizer] = T.optim.Adam,
        optimizer_kwargs: Union[dict, None] = None,
        lr: float = 1e-3,
        lr_scheduler: Union[Type[T.optim.lr_scheduler._LRScheduler], None] = None,
        lr_scheduler_kwargs: Union[dict, None] = None,
    ):
        assert input_dim > 0, "input_dim must be a positive integer"
        assert output_dim > 0, "output_dim must be a positive integer"
        assert (
            hidden_dims is None or len(hidden_dims) > 0
        ), "hidden_dims must be a list of positive integers"
        if not hidden_dims:
            hidden_dims = []
        assert dropout_probs is None or len(dropout_probs) == len(
            hidden_dims
        ), "dropout_probs must be a list of floats with the same length as hidden_dims"
        if not dropout_probs:
            dropout_probs = [0.0] * len(hidden_dims)
        assert activation in [
            T.nn.ReLU,
            T.nn.Tanh,
            T.nn.Sigmoid,
        ], "activation must be one of [T.nn.ReLU, T.nn.Tanh, T.nn.Sigmoid]"
        assert optimizer in [
            T.optim.Adam,
            T.optim.SGD,
        ], "optimizer must be one of [T.optim.Adam, T.optim.SGD]"
        assert lr > 0.0, "lr must be a positive float"
        if lr_scheduler is not None:
            assert (
                lr_scheduler_kwargs is not None
            ), "lr_scheduler_kwargs must be a dictionary"
            assert lr_scheduler in [
                T.optim.lr_scheduler.ExponentialLR,
                T.optim.lr_scheduler.StepLR,
            ], "lr_scheduler must be one of [T.optim.lr_scheduler.ExponentialLR, T.optim.lr_scheduler.StepLR]"

        super(MLP, self).__init__()

        self.name = name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.network = self.create_network()
        self.optimizer = optimizer(self.parameters(), **self.optimizer_kwargs)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer, **self.lr_scheduler_kwargs)
        else:
            self.lr_scheduler = None

    def create_network(self) -> T.nn.Module:
        layers: List[T.nn.Module] = []
        in_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(T.nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation())
            layers.append(T.nn.Dropout(p=self.dropout_probs[i]))
            in_dim = hidden_dim
        layers.append(T.nn.Linear(in_dim, self.output_dim))
        return T.nn.Sequential(*layers)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.network(x)

    def train_network(
        self,
        x: T.Tensor,
        y: T.Tensor,
        epochs: int,
        loss_fn: T.nn.Module = T.nn.MSELoss(),
    ):
        for _ in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def eval_network(
        self, x: T.Tensor, y: T.Tensor, loss_fn: T.nn.Module = T.nn.MSELoss()
    ) -> float:
        with T.no_grad():
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            return loss.item()

    def __str__(self) -> str:
        return f"{self.name}\n{str(self.network)}"
