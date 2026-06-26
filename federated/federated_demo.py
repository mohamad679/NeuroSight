import os
import sys
from collections import OrderedDict
from typing import Dict, Tuple

try:
    import flwr as fl
except ImportError as exc:
    raise SystemExit(
        "Flower is required: pip install flwr==1.8.0"
    ) from exc
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from neurosight.models.cognitive import CognitiveClassifier


SEED = 42
NUM_CLIENTS = 3
PATIENTS_PER_CLIENT = 20
NUM_FEATURES = 8
NUM_CLASSES = 6
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-2


rng = np.random.default_rng(SEED)
TRUE_WEIGHTS = rng.normal(0.0, 1.0, size=(NUM_FEATURES, NUM_CLASSES)).astype(np.float32)
TRUE_BIAS = rng.normal(0.0, 0.5, size=(NUM_CLASSES,)).astype(np.float32)
CRITERION = nn.CrossEntropyLoss()


def get_parameters(model: nn.Module):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict((k, torch.tensor(v)) for k, v in params_dict)
    model.load_state_dict(state_dict, strict=True)


def generate_hospital_data(site_id: int, n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    local_rng = np.random.default_rng(seed)
    site_shift = site_id * 0.35
    x = local_rng.normal(loc=site_shift, scale=1.0, size=(n_samples, NUM_FEATURES)).astype(np.float32)
    noise = local_rng.normal(loc=0.0, scale=0.7, size=(n_samples, NUM_CLASSES)).astype(np.float32)
    logits = x @ TRUE_WEIGHTS + TRUE_BIAS + noise
    y = logits.argmax(axis=1).astype(np.int64)
    return x, y


def split_train_val(x: np.ndarray, y: np.ndarray, train_fraction: float = 0.8):
    split_idx = int(len(x) * train_fraction)
    return (x[:split_idx], y[:split_idx]), (x[split_idx:], y[split_idx:])


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        features = torch.tensor(x, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        logits, _ = model(features)
        loss = CRITERION(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return float(loss), float(accuracy)


def make_client_dataset() -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    clients = {}
    for cid in range(NUM_CLIENTS):
        x, y = generate_hospital_data(cid, PATIENTS_PER_CLIENT, seed=SEED + cid + 1)
        train_data, val_data = split_train_val(x, y)
        clients[str(cid)] = {"train": train_data, "val": val_data}
    return clients


def make_heldout_set(n_samples: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    samples_per_site = n_samples // NUM_CLIENTS
    x_parts, y_parts = [], []
    for site_id in range(NUM_CLIENTS):
        x_site, y_site = generate_hospital_data(
            site_id, samples_per_site, seed=SEED + 100 + site_id
        )
        x_parts.append(x_site)
        y_parts.append(y_site)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


CLIENT_DATA = make_client_dataset()
HELDOUT_X, HELDOUT_Y = make_heldout_set()


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, cid: str, train_data, val_data):
        self.cid = cid
        self.model = CognitiveClassifier(num_classes=NUM_CLASSES)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        train_x, train_y = train_data
        val_x, val_y = val_data
        train_dataset = TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.long),
        )
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_x = val_x
        self.val_y = val_y

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        self.model.train()
        last_loss = 0.0
        for _ in range(local_epochs):
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = CRITERION(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.item())
        return get_parameters(self.model), len(self.train_loader.dataset), {"loss": last_loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.val_x, self.val_y)
        return loss, len(self.val_y), {"accuracy": accuracy}


def client_fn(cid: str):
    data = CLIENT_DATA[cid]
    return HospitalClient(cid, data["train"], data["val"]).to_client()


def evaluate_global(server_round: int, parameters, config):
    model = CognitiveClassifier(num_classes=NUM_CLASSES)
    set_parameters(model, parameters)
    loss, accuracy = evaluate_model(model, HELDOUT_X, HELDOUT_Y)
    if server_round >= 1:
        print(f"Round {server_round}/{NUM_ROUNDS} | Accuracy: {accuracy:.2f}")
    return loss, {"accuracy": accuracy}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda _: {"local_epochs": LOCAL_EPOCHS},
        evaluate_fn=evaluate_global,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    print("Federated training complete.")


if __name__ == "__main__":
    main()
