from __future__ import annotations

from datetime import UTC, datetime
from math import inf
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .price_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    _prepare_record_without_price,
    _regression_metrics,
    prepare_price_training_records,
)


def train_torch_price_model(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    cv_folds: int = 5,
    search_iterations: int = 16,
) -> dict[str, Any]:
    del cv_folds

    prepared = prepare_price_training_records(records)
    if len(prepared) < 20:
        raise ValueError("Need at least 20 healthy records to train the torch price model")

    dataset = pd.DataFrame(prepared)
    x = dataset[FEATURE_COLUMNS]
    y = dataset["price_pln"]

    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=random_seed,
    )

    preprocessor = fit_torch_feature_preprocessor(x_train)
    x_train_matrix = transform_torch_feature_matrix(x_train, preprocessor)
    x_validation_matrix = transform_torch_feature_matrix(x_validation, preprocessor)
    y_train_array = y_train.to_numpy(dtype=np.float32)
    y_validation_array = y_validation.to_numpy(dtype=np.float32)

    candidate_results = []
    for candidate_config in _candidate_configs(search_iterations):
        candidate_results.append(
            _train_torch_candidate(
                x_train_matrix,
                y_train_array,
                x_validation_matrix,
                y_validation_array,
                candidate_config=candidate_config,
                random_seed=random_seed,
            )
        )

    best_candidate = min(
        candidate_results,
        key=lambda item: item["validation_metrics"]["mae_pln"],
    )

    return {
        "version": 4,
        "backend": "torch",
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "candidate_scores": [
            {
                "candidate_name": item["candidate_name"],
                "cv_mae_pln": item["validation_metrics"]["mae_pln"],
            }
            for item in sorted(candidate_results, key=lambda item: item["validation_metrics"]["mae_pln"])
        ],
        "best_candidate_name": best_candidate["candidate_name"],
        "best_params": best_candidate["candidate_config"],
        "training_size": len(x_train),
        "validation_size": len(x_validation),
        "training_metrics": best_candidate["training_metrics"],
        "validation_metrics": best_candidate["validation_metrics"],
        "feature_importances": [],
        "estimator": {
            "model_type": "torch_mlp",
            "model_config": {
                "input_dim": int(x_train_matrix.shape[1]),
                "hidden_layers": best_candidate["candidate_config"]["hidden_layers"],
                "dropout": best_candidate["candidate_config"]["dropout"],
            },
            "preprocessor": preprocessor,
            "state_dict": best_candidate["state_dict"],
        },
    }


def predict_torch_price(record: dict[str, Any], model_bundle: dict[str, Any]) -> int | None:
    estimator = model_bundle.get("estimator")
    if not isinstance(estimator, dict):
        return None

    prepared = prepare_price_training_records([record])
    if not prepared:
        raw_record = _prepare_record_without_price(record)
        if raw_record is None:
            return None
        prepared = [raw_record]

    frame = pd.DataFrame(prepared)[FEATURE_COLUMNS]
    matrix = transform_torch_feature_matrix(frame, estimator["preprocessor"])
    if matrix.shape[0] == 0:
        return None

    torch = _require_torch()
    model = _get_runtime_model(model_bundle)
    with torch.inference_mode():
        prediction = model(torch.tensor(matrix, dtype=torch.float32)).squeeze(1).cpu().numpy()[0]
    return max(0, int(round(float(prediction))))


def fit_torch_feature_preprocessor(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_frame = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric_medians: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}

    for column in NUMERIC_FEATURES:
        series = numeric_frame[column]
        numeric_medians[column] = float(series.median()) if series.notna().any() else 0.0

    numeric_filled = numeric_frame.fillna(value=numeric_medians).astype(float)
    for column in NUMERIC_FEATURES:
        series = numeric_filled[column]
        numeric_means[column] = float(series.mean()) if len(series) else 0.0
        std_value = float(series.std(ddof=0)) if len(series) else 1.0
        numeric_stds[column] = std_value if std_value > 0.0 else 1.0

    categorical_frame = _normalize_categorical_frame(frame)
    dummy_columns = pd.get_dummies(categorical_frame, prefix=CATEGORICAL_FEATURES, prefix_sep="=").columns.tolist()

    return {
        "numeric_medians": numeric_medians,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "dummy_columns": dummy_columns,
    }


def transform_torch_feature_matrix(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> np.ndarray:
    numeric_frame = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric_filled = numeric_frame.fillna(value=preprocessor["numeric_medians"]).astype(float)
    numeric_scaled = numeric_filled.copy()
    for column in NUMERIC_FEATURES:
        mean_value = float(preprocessor["numeric_means"][column])
        std_value = float(preprocessor["numeric_stds"][column]) or 1.0
        numeric_scaled[column] = (numeric_scaled[column] - mean_value) / std_value

    categorical_frame = _normalize_categorical_frame(frame)
    dummy_frame = pd.get_dummies(categorical_frame, prefix=CATEGORICAL_FEATURES, prefix_sep="=")
    dummy_frame = dummy_frame.reindex(columns=preprocessor["dummy_columns"], fill_value=0)

    return np.concatenate(
        [
            numeric_scaled.to_numpy(dtype=np.float32, copy=False),
            dummy_frame.to_numpy(dtype=np.float32, copy=False),
        ],
        axis=1,
    )


class _TorchPriceRegressor:
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        *,
        dropout: float,
        torch: Any,
    ) -> None:
        self._torch = torch
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, 1))
        self.model = torch.nn.Sequential(*layers)

    def __call__(self, inputs: Any) -> Any:
        return self.model(inputs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)


def _candidate_configs(search_iterations: int) -> list[dict[str, Any]]:
    configs = [
        {
            "name": "torch_mlp_small",
            "hidden_layers": [64],
            "dropout": 0.0,
            "learning_rate": 0.005,
            "weight_decay": 0.0,
            "epochs": 220,
            "batch_size": 8,
            "patience": 24,
        },
        {
            "name": "torch_mlp_medium",
            "hidden_layers": [96, 48],
            "dropout": 0.05,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "epochs": 260,
            "batch_size": 12,
            "patience": 28,
        },
        {
            "name": "torch_mlp_wide",
            "hidden_layers": [128, 64],
            "dropout": 0.08,
            "learning_rate": 0.002,
            "weight_decay": 0.0005,
            "epochs": 320,
            "batch_size": 16,
            "patience": 32,
        },
        {
            "name": "torch_mlp_deep",
            "hidden_layers": [128, 64, 32],
            "dropout": 0.1,
            "learning_rate": 0.0015,
            "weight_decay": 0.0005,
            "epochs": 360,
            "batch_size": 16,
            "patience": 36,
        },
    ]
    limit = max(1, min(search_iterations, len(configs)))
    return configs[:limit]


def _train_torch_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    candidate_config: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    torch = _require_torch()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = _TorchPriceRegressor(
        int(x_train.shape[1]),
        candidate_config["hidden_layers"],
        dropout=float(candidate_config["dropout"]),
        torch=torch,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate_config["learning_rate"]),
        weight_decay=float(candidate_config["weight_decay"]),
    )
    loss_fn = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    batch_size = max(1, min(int(candidate_config["batch_size"]), len(dataset)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_state: dict[str, np.ndarray] | None = None
    best_validation_loss = inf
    epochs_without_improvement = 0

    for _ in range(int(candidate_config["epochs"])):
        model.train()
        for batch_inputs, batch_targets in data_loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

        validation_loss = _evaluate_mse(model, x_validation, y_validation, torch)
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_state = _serialize_state_dict(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(candidate_config["patience"]):
                break

    if best_state is None:
        best_state = _serialize_state_dict(model.state_dict())

    _load_serialized_state_dict(model, best_state, torch)
    training_predictions = _predict_matrix(model, x_train, torch)
    validation_predictions = _predict_matrix(model, x_validation, torch)

    return {
        "candidate_name": candidate_config["name"],
        "candidate_config": {
            key: value
            for key, value in candidate_config.items()
            if key != "name"
        },
        "training_metrics": _regression_metrics(y_train.astype(float).tolist(), training_predictions.tolist()),
        "validation_metrics": _regression_metrics(y_validation.astype(float).tolist(), validation_predictions.tolist()),
        "state_dict": best_state,
    }


def _predict_matrix(model: _TorchPriceRegressor, matrix: np.ndarray, torch: Any) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.tensor(matrix, dtype=torch.float32)).squeeze(1).cpu().numpy()
    return predictions.astype(np.float32, copy=False)


def _evaluate_mse(
    model: _TorchPriceRegressor,
    matrix: np.ndarray,
    targets: np.ndarray,
    torch: Any,
) -> float:
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.tensor(matrix, dtype=torch.float32)).squeeze(1)
        expected = torch.tensor(targets, dtype=torch.float32)
        mse = torch.nn.functional.mse_loss(prediction, expected)
    return float(mse.cpu().item())


def _normalize_categorical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    categorical = frame[CATEGORICAL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        categorical[column] = categorical[column].map(_normalize_categorical_value)
    return categorical


def _normalize_categorical_value(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "missing"


def _serialize_state_dict(state_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in state_dict.items()
    }


def _load_serialized_state_dict(
    model: _TorchPriceRegressor,
    state_dict: dict[str, np.ndarray],
    torch: Any,
) -> None:
    restored = {
        name: torch.tensor(value, dtype=torch.float32)
        for name, value in state_dict.items()
    }
    model.load_state_dict(restored)
    model.eval()


def _get_runtime_model(model_bundle: dict[str, Any]) -> _TorchPriceRegressor:
    cached = model_bundle.get("_runtime_model")
    if cached is not None:
        return cached

    estimator = model_bundle.get("estimator")
    if not isinstance(estimator, dict):
        raise ValueError("Torch model bundle estimator must be a dict")

    torch = _require_torch()
    model_config = estimator["model_config"]
    model = _TorchPriceRegressor(
        int(model_config["input_dim"]),
        [int(value) for value in model_config["hidden_layers"]],
        dropout=float(model_config["dropout"]),
        torch=torch,
    )
    _load_serialized_state_dict(model, estimator["state_dict"], torch)
    model_bundle["_runtime_model"] = model
    return model


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch backend requires the optional 'torch' dependency. "
            "Install it with: .venv/bin/pip install -e '.[torch]'"
        ) from exc
    return torch
