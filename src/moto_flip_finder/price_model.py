from __future__ import annotations

from datetime import UTC, datetime
from math import sqrt
import re
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .market_value import detect_generation_hint


NUMERIC_FEATURES = [
    "year",
    "engine_cc",
    "mileage_km",
    "image_count",
    "description_length",
    "title_length",
    "negotiable_flag",
]

CATEGORICAL_FEATURES = [
    "model_family",
    "normalized_model",
    "generation_hint",
    "brand",
    "seller_type",
    "origin_country",
    "vehicle_type",
    "technical_state",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


MODEL_HINT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("z1000sx", re.compile(r"\b(?:z\s?1000\s?sx|1000\s?sx|ninja\s?1000\s?sx)\b", re.IGNORECASE)),
    ("h2sx", re.compile(r"\bh2\s?sx\b", re.IGNORECASE)),
    ("zx4rr", re.compile(r"\bzx[-\s]?4rr\b", re.IGNORECASE)),
    ("zx10r", re.compile(r"\bzx[-\s]?10r\b", re.IGNORECASE)),
    ("zx9r", re.compile(r"\bzx[-\s]?9r\b", re.IGNORECASE)),
    ("zx7r", re.compile(r"\bzx[-\s]?7r\b", re.IGNORECASE)),
    ("zx6r", re.compile(r"\bzx[-\s]?6r\b", re.IGNORECASE)),
    ("z900rs", re.compile(r"\bz\s?900\s?rs\b", re.IGNORECASE)),
    ("z650rs", re.compile(r"\bz\s?650\s?rs\b", re.IGNORECASE)),
    ("z1000", re.compile(r"\bz\s?1000\b", re.IGNORECASE)),
    ("z900", re.compile(r"\bz\s?900\b", re.IGNORECASE)),
    ("z800", re.compile(r"\bz\s?800\b", re.IGNORECASE)),
    ("z750", re.compile(r"\bz\s?750\b", re.IGNORECASE)),
    ("z650", re.compile(r"\bz\s?650\b", re.IGNORECASE)),
    ("z400", re.compile(r"\bz\s?400\b", re.IGNORECASE)),
    ("z300", re.compile(r"\bz\s?300\b", re.IGNORECASE)),
    ("zrx", re.compile(r"\bzrx\b", re.IGNORECASE)),
    ("j125", re.compile(r"\bj\s?125\b", re.IGNORECASE)),
    ("eliminator", re.compile(r"\beliminator\b", re.IGNORECASE)),
    ("gtr1400", re.compile(r"\bgtr\s?1400\b", re.IGNORECASE)),
    ("gpz", re.compile(r"\bgpz\b", re.IGNORECASE)),
    ("versys", re.compile(r"\bversys\b", re.IGNORECASE)),
    ("ninja", re.compile(r"\bninja\b", re.IGNORECASE)),
    ("er6", re.compile(r"\ber[-\s]?6(?:n|f)?\b", re.IGNORECASE)),
    ("vulcan", re.compile(r"\bvulcan\b|\bvn\s?\d{2,4}\b|\bmean\s?streak\b", re.IGNORECASE)),
    ("kle", re.compile(r"\bkle\b", re.IGNORECASE)),
    ("klr", re.compile(r"\bklr\b", re.IGNORECASE)),
    ("w800", re.compile(r"\bw\s?800\b", re.IGNORECASE)),
    ("kx85", re.compile(r"\bkx\s?85\b", re.IGNORECASE)),
    ("kxf", re.compile(r"\bkx[-\s]?(?:f|250f|450f)\b|\bkxf\b", re.IGNORECASE)),
    ("kx", re.compile(r"\bkx\b", re.IGNORECASE)),
]

MODEL_FAMILY_MAP = {
    "versys": "versys",
    "ninja": "ninja",
    "h2sx": "h2sx",
    "z1000sx": "sx",
    "z1000": "z",
    "z900": "z",
    "z900rs": "z",
    "z800": "z",
    "z750": "z",
    "z650": "z",
    "z650rs": "z",
    "z400": "z",
    "z300": "z",
    "zrx": "zrx",
    "er6": "er6",
    "zx4rr": "zx",
    "zx6r": "zx",
    "zx7r": "zx",
    "zx9r": "zx",
    "zx10r": "zx",
    "gpz": "gpz",
    "gtr1400": "gtr",
    "vulcan": "vulcan",
    "kle": "dual",
    "klr": "dual",
    "kxf": "kx",
    "kx": "kx",
    "kx85": "kx",
    "w800": "w",
    "j125": "j",
    "eliminator": "eliminator",
}

GENERIC_TOKENS = {"kawasaki", "motor", "motocykl", "sprzedam", "stan", "rok", "raty", "abs"}


def prepare_price_training_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for record in records:
        price = _positive_int_or_none(record.get("price_pln"))
        title = _string_or_none(record.get("title"))
        description = _string_or_none(record.get("full_description")) or _string_or_none(
            record.get("short_description")
        )
        brand = _string_or_none(record.get("brand"))
        model_hint = detect_model_hint(title, description, brand)
        model_family = detect_model_family(
            title,
            description,
            brand,
            model_hint=model_hint,
            normalized_model=_string_or_none(record.get("normalized_model")),
        )
        normalized_model = _normalized_model_or_fallback(
            record,
            brand=brand,
            model_family=model_family,
        )
        if price is None or normalized_model is None:
            continue

        prepared.append(
            {
                "url": _string_or_none(record.get("url")),
                "price_pln": price,
                "model_family": model_family,
                "normalized_model": normalized_model,
                "generation_hint": detect_generation_hint(title, description),
                "brand": brand,
                "seller_type": _string_or_none(record.get("seller_type")),
                "origin_country": _string_or_none(record.get("origin_country")),
                "vehicle_type": _string_or_none(record.get("vehicle_type")),
                "technical_state": _string_or_none(record.get("technical_state")),
                "year": _int_or_none(record.get("year")),
                "engine_cc": _int_or_none(record.get("engine_cc")),
                "mileage_km": _int_or_none(record.get("mileage_km")),
                "image_count": len(_string_list(record.get("image_urls"))),
                "description_length": len(description or ""),
                "title_length": len(title or ""),
                "negotiable_flag": _bool_to_int(record.get("negotiable")),
            }
        )

    return prepared


def train_price_model(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    cv_folds: int = 5,
    search_iterations: int = 16,
) -> dict[str, Any]:
    prepared = prepare_price_training_records(records)
    if len(prepared) < 20:
        raise ValueError("Need at least 20 healthy records to train the sklearn price model")

    dataset = pd.DataFrame(prepared)
    x = dataset[FEATURE_COLUMNS]
    y = dataset["price_pln"]

    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=random_seed,
    )

    cv_splits = min(cv_folds, len(x_train))
    if cv_splits < 2:
        raise ValueError("Training split is too small for cross-validation")

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)

    candidate_results = []
    for candidate_name, estimator, param_distributions in _candidate_search_spaces(random_seed):
        search = RandomizedSearchCV(
            estimator=_build_model_pipeline(estimator),
            param_distributions=param_distributions,
            n_iter=search_iterations,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            random_state=random_seed,
            refit=True,
        )
        search.fit(x_train, y_train)
        candidate_results.append(
            {
                "candidate_name": candidate_name,
                "search": search,
                "cv_mae_pln": round(abs(float(search.best_score_)), 2),
            }
        )

    best_candidate = min(candidate_results, key=lambda item: item["cv_mae_pln"])
    best_search: RandomizedSearchCV = best_candidate["search"]
    best_model = best_search.best_estimator_

    training_predictions = best_model.predict(x_train)
    validation_predictions = best_model.predict(x_validation)

    training_metrics = _regression_metrics(y_train.to_list(), training_predictions.tolist())
    validation_metrics = _regression_metrics(y_validation.to_list(), validation_predictions.tolist())

    feature_importances = _extract_feature_importances(best_model)

    return {
        "version": 3,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "candidate_scores": [
            {
                "candidate_name": item["candidate_name"],
                "cv_mae_pln": item["cv_mae_pln"],
            }
            for item in sorted(candidate_results, key=lambda item: item["cv_mae_pln"])
        ],
        "best_candidate_name": best_candidate["candidate_name"],
        "best_params": best_search.best_params_,
        "training_size": len(x_train),
        "validation_size": len(x_validation),
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "feature_importances": feature_importances[:20],
        "estimator": best_model,
    }


def predict_price(record: dict[str, Any], model_bundle: dict[str, Any]) -> int | None:
    estimator = model_bundle.get("estimator")
    if estimator is None:
        return None

    prepared = prepare_price_training_records([record])
    if not prepared:
        raw_record = _prepare_record_without_price(record)
        if raw_record is None:
            return None
        prepared = [raw_record]

    frame = pd.DataFrame(prepared)[FEATURE_COLUMNS]
    prediction = estimator.predict(frame)[0]
    return max(0, int(round(float(prediction))))


def score_records_with_price_model(
    records: list[dict[str, Any]],
    model_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        predicted_price = predict_price(record, model_bundle)
        scored.append(
            {
                "original_listing": record,
                "predicted_price_pln": predicted_price,
            }
        )
    return scored


def _build_model_pipeline(regressor: Any) -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("model", regressor),
        ]
    )


def _candidate_search_spaces(random_seed: int) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "extra_trees",
            ExtraTreesRegressor(random_state=random_seed, n_jobs=-1),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 10, 16, 24],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [0.6, 0.8, 1.0],
            },
        ),
        (
            "random_forest",
            RandomForestRegressor(random_state=random_seed, n_jobs=-1),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 10, 16, 24],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.6, 0.8],
            },
        ),
    ]


def _extract_feature_importances(model: Pipeline) -> list[dict[str, Any]]:
    regressor = model.named_steps["model"]
    if not hasattr(regressor, "feature_importances_"):
        return []

    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    pairs = sorted(
        zip(feature_names.tolist(), importances.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in pairs
    ]


def _regression_metrics(actual: list[int], predicted: list[float]) -> dict[str, Any]:
    rounded_predictions = [float(value) for value in predicted]
    mae = round(float(mean_absolute_error(actual, rounded_predictions)), 2)
    rmse = round(float(sqrt(mean_squared_error(actual, rounded_predictions))), 2)
    median_abs_error = round(
        float(median(abs(float(pred) - float(act)) for act, pred in zip(actual, rounded_predictions))),
        2,
    )
    mape = round(
        float(
            sum(abs(float(pred) - float(act)) / float(act) for act, pred in zip(actual, rounded_predictions))
            / len(actual)
            * 100.0
        ),
        2,
    )
    return {
        "count": len(actual),
        "mae_pln": mae,
        "rmse_pln": rmse,
        "median_absolute_error_pln": median_abs_error,
        "mape_percent": mape,
    }


def _prepare_record_without_price(record: dict[str, Any]) -> dict[str, Any] | None:
    title = _string_or_none(record.get("title"))
    description = _string_or_none(record.get("full_description")) or _string_or_none(
        record.get("short_description")
    )
    brand = _string_or_none(record.get("brand"))
    model_hint = detect_model_hint(title, description, brand)
    model_family = detect_model_family(
        title,
        description,
        brand,
        model_hint=model_hint,
        normalized_model=_string_or_none(record.get("normalized_model")),
    )
    normalized_model = _normalized_model_or_fallback(record, brand=brand, model_family=model_family)
    if normalized_model is None:
        return None

    return {
        "url": _string_or_none(record.get("url")),
        "price_pln": None,
        "model_family": model_family,
        "normalized_model": normalized_model,
        "generation_hint": detect_generation_hint(title, description),
        "brand": brand,
        "seller_type": _string_or_none(record.get("seller_type")),
        "origin_country": _string_or_none(record.get("origin_country")),
        "vehicle_type": _string_or_none(record.get("vehicle_type")),
        "technical_state": _string_or_none(record.get("technical_state")),
        "year": _int_or_none(record.get("year")),
        "engine_cc": _int_or_none(record.get("engine_cc")),
        "mileage_km": _int_or_none(record.get("mileage_km")),
        "image_count": len(_string_list(record.get("image_urls"))),
        "description_length": len(description or ""),
        "title_length": len(title or ""),
        "negotiable_flag": _bool_to_int(record.get("negotiable")),
    }


def detect_model_hint(
    title: str | None,
    description: str | None,
    brand: str | None,
) -> str | None:
    text = " ".join(part for part in (title, description) if part)
    if not text:
        return None

    for label, pattern in MODEL_HINT_PATTERNS:
        if pattern.search(text):
            return label

    normalized_brand = brand.casefold() if isinstance(brand, str) else None
    if normalized_brand == "kawasaki":
        compact = re.sub(r"[^a-z0-9]+", " ", text.casefold())
        tokens = [token for token in compact.split() if token not in GENERIC_TOKENS]
        for token in tokens:
            canonical = _canonicalize_kawasaki_token(token)
            if canonical is not None:
                return canonical

    return None


def detect_model_family(
    title: str | None,
    description: str | None,
    brand: str | None,
    *,
    model_hint: str | None = None,
    normalized_model: str | None = None,
) -> str | None:
    explicit = normalized_model.casefold() if isinstance(normalized_model, str) and normalized_model.strip() else None
    if explicit:
        if "gsxr" in explicit:
            return "gsxr"
        for family in [
            "versys",
            "ninja",
            "sx",
            "h2sx",
            "zx",
            "zrx",
            "z",
            "er6",
            "vulcan",
            "dual",
            "gtr",
            "gpz",
            "w",
            "jx",
            "j",
            "kx",
            "eliminator",
        ]:
            if f"_{family}_" in f"_{explicit}_":
                return family

    hint = model_hint or detect_model_hint(title, description, brand)
    if hint is None:
        return None
    return MODEL_FAMILY_MAP.get(hint, hint)


def _canonicalize_kawasaki_token(token: str) -> str | None:
    compact = re.sub(r"[^a-z0-9]+", "", token.casefold())
    if not compact or compact.isdigit():
        return None

    if compact in {"1000sx", "ninja1000sx", "z1000sx"}:
        return "z1000sx"
    if compact in {"h2sx"}:
        return "h2sx"
    if compact in {"zx4rr", "zx4r"}:
        return "zx4rr"
    if compact in {"zx6r", "zx636r", "zx636"}:
        return "zx6r"
    if compact in {"zx7r"}:
        return "zx7r"
    if compact in {"zx9r"}:
        return "zx9r"
    if compact in {"zx10r"}:
        return "zx10r"
    if compact in {"z900rs"}:
        return "z900rs"
    if compact in {"z650rs"}:
        return "z650rs"
    if compact in {"z1000", "z900", "z800", "z750", "z650", "z400", "z300"}:
        return compact
    if compact in {"zrx"}:
        return "zrx"
    if compact.startswith("er6"):
        return "er6"
    if compact.startswith("vn") or compact.startswith("vulcan") or compact == "meanstreak":
        return "vulcan"
    if compact in {"versys", "ninja", "gpz", "kle", "klr", "w800", "eliminator", "j125"}:
        return compact
    if compact in {"kxf", "kx250f", "kx450f"}:
        return "kxf"
    if compact in {"kx", "kx85"}:
        return compact
    if compact == "gtr1400":
        return "gtr1400"
    if compact.startswith("z") and any(ch.isdigit() for ch in compact):
        return compact
    if compact.startswith("zx") and any(ch.isdigit() for ch in compact):
        return compact
    if compact.startswith("n") and any(ch.isdigit() for ch in compact):
        return compact
    return None


def _normalized_model_or_fallback(
    record: dict[str, Any],
    *,
    brand: str | None,
    model_family: str | None,
) -> str | None:
    explicit = _string_or_none(record.get("normalized_model"))
    if explicit is not None:
        return explicit.casefold()

    normalized_brand = brand.casefold().replace(" ", "_") if brand else None
    engine_cc = _int_or_none(record.get("engine_cc"))

    parts = [part for part in [normalized_brand, model_family] if part]
    if engine_cc is not None:
        parts.append(str(engine_cc))
    if parts:
        return "_".join(parts)

    return normalized_brand


def _to_dense(value: Any) -> Any:
    if hasattr(value, "toarray"):
        return value.toarray()
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _positive_int_or_none(value: object) -> int | None:
    number = _int_or_none(value)
    if number is None or number <= 0:
        return None
    return number


def _bool_to_int(value: object) -> int:
    return 1 if value is True else 0
