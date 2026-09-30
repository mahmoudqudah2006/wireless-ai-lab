from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .dataset import LinkDataset
from .link import CLASS_NAMES


def train_random_forest(
    dataset: LinkDataset, seed: int = 7
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=0.25,
        random_state=seed,
        stratify=dataset.labels,
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "classes": CLASS_NAMES,
        "feature_importance": {
            name: float(value)
            for name, value in zip(dataset.feature_names, model.feature_importances_, strict=True)
        },
    }
