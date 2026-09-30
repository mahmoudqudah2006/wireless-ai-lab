from wireless_ai.baseline import train_random_forest
from wireless_ai.dataset import generate_link_dataset


def test_baseline_trains() -> None:
    dataset = generate_link_dataset(800, seed=5)
    model, metrics = train_random_forest(dataset, seed=5)
    assert metrics["accuracy"] > 0.9
    assert len(model.feature_importances_) == len(dataset.feature_names)
