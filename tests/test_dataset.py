import numpy as np

from wireless_ai.dataset import generate_link_dataset
from wireless_ai.link import modulation_class


def test_dataset_is_reproducible() -> None:
    first = generate_link_dataset(100, seed=3)
    second = generate_link_dataset(100, seed=3)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)


def test_modulation_labels_increase_with_snr() -> None:
    labels = modulation_class(np.array([-5.0, 5.0, 12.0, 25.0]))
    assert labels.tolist() == [0, 1, 2, 3]
