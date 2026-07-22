from __future__ import annotations

import random
from typing import Sequence, TypeVar

from sklearn.model_selection import train_test_split


T = TypeVar("T")


def split_known_and_unknown(
    items: Sequence[T],
    *,
    class_index: int = 0,
    val_size: float = 0.15,
    test_size: float = 0.15,
    open_set_classes: int = 0,
    open_set_val_size: float = 0.5,
    seed: int = 42,
) -> tuple[list[T], list[T], list[T], list[T], list[T], list[str], list[str]]:
    """Split samples without leaking held-out identities into known-class splits.

    Returns known train/val/test, unknown calibration/test, known class names and
    held-out class names. Calibration and test use disjoint unknown identities.
    """
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("val_size y test_size deben estar entre 0 y 1.")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size debe ser menor que 1.")
    if open_set_classes < 0:
        raise ValueError("open_set_classes no puede ser negativo.")
    if open_set_classes == 1:
        raise ValueError(
            "open_set_classes debe ser 0 o al menos 2 para separar identidades "
            "unknown de calibración y test."
        )
    if open_set_classes and not 0.0 < open_set_val_size < 1.0:
        raise ValueError("open_set_val_size debe estar entre 0 y 1.")

    classes = sorted({str(item[class_index]) for item in items})
    if open_set_classes >= len(classes):
        raise ValueError(
            f"Se pidieron {open_set_classes} clases open-set, pero solo hay "
            f"{len(classes)} clases. Deben quedar al menos 2 clases conocidas."
        )
    if len(classes) - open_set_classes < 2:
        raise ValueError("Deben quedar al menos 2 clases conocidas para formar tripletas.")

    rng = random.Random(seed)
    unknown_classes = sorted(rng.sample(classes, open_set_classes))
    unknown_class_set = set(unknown_classes)
    known_classes = [name for name in classes if name not in unknown_class_set]
    shuffled_unknown = unknown_classes.copy()
    rng.shuffle(shuffled_unknown)
    n_unknown_val = min(
        open_set_classes - 1,
        max(1, round(open_set_classes * open_set_val_size)),
    ) if open_set_classes else 0
    unknown_val_classes = set(shuffled_unknown[:n_unknown_val])

    train: list[T] = []
    val: list[T] = []
    test: list[T] = []
    unknown_val: list[T] = []
    unknown_test: list[T] = []

    for class_name in classes:
        class_items = [item for item in items if str(item[class_index]) == class_name]
        if class_name in unknown_class_set:
            if class_name in unknown_val_classes:
                unknown_val.extend(class_items)
            else:
                unknown_test.extend(class_items)
            continue

        if len(class_items) < 4:
            raise ValueError(
                f"La clase conocida '{class_name}' tiene {len(class_items)} muestras; "
                "se necesitan al menos 4 para train/val/test."
            )
        train_val, class_test = train_test_split(
            class_items, test_size=test_size, random_state=seed
        )
        val_ratio = val_size / (1.0 - test_size)
        class_train, class_val = train_test_split(
            train_val, test_size=val_ratio, random_state=seed
        )
        if len(class_train) < 2 or len(class_val) < 2:
            raise ValueError(
                f"La clase conocida '{class_name}' quedó con menos de 2 muestras "
                "en train o val; aumentá las muestras o ajustá val_size/test_size."
            )
        train.extend(class_train)
        val.extend(class_val)
        test.extend(class_test)

    return train, val, test, unknown_val, unknown_test, known_classes, unknown_classes
