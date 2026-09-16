"""Optional convolutional models; TensorFlow is imported only on demand."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .evaluation import metrics


def split_groups(y, groups, seed):
    first = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train, rest = next(first.split(y, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_rel, test_rel = next(second.split(y[rest], y[rest], groups[rest]))
    val, test = rest[val_rel], rest[test_rel]
    partitions = (train, val, test)
    for i, idx in enumerate(partitions):
        if len(np.unique(y[idx])) != 2:
            raise ValueError("CNN train, validation, and test partitions must each contain both classes")
        for other in partitions[i + 1:]:
            if set(groups[idx]) & set(groups[other]):
                raise ValueError("Group leakage detected")
    return partitions


def build_model(shape, kind):
    from tensorflow import keras
    layers = keras.layers
    if kind == "cnn1d":
        if len(shape) != 2 or shape[0] < 2:
            raise ValueError("cnn1d requires at least two ordered features")
        stack = [layers.Input(shape=shape),
                 layers.Conv1D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(),
                 layers.Conv1D(32, 3, padding="same", activation="relu"), layers.MaxPooling1D(2),
                 layers.Dropout(0.2), layers.Conv1D(64, 3, padding="same", activation="relu"),
                 layers.BatchNormalization(), layers.Conv1D(64, 3, padding="same", activation="relu"),
                 layers.GlobalAveragePooling1D(), layers.Dropout(0.3)]
    elif kind == "cnn2d":
        if len(shape) != 3 or min(shape[:2]) < 4:
            raise ValueError("cnn2d requires image height and width of at least four")
        stack = [layers.Input(shape=shape)]
        for filters, dropout in [(32, 0.2), (64, 0.25)]:
            stack += [layers.Conv2D(filters, 3, padding="same", activation="relu"),
                      layers.BatchNormalization(), layers.Conv2D(filters, 3, padding="same", activation="relu"),
                      layers.MaxPooling2D(), layers.Dropout(dropout)]
        stack += [layers.Conv2D(128, 3, padding="same", activation="relu"), layers.BatchNormalization(),
                  layers.GlobalAveragePooling2D(), layers.Dropout(0.35)]
    else:
        raise ValueError("Unknown CNN model")
    stack += [layers.Dense(64, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")]
    model = keras.Sequential(stack)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def evaluate_cnn(images, y, groups, kind, epochs=50, seed=42):
    from tensorflow import keras
    from .data import channel_means
    if epochs < 1:
        raise ValueError("epochs must be positive")
    keras.utils.set_random_seed(seed)
    train, val, test = split_groups(y, groups, seed)
    X = channel_means(images) if kind == "cnn1d" else images.astype(np.float64)
    axes = (0,) if kind == "cnn1d" else (0, 1, 2)
    mean = X[train].mean(axis=axes, keepdims=True)
    std = X[train].std(axis=axes, keepdims=True)
    X = ((X - mean) / np.maximum(std, 1e-8)).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Normalization produced nonfinite values")
    if kind == "cnn1d":
        X = X[..., None]
    model = build_model(X.shape[1:], kind)
    counts = np.bincount(y[train], minlength=2)
    weights = {i: float(len(train) / (2 * counts[i])) for i in (0, 1)}
    model.fit(X[train], y[train], validation_data=(X[val], y[val]), epochs=epochs,
              batch_size=32, class_weight=weights, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)])
    prediction = (model.predict(X[test], verbose=0).ravel() >= 0.5).astype(int)
    return {"method": kind, "seed": seed, "max_epochs": epochs,
            **metrics(y[test], prediction, groups[test])}
