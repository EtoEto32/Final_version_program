# ============================================================
# CNN-based Malware Classification (Deep-Hook-aligned)
# Image-based / Sample-leak-safe / 10-fold CV
#
# Evaluation metrics (executable-level):
#  - AUC
#  - Accuracy
#  - TPR (Detection Rate / Recall)
#  - FPR (False Alarm Rate)
#  - Precision
#  - F1-score
# ============================================================

import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import Xception
from tensorflow.keras import mixed_precision

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt

# ============================================================
# mixed precision (GPU)
# ============================================================
mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# Configuration (paper-aligned)
# ============================================================
BASE_DIR = Path("data/memdump_images_256")
IMG_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 20
N_SPLITS = 10
SEED = 42
L2_PENALTY = 1e-3

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Sample structure
# ============================================================
@dataclass
class Sample:
    sample_id: str
    label: int
    image_paths: List[str]

# ============================================================
# Data collection (sample-level)
# ============================================================
def collect_samples(base_dir: Path, label: int) -> List[Sample]:
    cls = "benign" if label == 0 else "malicious"
    samples = []
    for d in sorted((base_dir / cls).iterdir()):
        if d.is_dir():
            imgs = sorted([str(p) for p in d.glob("*.jpg")])
            if imgs:
                samples.append(Sample(d.name, label, imgs))
    return samples

# ============================================================
# Expand samples to image-level (leak-safe)
# ============================================================
def expand_to_images(samples: List[Sample]) -> Tuple[List[str], List[int], List[str]]:
    paths, labels, sample_ids = [], [], []
    for s in samples:
        for p in s.image_paths:
            paths.append(p)
            labels.append(s.label)
            sample_ids.append(s.sample_id)
    return paths, labels, sample_ids

# ============================================================
# tf.data pipeline
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 1000), seed=SEED)
    ds = ds.map(load_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ============================================================
# Model (Xception + fine-tuning)
# ============================================================
def build_model():
    backbone = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    for layer in backbone.layers[:-20]:
        layer.trainable = False
    for layer in backbone.layers[-20:]:
        layer.trainable = True

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_PENALTY)
    )(x)

    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ============================================================
# Aggregate predictions by executable
# ============================================================
def aggregate_by_sample(probs, labels, sample_ids):
    buf, lab = {}, {}
    for p, y, sid in zip(probs, labels, sample_ids):
        buf.setdefault(sid, []).append(float(p))
        lab[sid] = int(y)

    sids = sorted(buf.keys())
    sample_probs = np.array([np.mean(buf[sid]) for sid in sids], dtype=np.float32)
    sample_labels = np.array([lab[sid] for sid in sids], dtype=np.int32)
    return sample_probs, sample_labels

# ============================================================
# Compute executable-level metrics
# ============================================================
def compute_metrics(sample_probs, sample_labels, threshold=0.5):
    preds = (sample_probs >= threshold).astype(int)

    acc = accuracy_score(sample_labels, preds)
    tpr = recall_score(sample_labels, preds, zero_division=0)
    precision = precision_score(sample_labels, preds, zero_division=0)
    f1 = f1_score(sample_labels, preds, zero_division=0)

    cm = confusion_matrix(sample_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return acc, tpr, fpr, precision, f1

# ============================================================
# Main: 10-fold Stratified CV
# ============================================================
def main():
    benign = collect_samples(BASE_DIR, 0)
    malicious = collect_samples(BASE_DIR, 1)
    samples = benign + malicious

    y = np.array([s.label for s in samples], dtype=np.int32)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    aucs, accs, tprs, fprs, precisions, f1s = [], [], [], [], [], []
    all_probs, all_labels = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(samples)), y), 1):
        print(f"\n===== Fold {fold} =====")

        train_samples = [samples[i] for i in tr_idx]
        test_samples  = [samples[i] for i in te_idx]

        tr_p, tr_y, _      = expand_to_images(train_samples)
        te_p, te_y, te_sid = expand_to_images(test_samples)

        train_ds = make_dataset(tr_p, tr_y, training=True)
        test_ds  = make_dataset(te_p, te_y, training=False)

        model = build_model()
        model.fit(train_ds, epochs=EPOCHS, verbose=1)

        probs = model.predict(test_ds, verbose=0).reshape(-1)
        samp_probs, samp_y = aggregate_by_sample(probs, te_y, te_sid)

        auc = roc_auc_score(samp_y, samp_probs)
        acc, tpr, fpr, prec, f1 = compute_metrics(samp_probs, samp_y)

        aucs.append(auc)
        accs.append(acc)
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(prec)
        f1s.append(f1)

        all_probs.extend(samp_probs.tolist())
        all_labels.extend(samp_y.tolist())

        print(
            f"AUC: {auc:.4f} | "
            f"ACC: {acc:.4f} | "
            f"TPR: {tpr:.4f} | "
            f"FPR: {fpr:.4f} | "
            f"Precision: {prec:.4f} | "
            f"F1: {f1:.4f}"
        )

    print("\n===== CV Result (Executable-level) =====")
    print(f"AUC       : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"TPR       : {np.mean(tprs):.4f} ± {np.std(tprs):.4f}")
    print(f"FPR       : {np.mean(fprs):.4f} ± {np.std(fprs):.4f}")
    print(f"Precision : {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"F1-score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # ========================================================
    # Pooled ROC Curve
    # ========================================================
    fpr_curve, tpr_curve, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_curve, tpr_curve, label=f"ROC (mean AUC = {np.mean(aucs):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Executable-level, pooled 10-fold CV)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
