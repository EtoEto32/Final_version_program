# ============================================================
# CNN-based Malware Classification
# (Paper-aligned, Image-based, Sample-leak-safe, GPU-utilized)
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
from sklearn.metrics import roc_auc_score

# ============================================================
# mixed precision（RTX 4060 最重要）
# ============================================================
mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# 設定（論文準拠）
# ============================================================
BASE_DIR = Path("data/memdump_images_256")
IMG_SIZE = (256, 256)
BATCH_SIZE = 64              # GPU向け（論文範囲内）
EPOCHS = 100
N_SPLITS = 10
SEED = 42
L2_PENALTY = 0.2

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 検体データ構造
# ============================================================
@dataclass
class Sample:
    sample_id: str
    label: int
    image_paths: List[str]

# ============================================================
# データ収集（検体単位）
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
# 検体 → 画像単位展開（リーク防止）
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
# tf.data
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
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ============================================================
# モデル構築（論文準拠 + fine-tuning）
# ============================================================
def build_model():
    backbone = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # 論文で一般的：上位層のみ fine-tuning
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    for layer in backbone.layers[-20:]:
        layer.trainable = True

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = backbone(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_PENALTY)
    )(x)

    # mixed precision 対策：出力は float32
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ============================================================
# 検体単位 AUC
# ============================================================
def aggregate_by_sample(probs, labels, sample_ids):
    buf, lab = {}, {}
    for p, y, sid in zip(probs, labels, sample_ids):
        buf.setdefault(sid, []).append(p)
        lab[sid] = y
    sample_probs = np.array([np.mean(buf[sid]) for sid in buf])
    sample_labels = np.array([lab[sid] for sid in buf])
    return sample_probs, sample_labels

# ============================================================
# メイン：Stratified 10-fold CV
# ============================================================
def main():
    benign = collect_samples(BASE_DIR, 0)
    malicious = collect_samples(BASE_DIR, 1)
    samples = benign + malicious

    y = np.array([s.label for s in samples])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    aucs = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(samples)), y), 1):
        print(f"\n===== Fold {fold} =====")

        train_samples = [samples[i] for i in tr_idx]
        test_samples  = [samples[i] for i in te_idx]

        tr_p, tr_y, _      = expand_to_images(train_samples)
        te_p, te_y, te_sid = expand_to_images(test_samples)

        train_ds = make_dataset(tr_p, tr_y, training=True)
        test_ds  = make_dataset(te_p, te_y)

        model = build_model()

        model.fit(
            train_ds,
            epochs=EPOCHS,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="loss",
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )

        probs = model.predict(test_ds).reshape(-1)
        samp_probs, samp_y = aggregate_by_sample(probs, te_y, te_sid)
        auc = roc_auc_score(samp_y, samp_probs)
        aucs.append(auc)

        print(f"検体単位AUC: {auc:.4f}")

    print("\n===== CV Result =====")
    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if __name__ == "__main__":
    main()
