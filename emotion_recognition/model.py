# ─────────────────────────────────────────────
#  model.py  |  CNN + LSTM hybrid
#
#  Architecture overview:
#    Input (T, F)
#      │
#      ├─ [CNN branch] Conv1D × 3 → GlobalMaxPool → Dense
#      │
#      └─ [LSTM branch] LSTM × 2 → last hidden state
#           │
#      Concatenate → Dense → Dropout → Softmax
# ─────────────────────────────────────────────

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from config import (
    NUM_CLASSES, MAX_PAD_LEN, N_MFCC,
    CNN_FILTERS, KERNEL_SIZE,
    LSTM_UNITS, DROPOUT_RATE, L2_REG, LEARNING_RATE
)


def build_cnn_lstm(input_shape=(MAX_PAD_LEN, N_MFCC),
                   num_classes=NUM_CLASSES) -> Model:
    """
    CNN captures local spectral texture; LSTM captures temporal dynamics.
    Both branches share the same input and their outputs are concatenated.
    """
    inp = layers.Input(shape=input_shape, name="mfcc_input")  # (T, F)

    # ── CNN branch ────────────────────────────
    x_cnn = inp
    for filters in CNN_FILTERS:
        x_cnn = layers.Conv1D(
            filters, KERNEL_SIZE,
            padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f"conv_{filters}"
        )(x_cnn)
        x_cnn = layers.BatchNormalization()(x_cnn)
        x_cnn = layers.MaxPooling1D(pool_size=2)(x_cnn)
        x_cnn = layers.Dropout(DROPOUT_RATE / 2)(x_cnn)

    x_cnn = layers.GlobalAveragePooling1D(name="gap")(x_cnn)
    x_cnn = layers.Dense(128, activation="relu",
                          kernel_regularizer=regularizers.l2(L2_REG))(x_cnn)

    # ── LSTM branch ───────────────────────────
    x_lstm = inp
    for i, units in enumerate(LSTM_UNITS):
        return_seq = (i < len(LSTM_UNITS) - 1)  # all but last return seqs
        x_lstm = layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=DROPOUT_RATE / 2,
            recurrent_dropout=0.1,
            name=f"lstm_{units}"
        )(x_lstm)

    x_lstm = layers.Dense(64, activation="relu",
                           kernel_regularizer=regularizers.l2(L2_REG))(x_lstm)

    # ── Merge ─────────────────────────────────
    merged = layers.Concatenate(name="merge")([x_cnn, x_lstm])
    merged = layers.Dense(128, activation="relu",
                          kernel_regularizer=regularizers.l2(L2_REG))(merged)
    merged = layers.Dropout(DROPOUT_RATE)(merged)
    out    = layers.Dense(num_classes, activation="softmax",
                          name="emotion_output")(merged)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_EmotionNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    m = build_cnn_lstm()
    m.summary()
