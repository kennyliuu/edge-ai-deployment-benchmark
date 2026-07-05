#!/usr/bin/env python3
"""Train a small MNIST CNN and export FP32 / INT8 TFLite models."""

import os
import sys

import numpy as np
import tensorflow as tf

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)


def export_fp32(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)


def export_int8(model, x_train, path):
    def representative_dataset():
        for i in range(100):
            yield [x_train[i : i + 1].reshape(1, 28, 28).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)


def evaluate_tflite(path, x_test, y_test, quantized=False):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    correct = 0
    for i in range(len(x_test)):
        x = x_test[i].reshape(1, 28, 28)
        if quantized:
            scale, zero_point = inp["quantization"]
            x_q = (x / scale + zero_point).astype(inp["dtype"])
            interpreter.set_tensor(inp["index"], x_q)
        else:
            interpreter.set_tensor(inp["index"], x.astype(np.float32))
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(out["index"])[0])
        if pred == y_test[i]:
            correct += 1
    return correct / len(x_test)


def main():
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    print("Training model (3 epochs)...")
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

    fp32_path = os.path.join(MODELS_DIR, "mnist_fp32.tflite")
    int8_path = os.path.join(MODELS_DIR, "mnist_int8.tflite")

    print("Exporting FP32 TFLite...")
    export_fp32(model, fp32_path)

    print("Exporting INT8 TFLite...")
    try:
        export_int8(model, x_train, int8_path)
    except Exception as exc:
        print(f"INT8 export failed: {exc}", file=sys.stderr)
        int8_path = None

    fp32_acc = evaluate_tflite(fp32_path, x_test, y_test, quantized=False)
    print(f"FP32 test accuracy: {fp32_acc:.4f}")

    if int8_path and os.path.exists(int8_path):
        int8_acc = evaluate_tflite(int8_path, x_test, y_test, quantized=True)
        print(f"INT8 test accuracy: {int8_acc:.4f}")
    else:
        int8_acc = None

    keras_path = os.path.join(MODELS_DIR, "mnist.keras")
    model.save(keras_path)
    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
