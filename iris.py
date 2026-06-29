import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os

iris = load_iris()
X = iris.data
y = to_categorical(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODEL_PATH = "iris_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        Dense(10, input_shape=(4,), activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
    model.save(MODEL_PATH)
    plt.figure(figsize=(6,3))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.title("Training Performance")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {acc*100:.2f}%")

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)
    species_index = np.argmax(prediction)
    species_name = iris.target_names[species_index]
    confidence = prediction[0][species_index] * 100
    probabilities = {name: float(pred) for name, pred in zip(iris.target_names, prediction[0])}
    result = f"Predicted Species: {species_name.capitalize()} (Confidence: {confidence:.2f}%)"
    return probabilities, result

inputs = [
    gr.Number(label="Sepal Length (cm)"),
    gr.Number(label="Sepal Width (cm)"),
    gr.Number(label="Petal Length (cm)"),
    gr.Number(label="Petal Width (cm)")
]
outputs = [
    gr.Label(num_top_classes=3, label="Prediction Probabilities"),
    gr.Textbox(label="Predicted Species")
]
examples = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.9, 3.1, 5.4, 2.1]
]
app = gr.Interface(
    fn=predict_species,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Iris Classification",
    theme="default",
    allow_flagging="never"
)
app.launch(share=True)
