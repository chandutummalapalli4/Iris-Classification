# 🌸 Iris Flower Classification Web App (Gradio + TensorFlow)

This project is a **machine learning web application** that predicts the species of an Iris flower based on input measurements using a trained neural network model. The interface is built using **Gradio**, making it easy to interact with the model in a browser.

---

## 🚀 Features

- Predict Iris flower species in real time
- Interactive web UI using Gradio
- Neural network model built with TensorFlow/Keras
- Saves and reloads trained model automatically
- Displays prediction probabilities
- Example inputs for quick testing

---

## 🛠️ Technologies Used

- Python 🐍
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Gradio

---

## 📂 Project Structure

iris-classification/
│
├── app.py              # Main application file
├── iris_model.h5       # Saved trained model (auto-generated)
├── README.md           # Project documentation

---

## 🧠 How It Works

1. Loads the Iris dataset from scikit-learn
2. Converts labels into categorical format
3. Splits data into training and testing sets
4. Builds a neural network:
   - Input layer: 4 features
   - Hidden layers: 10 and 8 neurons
   - Output layer: 3 classes (Setosa, Versicolor, Virginica)
5. Trains model (if not already saved)
6. Saves model as `iris_model.h5`
7. Uses trained model for predictions via Gradio UI

---

## 📊 Model Performance

- Accuracy is evaluated on test dataset
- Training graph shows accuracy and loss trends

---

## ⚙️ How to Run the Project

### 1. Install dependencies
```bash
pip install gradio numpy matplotlib scikit-learn tensorflow
```

### 2. Run the application
```bash
python app.py
```

### 3. Open in browser
A local or public Gradio link will be generated automatically.

---

## 🌐 Input Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 📌 Example Predictions

- `[5.1, 3.5, 1.4, 0.2]` → Setosa
- `[6.0, 2.9, 4.5, 1.5]` → Versicolor
- `[6.9, 3.1, 5.4, 2.1]` → Virginica

---

## 🔮 Future Improvements

- Improve model accuracy with deeper networks
- Add data visualization dashboard
- Deploy on Hugging Face / Streamlit Cloud
- Add dataset exploration page
- Export predictions to CSV

---

## 👨‍💻 Author
Chandu Tummlapalli

---

## 📜 License

This project is open-source and free to use for learning purposes.
