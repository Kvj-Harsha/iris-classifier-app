import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gradio as gr

# Load dataset
df = pd.read_csv("iris.data", header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df.dropna(inplace=True)

# Prepare data
X = df.drop("species", axis=1)
y = df["species"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y_encoded)

# Prediction function
def classify(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(features)[0]
    return le.inverse_transform([pred])[0]

# Gradio UI
iface = gr.Interface(
    fn=classify,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs=gr.Text(label="Predicted Species"),
    title="🌸 Iris Flower Classifier",
    description="Enter flower measurements to predict its species!"
)

iface.launch()
