import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

# Generate and save confusion matrix image
y_train_pred = model.predict(X)
cm = confusion_matrix(y_encoded, y_train_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Prediction function
def classify(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(features)[0]
    return le.inverse_transform([pred])[0]

# Preset examples
examples = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [6.0, 2.2, 4.0, 1.0],  # Versicolor
    [6.9, 3.1, 5.1, 2.3]   # Virginica
]

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌸 Iris Flower Classifier

        Predict the species of an Iris flower based on its measurements.  
        Built using **Logistic Regression**.

        👉 Click on the **example values** below to auto-fill and test.

        A complete guide and README: [GitHub Repo](https://github.com/kvj-harsha/iris-classifier-app)

        **Author:**  
        [@kvjharsha](https://linkedin.com/in/kvjharsha) | [@kvj-harsha](https://github.com/kvj-harsha)
        """
    )

    with gr.Row():
        with gr.Column():
            sepal_length = gr.Number(label="Sepal Length")
            sepal_width = gr.Number(label="Sepal Width")
            petal_length = gr.Number(label="Petal Length")
            petal_width = gr.Number(label="Petal Width")
            submit_btn = gr.Button("🔍 Predict")

        with gr.Column():
            result = gr.Textbox(label="Predicted Species", interactive=False)

    gr.Examples(
        examples=examples,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        label="💡 Example Presets (click to auto-fill above)"
    )

    gr.Image("confusion_matrix.png", label="📊 Confusion Matrix (on training data)")

    submit_btn.click(fn=classify,
                     inputs=[sepal_length, sepal_width, petal_length, petal_width],
                     outputs=result)

demo.launch()
