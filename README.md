# Iris Flower Classifier

A lightweight application that predicts the species of an Iris flower using **Logistic Regression**, built with **Gradio** hosted on hugging face space.

It takes 4 numeric inputs — *sepal length*, *sepal width*, *petal length*, and *petal width* — and classifies the flower into one of the three Iris species: **Setosa**, **Versicolor**, or **Virginica**.

## 🚀 Demo

Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/kvj-harsha/iris-classifier-app) using Gradio UI.

## Workflow

<img src="https://github.com/user-attachments/assets/bbe9391e-9b60-45f9-9acf-c4903b10c5ea" alt="workflow" width="600"/>

## 🧠 Model Info

- **Model**: Logistic Regression  
- **Dataset**: [Iris Dataset (UCI ML Repo)](https://archive.ics.uci.edu/ml/datasets/iris)  
- **Metrics**: Confusion matrix shown in-app  
- **Accuracy**: ~97% on full dataset

## 📂 Project Structure

```
.
├── app.py                  # Main Gradio app
├── iris.data              # Raw iris dataset
├── confusion_matrix.png   # Generated after training
├── iris_classifier.ipynb  # Jupyter notebook version
└── requirements.txt       # Dependencies
```

## 📝 Usage

### ⬇️ Clone and Run Locally

```bash
git clone https://github.com/kvj-harsha/iris-classifier-app.git
cd iris-classifier-app
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### ▶️ Or Try on Hugging Face Spaces

You can try it out directly via the interactive web UI hosted on Hugging Face:  
👉 [Launch App](https://huggingface.co/spaces/kvj-harsha/iris-classifier-app)

## 📊 Example Input

| Sepal Length | Sepal Width | Petal Length | Petal Width | Predicted |
|--------------|-------------|---------------|--------------|------------|
| 5.1          | 3.5         | 1.4           | 0.2          | Setosa     |
| 6.0          | 2.2         | 4.0           | 1.0          | Versicolor |
| 6.9          | 3.1         | 5.1           | 2.3          | Virginica  |

## 🧪 Notebook

The notebook `iris_classifier.ipynb` contains:
- Data preprocessing steps
- Model training
- Accuracy evaluation
- Confusion matrix visualization

## 🪄 Built With

- [Gradio](https://gradio.app/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

---

⭐ If you like this project, consider starring it on GitHub!
