import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ============================================
# Train Multiple Models
# ============================================

def train_models(X_train, X_test, y_train, y_test, feature_names):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC()
    }

    accuracy = {}

    best_score = 0
    best_model = None
    best_model_name = ""

    best_prediction = None

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        score = accuracy_score(y_test, prediction)

        accuracy[name] = round(score * 100, 2)

        if score > best_score:

            best_score = score
            best_model = model
            best_model_name = name
            best_prediction = prediction

    return {
        "accuracy": accuracy,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "prediction": best_prediction
    }


# ============================================
# Accuracy Comparison Plot
# ============================================

def plot_accuracy(accuracy_dict):

    plt.figure(figsize=(8,5))

    models = list(accuracy_dict.keys())
    scores = list(accuracy_dict.values())

    plt.bar(models, scores)

    plt.xticks(rotation=20)

    plt.ylabel("Accuracy (%)")

    plt.xlabel("Models")

    plt.title("Model Accuracy Comparison")

    plt.tight_layout()

    path = "static/accuracy.png"

    plt.savefig(path)

    plt.close()

    return path


# ============================================
# Confusion Matrix
# ============================================

def plot_confusion_matrix(y_test, y_pred, model_name):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    path = "static/confusion_matrix.png"

    plt.savefig(path)

    plt.close()

    return path


# ============================================
# Predict New Sample (Optional)
# ============================================

def predict_sample(model, sample):

    prediction = model.predict([sample])

    return prediction[0]


# ============================================
# Display Accuracy Table
# ============================================

def accuracy_table(accuracy_dict):

    table = []

    for model, score in accuracy_dict.items():

        table.append({
            "Model": model,
            "Accuracy": score
        })

    return table