from fileinput import filename

from flask_sqlalchemy import SQLAlchemy

import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils import (
    allowed_file,
    preprocess_data,
    generate_heatmap,
    plot_feature_importance,
)

from models import (
    train_models,
    plot_accuracy,
    plot_confusion_matrix,
)

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ml_insight.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Analysis(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(db.String(200))

    rows = db.Column(db.Integer)

    columns = db.Column(db.Integer)

    best_model = db.Column(db.String(100))

    accuracy = db.Column(db.Float)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("index.html", error="No file selected.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="Please choose a CSV file.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Only CSV files are allowed.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return render_template("index.html", error=f"Unable to read CSV: {e}")

    rows, columns = df.shape
    column_names = list(df.columns)
    preview = df.head().to_html(classes="table table-striped", index=False)

    try:
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    except Exception as e:
        return render_template("index.html", error=f"Preprocessing failed: {e}")

    heatmap = generate_heatmap(df)

    results = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )

    accuracy = results["accuracy"]
    best_model = results["best_model"]
    best_model_name = results["best_model_name"]
    prediction = results["prediction"]

    # Find highest accuracy
    best_accuracy = max(accuracy.values())

# Create database record
    analysis = Analysis(
        filename=filename,
        rows=rows,
        columns=columns,
        best_model=best_model_name,
        accuracy=best_accuracy
    )

# Save to database
    db.session.add(analysis)
    db.session.commit()

    accuracy_plot = plot_accuracy(accuracy)
    confusion_plot = plot_confusion_matrix(
        y_test,
        prediction,
        best_model_name,
    )

    try:
        feature_plot = plot_feature_importance(
            best_model,
            feature_names,
        )
    except Exception:
        feature_plot = None

    return render_template(
        "result.html",
        rows=rows,
        columns=columns,
        column_names=column_names,
        preview=preview,
        accuracy=accuracy,
        best_model=best_model_name,
        heatmap=heatmap,
        accuracy_plot=accuracy_plot,
        confusion_plot=confusion_plot,
        feature_plot=feature_plot,
    )

@app.route("/history")
def history():

    records = Analysis.query.order_by(Analysis.id.desc()).all()

    return render_template(
        "history.html",
        records=records
    )

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)
