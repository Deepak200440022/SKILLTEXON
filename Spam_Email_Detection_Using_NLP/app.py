from flask import Flask, render_template, request
import joblib

# Load saved model and vectorizer
model = joblib.load("spam_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def classify():
    message = ""
    result = None

    if request.method == "POST":
        # Handle file input or direct message
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename.endswith(".txt"):
            message = uploaded_file.read().decode("utf-8").strip()
        else:
            message = request.form.get("message", "").strip()

        if message:
            # Transform and classify
            transformed = vectorizer.transform([message])
            prediction = model.predict(transformed)[0]
            result = "Spam" if prediction == 1 else "Ham"

    return render_template("index.html", message=message, result=result)

if __name__ == "__main__":
    app.run(debug=True)
