from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def classify_email():
    if request.method == "POST":
        # Handle text input or file upload
        if "file" in request.files:
            file = request.files["file"]
            email_content = file.read().decode("utf-8")
        else:
            email_content = request.form["email"]

        # Transform email content
        email_vector = vectorizer.transform([email_content])
        prediction = model.predict(email_vector)[0]
        confidence = model.predict_proba(email_vector).max() * 100
        result = "Spam" if prediction == 1 else "Not Spam"

        return render_template("index.html", result=result, confidence=confidence)
    return render_template("index.html")

@app.route("/batch", methods=["POST"])
def classify_batch():
    # Endpoint for batch processing
    email_texts = request.json["emails"]  # Expecting a list of emails
    email_vectors = vectorizer.transform(email_texts)
    predictions = model.predict(email_vectors)
    confidences = model.predict_proba(email_vectors).max(axis=1) * 100
    results = [{"email": email, "result": "Spam" if pred == 1 else "Not Spam", "confidence": conf}
               for email, pred, conf in zip(email_texts, predictions, confidences)]
    return {"results": results}

if __name__ == "__main__":
    app.run(debug=True)
