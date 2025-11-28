from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained vectorizer and model
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")


def is_text_too_short(text: str) -> bool:
    """
    Simple check: is this message too short to classify?
    - empty text
    - fewer than 3 words
    """
    if not text:
        return True

    # split into words and ignore pure punctuation
    words = [w for w in text.strip().split() if any(c.isalnum() for c in w)]
    return len(words) < 3


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    # 🔴 New: guard for very short / meaningless input
    if is_text_too_short(text):
        return jsonify({
            "error": "Please type a full support message (at least a short sentence)."
        })

    # Normal prediction path
    X_new = vectorizer.transform([text])
    pred_label = model.predict(X_new)[0]

    return jsonify({"label": pred_label})


if __name__ == "__main__":
    app.run(debug=True)
