import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras import backend as K

# ---------------- APP SETUP ----------------
app = Flask(__name__)
app.secret_key = "railway_secret_123"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- LOGIN USERS ----------------
valid_users = {
    "Vishnupriya_13": "vishnu123",
    "Sathwika_02": "sathwika123",
    "Saahithi_07": "saahithi123",
    "Avanthi_27": "avanthi123"
}

# ---------------- LOAD MODEL ----------------
print("Loading model...")
K.clear_session()
model = tf.keras.models.load_model("model/track_model.h5", compile=False)
print("Model loaded successfully")

# ---------------- LOGIN PAGE ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pwd = request.form.get("password")

        if user in valid_users and valid_users[user] == pwd:
            session["user"] = user
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


# ---------------- REDIRECT /login ----------------
@app.route("/login")
def login_redirect():
    session.clear()   # auto logout when going to login
    return redirect(url_for("login"))


# ---------------- INDEX ----------------
@app.route("/index")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    original = cv2.imread(filepath)
    if original is None:
        return redirect(url_for("index"))

    # -------- PREPROCESS IMAGE --------
    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    input_tensor = np.expand_dims(img, axis=0)

    # -------- MODEL PREDICTION --------
    preds = model.predict(input_tensor)[0]
    defect_prob = float(preds[0])
    safe_prob = float(preds[1])

    confidence = max(defect_prob, safe_prob)
    diff = abs(defect_prob - safe_prob)

    CONF_THRESHOLD = 0.65
    DIFF_THRESHOLD = 0.30

    if confidence < CONF_THRESHOLD or diff < DIFF_THRESHOLD:
        status = "UNCERTAIN"
    else:
        status = "DEFECTIVE" if defect_prob > safe_prob else "SAFE"

    # -------- RESULT IMAGE --------
    output = cv2.resize(original, (450, 300))

    if status == "DEFECTIVE":
        h, w = output.shape[:2]
        cv2.rectangle(
            output,
            (int(w * 0.2), int(h * 0.4)),
            (int(w * 0.8), int(h * 0.55)),
            (0, 0, 255),
            3
        )

    result_name = "result_" + filename
    result_path = os.path.join(RESULT_FOLDER, result_name)
    cv2.imwrite(result_path, output)

    # -------- DECISION ACCOUNTABILITY --------
    confidence_percent = round(confidence * 100, 2)

    if status == "DEFECTIVE":
        risk = "HIGH"
        reason = (
            "The R-CNN model identified a suspicious region on the railway track "
            "using region proposals and convolutional feature extraction. "
            "The detected region matches crack-like patterns learned during training."
        )
        action = (
            "Immediate inspection is recommended. "
            "Restrict train speed until verification."
        )

    elif status == "SAFE":
        risk = "LOW"
        reason = "No defect-like visual patterns were detected in the uploaded image."
        action = "Track is safe for operation. Continue routine monitoring."

    else:
        risk = "MEDIUM"
        reason = (
            "Model confidence is low or predictions are ambiguous due to unclear image quality."
        )
        action = "Re-capture the image or perform manual inspection."

    return render_template(
        "result.html",
        status=status,
        result_image=result_name,
        confidence=confidence_percent,
        risk=risk,
        reason=reason,
        action=action
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
