import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS
import numpy as np
import shap
import traceback
import matplotlib.pyplot as plt
import os
import uuid

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Device configuration for CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "Pulk17/Fake-News-Detection"
model = BertForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model prediction function
def model_predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probas = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probas.cpu().numpy()

# Explainability function using SHAP
def explain_prediction(text):
    def prediction_function(inputs):
        tokenized = tokenizer(list(inputs), return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = model(**tokenized)
            return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

    explainer = shap.Explainer(prediction_function, tokenizer)
    shap_values = explainer([text])
    return shap_values

# Align tokens with SHAP values
def align_tokens_and_shap_values(tokens, shap_values):
    aligned_tokens = []
    aligned_shap_values = []

    current_token = ""
    current_value = 0.0

    for token, value in zip(tokens, shap_values):
        if token in {"[CLS]", "[SEP]"}:  # Skip special tokens
            continue
        if token.startswith("##"):
            current_token += token[2:]  # Append subword without '##'
            current_value += value
        else:
            if current_token:
                aligned_tokens.append(current_token)
                aligned_shap_values.append(current_value)
            current_token = token
            current_value = value

    if current_token:
        aligned_tokens.append(current_token)
        aligned_shap_values.append(current_value)

    return aligned_tokens, aligned_shap_values

# Summarize SHAP explanations with advanced formatting
def summarize_explanation(tokens, shap_values, top_n=5, predicted_class=0):
    token_contributions = list(zip(tokens, shap_values))
    sorted_contributions = sorted(token_contributions, key=lambda x: abs(x[1]), reverse=True)

    supporting_contributions = [(token, value) for token, value in sorted_contributions if (value > 0 if predicted_class == 0 else value < 0)]

    def format_token(token, value, predicted_class):
        color = "green" if predicted_class == 0 else "red"
        return f"<b>{token}</b> (<span style='color:{color}'>{abs(value) * 100:.2f}%</span>)"

    supporting_tokens = [format_token(token, value, predicted_class) for token, value in supporting_contributions[:top_n]]

    total_supporting = sum(value for _, value in supporting_contributions) * 100

    explanation = (
        f"<p>The words such as {', '.join(supporting_tokens)} enhance the credibility of the headline, "
        f"suggesting it is likely { 'real' if predicted_class == 0 else 'fake'}. The total contribution is <b>{total_supporting:.2f}%</b>.</p>"
    )

    return explanation

# Plot SHAP values for smaller and larger texts
def plot_shap_values(tokens, explanation, output_path="static/shap_plots", is_large=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    unique_filename = f"shap_plot_{uuid.uuid4().hex}.png"
    full_path = os.path.join(output_path, unique_filename)

    plt.figure(figsize=(12, 8) if is_large else (10, 6))
    if is_large:
        plt.scatter(tokens, explanation, color=["green" if value > 0 else "red" for value in explanation], edgecolor="black")
    else:
        plt.bar(tokens, explanation, color=["green" if value > 0 else "red" for value in explanation], edgecolor="black")
    
    plt.xlabel("Tokens", fontsize=14)
    plt.ylabel("SHAP Value", fontsize=14)
    plt.title("Token Contributions to Prediction", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()

    return unique_filename

# Analyze endpoint
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        probas = model_predict(text)
        predicted_class = np.argmax(probas, axis=1).item()
        confidence = probas[0][predicted_class] * 100
        label = "Real" if predicted_class == 0 else "Fake"

        shap_values = explain_prediction(text)
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
        )
        explanation_values = shap_values[0].values.flatten().tolist()

        aligned_tokens, aligned_shap_values = align_tokens_and_shap_values(tokens, explanation_values)

        is_headline = len(text.split()) <= 15

        if confidence < 99.0 and np.abs(np.sum(explanation_values)) < 1e-5:
            return jsonify({
                "label": label,
                "confidence": f"{confidence:.2f}%",
                "message": f"The headline is likely {label} with a confidence of {confidence:.2f}%. "
                           "For a more accurate analysis, please provide the full article or news content.",
                "redirect": True
            })

        textual_explanation = summarize_explanation(aligned_tokens, aligned_shap_values, predicted_class=predicted_class)
        is_large = len(text.split()) > 30
        plot_filename = plot_shap_values(aligned_tokens, aligned_shap_values, is_large=is_large)

        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2f}%",
            "textual_explanation": textual_explanation,
            "plot_path": f"/static/shap_plots/{plot_filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500

# Home endpoint
@app.route("/")
def home():
    return "Explainable Fake News Detection API"

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
