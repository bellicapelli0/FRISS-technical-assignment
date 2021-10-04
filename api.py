import json
import os
import pandas as pd
import torch
import numpy as np

from claim_classifier import ClaimClassifier

from flask import Flask, request

PREDICTION_FILE = "./test_predictions.csv"
MODEL_FILE = "./least_bad_model.mdl"
NUMBER_OF_FEATURES = 23

app = Flask(__name__)

df = pd.read_csv(PREDICTION_FILE)
df["ID"] = df["ID"].astype(str)

model = ClaimClassifier(NUMBER_OF_FEATURES)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

softmax = torch.nn.Softmax(dim=0)

@app.route('/score/<Line>', methods=["GET"])
def score(Line):
    sub_df = df[df["ID"] == Line]["prediction"]
    if len(sub_df) == 1:
        score = sub_df[0]
        response = {"score":score}
        return response
    else:
        return "could not find claimid\n"
    
@app.route('/predict/', methods=["POST"])
def predict():
    content = request.json
    ID = content["id"]
    x = torch.tensor(content["features"])
    
    with torch.no_grad():
        logits = model(x)
    probs = softmax(logits).detach().cpu().numpy()
    pred = int(np.argmax(logits))

    r = { "prediction" : pred,
         "score" : float(probs[pred])
        }
    
    return r
    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')