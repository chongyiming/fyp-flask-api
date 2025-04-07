from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.signal import butter, filtfilt
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# ======== Model Definition ============
class ECG_CNN_Combined(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = x.unsqueeze(1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ======== Filters =====================
def highpass_filter(signal, cutoff=0.5, fs=100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

def moving_average_filter(signal, window_size=3):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# ======== ECG Loader ==================
def load_ecg_data(file_path, num_leads=12, samples_per_lead=1000, fs=100):
    data = np.fromfile(file_path, dtype=np.int16)
    if len(data) != num_leads * samples_per_lead:
        raise ValueError(f"Expected {num_leads * samples_per_lead} points, got {len(data)}.")

    leads_data = np.zeros((num_leads, samples_per_lead))
    for i in range(len(data)):
        leads_data[i % num_leads, i // num_leads] = data[i]

    filtered_leads = np.zeros_like(leads_data)
    for i in range(num_leads):
        filtered = highpass_filter(leads_data[i], fs=fs)
        filtered = moving_average_filter(filtered)
        filtered_leads[i] = filtered

    return filtered_leads

# ======== Load Model ==================
model = ECG_CNN_Combined()
model.load_state_dict(torch.load("ecg_cnn_combined.pth", map_location=torch.device("cpu")))
model.eval()

# ======== API Endpoint ================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = file.filename
    file.save(file_path)

    try:
        ecg_data = load_ecg_data(file_path)  # (12, 1000)
        input_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][predicted].item()

        norm_pred = 0
        mi_pred = 1
        if int(predicted) == 0:
            norm_prob = round(confidence, 2)
            mi_prob = round(1 - confidence, 2)
        else:
            mi_prob = round(confidence, 2)
            norm_prob = round(1 - confidence, 2)
        return jsonify({'prediction': int(predicted), 'confidence': float(confidence),'norm_prob':norm_prob*100,'mi_prob':mi_prob*100})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)

# ========== Run App ====================
if __name__ == '__main__':
    app.run(debug=True)
