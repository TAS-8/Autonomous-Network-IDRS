# NetSense: Autonomous Network Intrusion Detection & Response System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.104.1-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Stable%20Baselines3-2.1.0-00599c?logo=github" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-f7931e?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/uvicorn-0.24.0-222?logo=uvicorn" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
</p>

NetSense is a modern, AI-powered Intrusion Detection and Response System (IDRS) designed to monitor network traffic, detect malicious activity, and autonomously respond to threats in real time. It features a web dashboard for live monitoring and control, and leverages deep learning and reinforcement learning for intelligent decision-making.

---

## Features
- **Real-time Intrusion Detection** using a trained neural network (PyTorch)
- **Autonomous Response** with a reinforcement learning agent (Stable Baselines3 PPO)
- **Live Web Dashboard** for monitoring stats, traffic, and recent events
- **REST API** for integration and automation
- **Easy extensibility** for research and production

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/TAS-8/Autonomous-Network-IDRS.git
cd Autonomous-Network-IDRS
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Model Files
Ensure the following files are present in the project root:
- `analyst_model_mlp.pth` (PyTorch model)
- `scaler.pkl` (feature scaler)
- `ppo_response_agent.zip` (RL agent)

> **Note:** If you need to retrain, use `training_script.py`.

### 4. Run the Server
```bash
uvicorn app:app --reload
```

### 5. Open the Dashboard
Visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## Project Structure
```
├── app.py                  # FastAPI backend (API + static server)
├── inference.py            # Model inference and RL agent
├── client_send_packets.py  # Example client for sending packets
├── training_script.py      # Model training pipeline
├── requirements.txt        # Python dependencies
├── public/                 # Frontend (HTML, CSS, JS)
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── ...
```

---

## API Endpoints
- `POST /api/packet` — Analyze a packet (features as JSON)
- `GET /api/stats` — Get live statistics
- `GET /api/logs` — Recent detection logs
- `POST /api/control/start|stop|inject` — Control endpoints

See `public/app.js` for usage examples.

---

## How It Works
- **Detection:** Extracted features from network packets are scaled and passed to a neural network classifier.
- **Response:** The RL agent decides to ALLOW, BLOCK, or THROTTLE based on the detection confidence and traffic context.
- **Dashboard:** Live stats, traffic charts, and logs are updated in real time via REST API.

---

## Customization & Training
- To retrain or fine-tune models, use `training_script.py` and follow the comments inside.
- Update the frontend in `public/` as needed for your use case.

---

## License
This project is for academic and research purposes. For commercial use, please contact the author.

---

## Acknowledgements
- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- PyTorch, FastAPI, Stable Baselines3
