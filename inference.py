import torch
import numpy as np
import joblib
from stable_baselines3 import PPO
from torch import nn

# --- ARCHITECTURE DEFINITION ---
class AnalystModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(AnalystModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class IntrusionSystem:
    def __init__(self):
        self.device = torch.device("cpu")
        print("Loading system components...")
        
        # 1. Load Scaler
        self.scaler = joblib.load("scaler.pkl")
        
        # 2. Load Analyst
        self.analyst = AnalystModel(input_shape=78, num_classes=6) 
        self.analyst.load_state_dict(torch.load("analyst_model_mlp.pth", map_location=self.device))
        self.analyst.to(self.device)
        self.analyst.eval()
        
        # 3. Load RL Agent
        self.agent = PPO.load("ppo_intrusion_response_agent.zip")
        print("System loaded successfully.")

    def predict_action(self, raw_features):
        """
        Takes raw packet features -> Returns Decision (String)
        """
        # A. Preprocessing
        # Reshape to 2D array because scaler expects [n_samples, n_features]
        raw_array = np.array(raw_features).reshape(1, -1)
        scaled_features = self.scaler.transform(raw_array)
        
        # B. Analyst Inference
        with torch.no_grad():
            tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            analyst_output = self.analyst(tensor_input)
            probs = torch.softmax(analyst_output, dim=1)
            
            # Calculate Confidence (Probability of Attack)
            # Assuming Class 0 is Normal, so Attack Prob = 1 - Prob(Normal)
            attack_confidence = 1.0 - probs[0][0].item()
            
        # C. Construct State for RL (Features + Confidence)
        rl_state = np.append(scaled_features[0], attack_confidence).astype(np.float32)
        
        # D. RL Agent Decision
        action, _ = self.agent.predict(rl_state, deterministic=True)
        
        # Map Action to English
        actions_map = {0: "ALLOW", 1: "BLOCK", 2: "THROTTLE"}
        return actions_map[int(action)], attack_confidence

# --- EXAMPLE USAGE ---
# system = IntrusionSystem()
# dummy_packet = [0.5, 1200, ...] # Needs to be 78 raw numbers
# decision, confidence = system.predict_action(dummy_packet)
# print(f"AI Decision: {decision} (Confidence: {confidence:.2f})")