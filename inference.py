import torch
import numpy as np
import joblib
from stable_baselines3 import PPO
from torch import nn

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


# -------------------------------
# INTRUSION DETECTION + RESPONSE SYSTEM
# -------------------------------
class IntrusionSystem:
    def __init__(self):
        self.device = torch.device("cpu")
        print("🔹 Loading Intrusion Detection System...")

        # ===============================
        # HARD CONSTRAINTS (FROM TRAINING)
        # ===============================
        self.INPUT_SHAPE = 78        # Number of features after preprocessing
        self.NUM_CLASSES = 7         # Number of consolidated classes
        self.NORMAL_CLASS_IDX = 4    # <<< CRITICAL: Normal = 4 (from LabelEncoder)

        # 1️⃣ Load scaler
        self.scaler = joblib.load("scaler.pkl")

        # 2️⃣ Load Analyst model
        self.analyst = AnalystModel(
            input_shape=self.INPUT_SHAPE,
            num_classes=self.NUM_CLASSES
        )
        self.analyst.load_state_dict(
            torch.load("analyst_model_mlp.pth", map_location=self.device)
        )
        self.analyst.to(self.device)
        self.analyst.eval()

        # 3️⃣ Load RL Response Agent
        self.agent = PPO.load(
            "ppo_response_agent.zip",
            device=self.device
        )

        print("✅ System loaded successfully.")

    # -------------------------------
    # PREDICT ACTION
    # -------------------------------
    def predict_action(self, raw_features):
        if len(raw_features) != self.INPUT_SHAPE:
            raise ValueError(
                f"Expected {self.INPUT_SHAPE} features, got {len(raw_features)}"
            )

        raw_array = np.array(raw_features, dtype=np.float32).reshape(1, -1)
        scaled_features = self.scaler.transform(raw_array)

        with torch.no_grad():
            tensor_input = torch.tensor(
                scaled_features,
                dtype=torch.float32
            ).to(self.device)

            logits = self.analyst(tensor_input)
            probs = torch.softmax(logits, dim=1)

            prob_normal = probs[0][self.NORMAL_CLASS_IDX].item()
            attack_confidence = float(np.clip(1.0 - prob_normal, 0.01, 0.99))

        rl_state = np.append(
            scaled_features[0],
            attack_confidence
        ).astype(np.float32)

        rl_state = rl_state.reshape(1, -1)

        action, _ = self.agent.predict(
            rl_state,
            deterministic=True
        )

        action_map = {
            0: "ALLOW",
            1: "BLOCK",
            2: "THROTTLE"
        }

        return action_map[int(action)], attack_confidence
