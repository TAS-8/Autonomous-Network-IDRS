import requests
import time
import random
import pandas as pd
API_URL = "http://localhost:8000/api/packet"

df = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")
def generate_packet():
    df_sample = df.sample(n=1).iloc[0]
    
    features = [x.item() if hasattr(x, 'item') else x for x in df_sample.tolist()]
    
    return {
        "features": features,
        "source_ip": f"192.168.1.{random.randint(1, 254)}"
    }

def main():
    while True:
        packet = generate_packet()
        try:
            response = requests.post(API_URL, json=packet)
            if response.ok:
                result = response.json()
                print(f"Sent packet from {packet['source_ip']} | Action: {result['action']} | Confidence: {result['confidence']:.2f}")
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Request failed: {e}")
        time.sleep(1)  

if __name__ == "__main__":
    main()
