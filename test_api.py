import pandas as pd
import requests
# Load dataset
df = pd.read_csv("archive/mitbih_test.csv", header=None)
sample = df.iloc[0, :-1].tolist()
payload = {"signal": sample}
response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
