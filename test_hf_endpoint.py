import requests

HF_TOKEN = "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"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
print(response.status_code)
print(response.json())
