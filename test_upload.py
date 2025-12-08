import base64
import json
import requests

with open("bottle.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

payload = {"image_base64": encoded}

resp = requests.post(
    "http://submuscularly-tapestried-london.ngrok-free.dev/user/spencer/favorite/from-photo",
    json=payload,
)

print("Status:", resp.status_code)
print(resp.text)