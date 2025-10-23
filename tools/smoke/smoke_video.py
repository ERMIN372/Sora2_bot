import os, json, time
from google import genai

c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = os.getenv("GEMINI_MODEL_VIDEO", "veo-3.1-generate-preview")
op = c.models.generate_videos(model=model, prompt="a cat dancing, 6 seconds")
# Если SDK возвращает long-running operation: опросить 2-3 раза
for i in range(6):
    if getattr(op, "done", False):
        break
    time.sleep(5)
    op = c.operations.get(op)
print("OK video op:", getattr(op, "done", False), "fields:", [k for k in dir(op) if not k.startswith("_")][:8])
