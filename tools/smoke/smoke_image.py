import os, sys
from google import genai

c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = os.getenv("GEMINI_MODEL_IMAGE", "gemini-2.5-flash-image")
r = c.models.generate_images(model=model, prompt="cat icon, flat, 256x256")
b = r.images[0].data
assert isinstance(b, (bytes, bytearray)) and b[:4] == b"\x89PNG", "Expected PNG bytes"
print("OK image bytes:", len(b))
