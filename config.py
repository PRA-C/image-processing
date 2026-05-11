import os

# ── Ollama Settings ────────────────────────────────────────────
OLLAMA_MODEL = "moondream:latest"   # vision model for understanding
OLLAMA_HOST  = "http://localhost:11434"

# ── Diffusion Model Settings ───────────────────────────────────
# Options: "timbrooks/instruct-pix2pix"  (best for edits)
#          "runwayml/stable-diffusion-v1-5"  (img2img)
DIFFUSION_MODEL = "timbrooks/instruct-pix2pix"

# Local cache folder so models don't re-download
MODEL_CACHE_DIR = "./models"

# ── Image Settings ─────────────────────────────────────────────
INPUT_DIR  = "./input_images"
OUTPUT_DIR = "./output_images"
IMAGE_SIZE = (512, 512)

# ── Generation Settings ────────────────────────────────────────
# 0.0 = keep original, 1.0 = fully regenerate
STRENGTH         = 0.7
GUIDANCE_SCALE   = 7.5
IMAGE_GUIDANCE   = 1.5   # only for pix2pix
INFERENCE_STEPS  = 30

# ── Device ────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[Config] Using device: {DEVICE}")