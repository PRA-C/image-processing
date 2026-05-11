import os
import base64
import ollama
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import config

# ── Helpers ────────────────────────────────────────────────────
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB").resize(config.IMAGE_SIZE)

# ── Step 1: Understand image with Ollama ───────────────────────
def describe_image(image_path: str) -> str:
    print("\n[1/3] Analyzing image with Ollama...")
    image_b64 = encode_image(image_path)

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": "Describe this image in detail for an AI image editing prompt.",
            "images": [image_b64]
        }]
    )
    description = response["message"]["content"]
    print(f"     Description: {description[:100]}...")
    return description

# ── Step 2: Build edit instruction using Ollama ────────────────
def build_edit_prompt(description: str, user_instruction: str) -> str:
    print("\n[2/3] Building edit prompt with Ollama...")

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": f"""
            Image description: {description}
            User wants to: {user_instruction}

            Create a clear, concise image editing instruction (max 20 words).
            Return ONLY the instruction, nothing else.
            Example: ""make background blue and add snow fall"
            """
        }]
    )
    prompt = response["message"]["content"].strip()
    print(f"     Edit prompt: {prompt}")
    return prompt

# ── Step 3: Modify image using diffusion model ─────────────────
def load_pipeline():
    print("\n[Loading] Downloading diffusion model (first run only)...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        config.DIFFUSION_MODEL,
        torch_dtype=config.DTYPE,
        safety_checker=None,
        cache_dir=config.MODEL_CACHE_DIR
    ).to(config.DEVICE)
    print("[Loading] Model ready!")
    return pipe

def modify_image(pipe, image_path: str, edit_prompt: str, output_name: str):
    print("\n[3/3] Applying modification...")

    image = load_image(image_path)

    result = pipe(
        prompt=edit_prompt,
        image=image,
        num_inference_steps=config.INFERENCE_STEPS,
        image_guidance_scale=config.IMAGE_GUIDANCE,
        guidance_scale=config.GUIDANCE_SCALE,
    ).images[0]

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, output_name)
    result.save(output_path)
    print(f"\n✅ Done! Saved to: {output_path}")
    return output_path

# ── Main ───────────────────────────────────────────────────────
def run(image_filename: str, instruction: str):
    image_path = os.path.join(config.INPUT_DIR, image_filename)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Load pipeline once
    pipe = load_pipeline()

    # Run pipeline
    description  = describe_image(image_path)
    edit_prompt  = build_edit_prompt(description, instruction)
    output_name  = f"modified_{image_filename}"

    modify_image(pipe, image_path, edit_prompt, output_name)


if __name__ == "__main__":
    # ── Configure your edit here ───────────────────────────────
    IMAGE_FILE  = "IMG_8683.jpg"          # file inside input_images/
    INSTRUCTION = "make background blue and add snow fall"

    run(IMAGE_FILE, INSTRUCTION)