from fastapi import FastAPI
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe.to(device)

@app.post("/text-to-image/")
async def text_to_image(prompt: str):
    try:
        # Generate an image using the Stable Diffusion model
        image = pipe(prompt).images[0]

        # Save the image to a file
        image.save("generated_image.png")

        # Return the generated image
        return {"image": "generated_image.png"}
    except Exception as e:
        return {"error": str(e)}
