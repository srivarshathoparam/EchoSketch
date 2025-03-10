import gradio as gr
import os
import torch
from faster_whisper import WhisperModel
from diffusers import StableDiffusionPipeline
import time  # For simulating progress

def process_audio(audio_path):
    # Check if file exists
    if not os.path.exists(audio_path):
        raise ValueError(f"Error: The file '{audio_path}' does not exist or is not accessible.")

    print(f"Processing audio file: {audio_path}")  # Debugging print

    # Load Whisper model
    model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float32")

    # Transcribe audio
    segments, _ = model.transcribe(audio_path)
    transcription = " ".join(segment.text for segment in segments)

    return transcription

def generate_image(prompt, progress=gr.Progress()):
    # Check if CUDA is available for GPU usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Stable Diffusion model with appropriate dtype
    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Move model to the correct device (GPU or CPU)
    pipe.to(device)

    print(f"Generating image for prompt: {prompt}")  # Debugging print

    # Simulate some processing time and update progress bar
    total_steps = 10  # Total number of steps in the progress
    for i in range(total_steps):  # Adjust number of steps as needed
        time.sleep(0.5)  # Simulate work (you can adjust this for actual processing time)
        progress(i + 1, f"Step {i+1} of {total_steps}")  # Update progress with current step and description

    # Generate image
    image = pipe(prompt).images[0]
    image_path = "generated_image.png"
    image.save(image_path)

    return image_path

def audio_to_image(audio_file):
    print(f"Received audio file: {audio_file}")  # Debugging print

    transcription = process_audio(audio_file)
    return transcription, generate_image(transcription)

iface = gr.Interface(
    fn=audio_to_image,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Transcription"), gr.Image(label="Generated Image")],
    title="Audio2Art: Convert Speech to AI Art",
    description="Upload an audio file with a description of an image, and get an AI-generated image based on the transcription.",
    live=True  # Enable live updates, if needed
)

if __name__ == "__main__":
    iface.launch(share=True)
