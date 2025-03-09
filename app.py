import gradio as gr
import requests

# Create a Gradio interface
demo = gr.Interface(
    fn=lambda prompt: requests.post("http://localhost:8000/text-to-image/", json={"prompt": prompt}).json(),
    inputs="text",
    outputs="image",
    title="Text-to-Image Demo",
    description="Enter a prompt to generate an image",
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
