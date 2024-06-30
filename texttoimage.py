import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4" # Load the Stable Diffusion model from hugging face
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
def generate_image_from_text(prompt):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image
def on_generate():
    prompt = entry_text.get()
    if prompt:
        image = generate_image_from_text(prompt)
        image.thumbnail((400, 400), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image  # Keep a reference to avoid garbage collection

root = tk.Tk()
root.title("Text to Image Generator")
tk.Label(root, text="Enter text prompt:").grid(row=0, column=0, padx=10, pady=10)
entry_text = tk.Entry(root, width=50)
entry_text.grid(row=0, column=1, padx=10, pady=10)

generate_button = ttk.Button(root, text="Generate Image", command=on_generate)
generate_button.grid(row=1, columnspan=2, padx=10, pady=10)

image_label = tk.Label(root)
image_label.grid(row=2, columnspan=2, padx=10, pady=10)

root.mainloop()
