from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables
load_dotenv()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Function to generate lyrics using Hugging Face's GPT-NEO model
def generate_lyrics(prompt):
    # Initialize text generation pipeline with GPT-NEO model
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    # Generate lyrics based on the prompt
    response = generator(prompt, max_length=50, temperature=0.7, do_sample=True)
    # Extract generated text from response
    output = response[0]['generated_text']
    # Format the generated lyrics
    cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {cleaned_output} ♪"
    return formatted_lyrics

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-music")
async def generate_music(prompt: str = Form(...), duration: int = Form(...)):
    lyrics = generate_lyrics(prompt)
    prompt_with_lyrics = lyrics
    print(prompt_with_lyrics)
    output = replicate.run(
        "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
        input={
            "prompt": prompt_with_lyrics,
            "text_temp": 0.7,
            "output_full": False,
            "waveform_temp": 0.7
        }
    )
    print(output)
    music_url = output['audio_out']
    music_path_or_url = music_url
    
    print(music_path_or_url)
    return JSONResponse(content={"url": music_path_or_url})