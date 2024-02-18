from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse,  JSONResponse
import replicate
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI

app = FastAPI()

load_dotenv()
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def generate_lyrics(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a music lyrics writer and your task is to write lyrics of music under 30 words based on user's prompt. Just return the lyrics and nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.choices[0].message.content
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
