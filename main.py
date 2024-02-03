from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse,  JSONResponse
import replicate
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

load_dotenv()
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-music")
async def generate_music(prompt: str = Form(...), duration: int = Form(...)):
    output = replicate.run(
        "meta/musicgen:b05b1dff1d8c6dc63d14b0cdb42135378dcb87f6373b0d3d341ede46e59e2b38",
        input={
            "top_k": 250,
            "top_p": 0,
            "prompt": prompt,
            "duration": duration,
            "temperature": 1,
            "continuation": False,
            "model_version": "stereo-melody-large",
            "output_format": "wav",
            "continuation_start": 0,
            "multi_band_diffusion": False,
            "normalization_strategy": "peak",
            "classifier_free_guidance": 3
        }
    )
    music_path_or_url = output
    return JSONResponse(content={"url": music_path_or_url})

