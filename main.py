import os
import uuid
import random
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from urllib.request import urlretrieve
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
from ultralytics import YOLO

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not API_KEY:
    raise EnvironmentError("Missing HUGGINGFACEHUB_API_TOKEN in environment.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = API_KEY

# Initialize YOLO model
best_model = YOLO("best_yolov8_model.pt")  # Use relative path for deployment

# FastAPI app setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Mount static folder for audio
if not os.path.exists("audio"):
    os.makedirs("audio")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# Pydantic input model
class ImageURL(BaseModel):
    url: str

# Function to classify image
def classify_piece(image_path):
    results = best_model(image_path)
    for result in results:
        class_ids = result.boxes.cls
        class_labels = [best_model.names[int(cls)] for cls in class_ids]
        final_output = class_labels[0]
        if final_output == "Green Head":
            return "The Berlin Green Head is an ancient Egyptian statue head"
        return final_output

# Generate short fact using HuggingFace LLM
def generate_facts(item):
    template = """
    You are a Tour Guide That tells some facts about the {item}.
    Create a short, engaging, and keyword-rich story for {item} based on the provided context.
    Strictly adhere to 10–20 words.
    """
    prompt = PromptTemplate(template=template, input_variables=["item"])
    description_llm = LLMChain(
        llm=HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                           model_kwargs={"temperature": random.uniform(0.9, 1), "max_length": 128}),
        prompt=prompt
    )
    description = description_llm.predict(item=item)
    return description.split('\n')[-1].strip()

# Text-to-speech function
def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    # Save MP3 audio
    mp3_file_path = "audio/audio.mp3"
    with open(mp3_file_path, "wb") as f:
        f.write(response.content)


# Endpoint: Upload image by URL
@app.post("/museum/url/")
async def night_at_the_museum_url(data: ImageURL):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        urlretrieve(data.url, temp_filename)

        item = classify_piece(temp_filename)
        fact = generate_facts(item)
        text_to_speech(fact)

        os.remove(temp_filename)

        return {
            "artifact": item,
            "fact": fact,
            "audio_path": "audio/audio.mp3"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))