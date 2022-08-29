import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import uvicorn
from PIL import Image
from os import getcwd
import sys
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import FastAPI, Request


NUM_SAMPLES = 4


def get_model():

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    print(torch.cuda.is_available())

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token='')
    pipe = pipe.to(device)
    return pipe

MODEL = get_model()


app = FastAPI(
    title="ML Model",
    description="Stable Diffusion",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class UserRequestIn(BaseModel):
    text: str = "A solitary old tree of life in the forest asher brown durand greg rutkowski featured on artstation"


@app.post('/predict', tags=["predictions"])
def get_result(request: Request, user_request: UserRequestIn):
    prompt = user_request.text
    prompt = [prompt] * NUM_SAMPLES
    if prompt:
        with autocast("cuda"):
            images = MODEL(prompt)["sample"]  # [0]

        [image_file.save(os.path.join('static/results', f"{i}.png")) for i, image_file in enumerate(images)]
        file_paths = os.listdir('static/results')
        print(file_paths)
        file_paths = ['results/' + image for image in file_paths]
        img_urls = [request.url_for('static', path=path_) for path_ in file_paths]
        keys = [f"img_path_{i}" for i in range(len(img_urls))]
        return {k: v for k,v in zip(keys, img_urls)}#{"img_path": img_url}
    return {"img_path": None}


@app.get("/")
def hello():
    return {"message": "Go to /docs"}


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
