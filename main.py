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

from random import randrange
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO

from imagetoimage import StableDiffusionInpaintingPipeline
NUM_SAMPLES = 4



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def inpaint(p, init_image, mask_image=None, strength=0.75, guidance_scale=7.5, generator=None, num_samples=3, n_iter=1):
    all_images = []
    for _ in range(n_iter):
        with autocast("cuda"):
            images = MODEL_IN(
                prompt=[p] * num_samples,
                init_image=init_image,
                mask_image=mask_image,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
            )["sample"]
        all_images.extend(images)
    return all_images

def get_inpainting_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionInpaintingPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token='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'
    ).to(device)
    return pipe

def get_model():

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    print(torch.cuda.is_available())

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token='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')
    pipe = pipe.to(device)
    return pipe

MODEL = get_model()
MODEL_IN = get_inpainting_model()

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
    text: str = 'Pixar Style, Tiny cute and scowling snow fox adventurer dressed in fantasy armour , jean - baptiste monge , anthropomorphic , dramatic lighting, 8k, painted portrait --testp'#"A solitary old tree of life in the forest asher brown durand greg rutkowski featured on artstation"


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

@app.post("/inpaiting")
def image_filter(request: Request, prompt: str = "a kungfu panda sitting on a bench",  img: UploadFile = File(description="Upload init image only images with the following MIME type: image/jpeg or image/png"), mask: UploadFile = File(description="Upload only mask image with the following MIME type: image/jpeg or image/png")):
    #prompt = 'a panda sitting on bunch'#'#user_request2.text
    allowedFiles = {"image/jpeg", "image/png"}
    if img.content_type and  mask.content_type in allowedFiles:
        print("ok")
        print(img.filename)
        print(img.file)
        original_image = Image.open(img.file).convert("RGB").resize((512, 512), Image.ANTIALIAS)
        mask_image = Image.open(mask.file).convert("RGB").resize((512, 512), Image.ANTIALIAS)
        print("a panda sitting on bunch")
        generator = torch.Generator(device="cuda").manual_seed(randrange(1000))
        images = inpaint(p=prompt, mask_image=mask_image, init_image=original_image, generator=generator)

        [image_file.save(os.path.join('static/inpating', f"{i}.png")) for i, image_file in enumerate(images)]
        file_paths = os.listdir('static/inpating')
        print(file_paths)
        file_paths = ['inpating/' + image for image in file_paths]
        img_urls = [request.url_for('static', path=path_) for path_ in file_paths]
        keys = [f"img_path_{i}" for i in range(len(img_urls))]
        return {k: v for k, v in zip(keys, img_urls)}  # {"img_path": img_url}
    return {"img_path": None, "allowedFiles": ["image/jpeg", "image/png"]}


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

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)


