from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rembg import remove
import os
import uuid


app = FastAPI()


# Create required folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Allow browser to access processed images
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)


# HTML template location
templates = Jinja2Templates(
    directory="templates"
)


# Home page
@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# Process image + addition
@app.post("/process")
async def process(
    image: UploadFile = File(...),
    addition: str = Form(...)
):

    # Read image bytes
    image_bytes = await image.read()


    # Remove background using rembg
    output_image = remove(image_bytes)


    # Save output image
    filename = f"{uuid.uuid4()}.png"

    output_path = os.path.join(
        "outputs",
        filename
    )

    with open(output_path, "wb") as file:
        file.write(output_image)


    # Solve addition
    try:

        numbers = addition.split("+")

        answer = sum(
            int(num.strip())
            for num in numbers
        )

    except:

        answer = "Invalid addition format"


    return JSONResponse(
        {
            "answer": answer,
            "image": f"/outputs/{filename}"
        }
    )



