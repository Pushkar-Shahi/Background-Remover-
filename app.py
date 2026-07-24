from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rembg import remove

import os
import uuid
import time


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

    # Read uploaded image
    image_bytes = await image.read()


    # -------------------------------
    # Start measuring AI processing time
    # -------------------------------

    start_time = time.time()


    # Remove background using rembg
    output_image = remove(image_bytes)


    # Stop timer
    end_time = time.time()


    # Calculate total processing time
    processing_time = end_time - start_time


    print(
        f"Background removal took {processing_time:.2f} seconds"
    )


    # -------------------------------
    # Save processed image
    # -------------------------------

    filename = f"{uuid.uuid4()}.png"


    output_path = os.path.join(
        "outputs",
        filename
    )


    with open(output_path, "wb") as file:
        file.write(output_image)



    # -------------------------------
    # Solve addition problem
    # -------------------------------

    try:

        numbers = addition.split("+")


        answer = sum(
            int(num.strip())
            for num in numbers
        )


    except:

        answer = "Invalid addition format"



    # -------------------------------
    # Send response back to browser
    # -------------------------------

    return JSONResponse(
        {
            "answer": answer,
            "image": f"/outputs/{filename}",
            "processing_time": f"{processing_time:.2f} seconds"
        }
    )