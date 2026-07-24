from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rembg import remove

from typing import List
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


# Process multiple images + addition
@app.post("/process")
async def process(
    images: List[UploadFile] = File(...),
    addition: str = Form(...)
):

    # -------------------------------
    # Allow maximum 10 images
    # -------------------------------

    if len(images) > 10:

        return JSONResponse(
            {
                "error": "Maximum 10 images are allowed."
            },
            status_code=400
        )


    # -------------------------------
    # Start timer
    # -------------------------------

    start_time = time.time()


    processed_images = []


    # -------------------------------
    # Process every uploaded image
    # -------------------------------

    for image in images:

        image_bytes = await image.read()


        # Remove background
        output_image = remove(image_bytes)


        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"


        output_path = os.path.join(
            "outputs",
            filename
        )


        # Save processed image
        with open(output_path, "wb") as file:
            file.write(output_image)


        # Save image path for frontend
        processed_images.append(
            f"/outputs/{filename}"
        )


    # -------------------------------
    # Stop timer
    # -------------------------------

    end_time = time.time()

    processing_time = end_time - start_time


    print(
        f"Processed {len(images)} image(s) in "
        f"{processing_time:.2f} seconds"
    )


    # -------------------------------
    # Solve addition
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
    # Send response
    # -------------------------------

    return JSONResponse(
        {
            "answer": answer,
            "images": processed_images,
            "processing_time": f"{processing_time:.2f} seconds",
            "total_images": len(images)
        }
    )