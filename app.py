# ============================================================
# ONNX RUNTIME / CUDA
# ============================================================

import onnxruntime as ort

# Load CUDA/cuDNN DLLs before importing rembg
ort.preload_dlls()

print("ONNX Runtime providers:")
print(ort.get_available_providers())


# ============================================================
# FASTAPI
# ============================================================

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ============================================================
# REMBG
# ============================================================

from rembg import remove, new_session


# ============================================================
# OTHER IMPORTS
# ============================================================

from typing import List

import os
import uuid
import time
import cv2
import subprocess
import shutil


# ============================================================
# APP
# ============================================================

app = FastAPI()


# ============================================================
# FOLDERS
# ============================================================

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)


# ============================================================
# STATIC OUTPUTS
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# GPU PROVIDERS
# ============================================================

providers = ort.get_available_providers()

print("Available providers:", providers)


if "CUDAExecutionProvider" in providers:

    print("CUDA GPU acceleration: ENABLED")

    REMBG_PROVIDERS = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

else:

    print(
        "WARNING: CUDAExecutionProvider is NOT available."
    )

    print(
        "Using CPU instead."
    )

    REMBG_PROVIDERS = [
        "CPUExecutionProvider"
    ]


# ============================================================
# REMBG TUNING
#
# This is what actually controls "cutting into the character":
#
# rembg's alpha matting builds a trimap by marking pixels above
# alpha_matting_foreground_threshold as sure-foreground and pixels
# below alpha_matting_background_threshold as sure-background, then
# ERODES both of those regions with a square
# alpha_matting_erode_size x alpha_matting_erode_size kernel. Only
# what's left over becomes the "unknown" band that the matting
# solver actually gets to refine.
#
# At the rembg default of erode_size=10, any foreground detail
# narrower than ~10px - fingers, hair strands, thin straps - gets
# wiped out of "sure foreground" completely and dumped into the
# unknown band, where the solver is free to decide it's background
# if there isn't strong colour contrast there. That's the usual
# cause of a mask that eats into the subject, not a bug in the
# code. Lowering erode_size, and being a little more generous with
# foreground_threshold, keeps more of the character "confident"
# before erosion ever runs.
# ============================================================

# Model. Swap this string to try a different one, no other code
# changes needed:
#
#   "u2net"              - previous default, general purpose
#   "isnet-general-use"  - sharper general-purpose upgrade over u2net (new default below)
#   "u2net_human_seg"    - tuned specifically for people
#   "isnet-anime"        - tuned specifically for anime / illustrated characters
#   "birefnet-general"   - highest precision, noticeably slower/heavier
#   "birefnet-portrait"  - highest precision, people specifically
REMBG_MODEL = "isnet-general-use"

ALPHA_MATTING_ERODE_SIZE = 5
ALPHA_MATTING_FOREGROUND_THRESHOLD = 225
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10

# Alpha matting's solve runs on CPU regardless of GPU availability,
# so it's genuinely expensive per video frame - that's why it was
# off for video. Flip this on and benchmark fps if video edges
# still aren't precise enough after the tuning above.
VIDEO_ALPHA_MATTING = False


# ============================================================
# REMBG MODEL
# ============================================================

print("Loading rembg model...")
print(f"Model: {REMBG_MODEL}")

session = new_session(
    REMBG_MODEL,
    providers=REMBG_PROVIDERS
)

print("rembg model loaded.")


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    input_path: str,
    output_path: str
):

    print(
        f"Processing image: {input_path}"
    )

    with open(
        input_path,
        "rb"
    ) as f:

        image = f.read()


    # --------------------------------------------------------
    # Remove background
    #
    # Alpha matting improves fine edges such as:
    # - hair
    # - clothes
    # - fingers
    # --------------------------------------------------------

    result = remove(
        image,
        session=session,

        alpha_matting=True,

        alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,

        alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,

        alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE
    )


    # --------------------------------------------------------
    # Save RGBA PNG
    #
    # DO NOT convert to RGB/BGR.
    # The alpha channel contains transparency.
    # --------------------------------------------------------

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(result)


    print(
        f"Image saved: {output_path}"
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video_frames(
    input_path: str,
    frames_dir: str
):

    os.makedirs(
        frames_dir,
        exist_ok=True
    )


    cap = cv2.VideoCapture(
        input_path
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if not fps or fps <= 0:

        fps = 30.0


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    frame_count = 0


    print(
        f"Video size: {width}x{height}"
    )

    print(
        f"Video FPS: {fps}"
    )


    while True:

        success, frame = cap.read()


        if not success:

            break


        # ----------------------------------------------------
        # rembg (and the model underneath it) expect RGB, but
        # OpenCV decodes frames as BGR. Handing it BGR directly
        # means the model does inference with red and blue
        # swapped, which quietly costs some precision on
        # colour-dependent edges (skin against a warm-toned
        # background, for example). Convert here, then convert
        # the result back to BGRA below for cv2.imwrite.
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result_rgba = remove(
            frame_rgb,
            session=session,

            # Gated behind VIDEO_ALPHA_MATTING (see the tuning
            # section near the top of this file). Full alpha
            # matting is the more precise option, but its solve
            # runs on CPU per frame - benchmark fps before
            # relying on it for long videos.
            alpha_matting=VIDEO_ALPHA_MATTING,

            alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,

            alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,

            alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,

            # Cheap even with alpha matting off: a small
            # morphological opening + gaussian smoothing pass
            # that cleans up single-pixel noise and softens
            # jagged mask edges.
            post_process_mask=True
        )

        result = cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGRA)


        output_path = os.path.join(
            frames_dir,
            f"{frame_count:08d}.png"
        )


        # ----------------------------------------------------
        # Save RGBA PNG.
        #
        # PNG preserves transparency.
        # ----------------------------------------------------

        success_write = cv2.imwrite(
            output_path,
            result
        )


        if not success_write:

            raise RuntimeError(
                f"Could not write frame "
                f"{frame_count}"
            )


        frame_count += 1


        if frame_count % 30 == 0:

            print(
                f"Processed {frame_count} frames"
            )


    cap.release()


    print(
        f"Total frames processed: {frame_count}"
    )


    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_count
    }


# ============================================================
# PNG FRAMES -> TRANSPARENT WEBM
# ============================================================

def encode_transparent_webm(
    frames_dir: str,
    output_path: str,
    fps: float
):

    ffmpeg = shutil.which(
        "ffmpeg"
    )


    if ffmpeg is None:

        raise RuntimeError(
            "FFmpeg is not installed "
            "or not available in PATH."
        )


    input_pattern = os.path.join(
        frames_dir,
        "%08d.png"
    )


    command = [

        ffmpeg,

        "-y",

        "-framerate",
        str(fps),

        "-i",
        input_pattern,

        # VP9 supports alpha
        "-c:v",
        "libvpx-vp9",

        # Preserve transparency
        "-pix_fmt",
        "yuva420p",

        # Quality
        "-crf",
        "18",

        "-b:v",
        "0",

        output_path
    ]


    print(
        "Encoding transparent WebM..."
    )


    result = subprocess.run(
        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True
    )


    if result.returncode != 0:

        print(
            result.stderr
        )

        raise RuntimeError(
            "FFmpeg failed to encode "
            "transparent WebM."
        )


    print(
        f"WebM saved: {output_path}"
    )


# ============================================================
# UPLOAD + PROCESS
# ============================================================

@app.post("/process")
async def process(
    media: List[UploadFile] = File(...)
):

    if len(media) > 10:

        return JSONResponse(
            {
                "error":
                "Maximum 10 files allowed"
            },
            status_code=400
        )


    start = time.time()

    results = []


    # ========================================================
    # PROCESS EACH FILE
    # ========================================================

    for file in media:

        file_id = str(
            uuid.uuid4()
        )


        filename = (
            file.filename
            or "file"
        )


        ext = os.path.splitext(
            filename
        )[1].lower()


        input_path = (
            f"uploads/"
            f"{file_id}"
            f"{ext}"
        )


        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        content = await file.read()


        with open(
            input_path,
            "wb"
        ) as f:

            f.write(content)


        # ====================================================
        # IMAGE
        # ====================================================

        if (
            file.content_type
            and
            file.content_type.startswith(
                "image/"
            )
        ):

            output_path = (
                f"outputs/"
                f"{file_id}.png"
            )


            try:

                process_image(
                    input_path,
                    output_path
                )


                results.append(
                    {
                        "type": "image",

                        "url":
                        f"/outputs/"
                        f"{file_id}.png"
                    }
                )


            except Exception as e:

                print(
                    "IMAGE ERROR:",
                    repr(e)
                )


                results.append(
                    {
                        "type": "image",

                        "error":
                        str(e)
                    }
                )


        # ====================================================
        # VIDEO
        # ====================================================

        elif (
            file.content_type
            and
            file.content_type.startswith(
                "video/"
            )
        ):

            frames_dir = (
                f"temp/"
                f"{file_id}"
            )


            output_path = (
                f"outputs/"
                f"{file_id}.webm"
            )


            try:

                print(
                    f"Processing video: "
                    f"{filename}"
                )


                # ------------------------------------------------
                # Video -> transparent PNG frames
                # ------------------------------------------------

                video_info = (
                    process_video_frames(
                        input_path,
                        frames_dir
                    )
                )


                # ------------------------------------------------
                # Transparent PNG frames -> WebM
                # ------------------------------------------------

                encode_transparent_webm(
                    frames_dir,
                    output_path,
                    video_info["fps"]
                )


                results.append(
                    {
                        "type": "video",

                        "url":
                        f"/outputs/"
                        f"{file_id}.webm",

                        "fps":
                        video_info["fps"],

                        "frames":
                        video_info["frames"]
                    }
                )


            except Exception as e:

                print(
                    "VIDEO ERROR:",
                    repr(e)
                )


                results.append(
                    {
                        "type": "video",

                        "error":
                        str(e)
                    }
                )


            finally:

                # ------------------------------------------------
                # Delete temporary PNG frames
                # ------------------------------------------------

                if os.path.exists(
                    frames_dir
                ):

                    shutil.rmtree(
                        frames_dir,
                        ignore_errors=True
                    )


        # ====================================================
        # UNSUPPORTED
        # ====================================================

        else:

            results.append(
                {
                    "error":
                    f"{filename} "
                    f"not supported"
                }
            )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (
        time.time() - start
    )


    return JSONResponse(
        {
            "files": results,

            "time":
            f"{total_time:.2f} seconds"
        }
    )