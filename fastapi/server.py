from detection import get_detections, get_detector
from pydantic_models import SegmentationReport
from segmentation import get_segmentator, get_segments

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

session_segmentation = get_segmentator()

# session_detection, img_size_h, img_size_w = get_detector()

app = FastAPI(
    title="CITC image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via U-Net implemented in Tensorflow.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["Startup"],
    description="démarrage de l'API sur la page de documentation.",
)
def main():
    return RedirectResponse(url="/docs/")


@app.post("/segmentation", response_model=SegmentationReport)
async def get_segmentation_map(file: UploadFile = File(...)):
    """Get segmentation maps from image file"""
    segmented_image = get_segments(session_segmentation, file)
    # bytes_io = io.BytesIO()
    # segmented_image.save(bytes_io, format="png")
    report = SegmentationReport(filename=file.filename)
    # return FileResponse("/main/result.jpg", media_type="image/jpg")
    return report


# @app.post("/detection")
# def get_detection(file: bytes = File(...)):
#     """Get segmentation maps from image file"""
#     detection_image = get_detections(session_detection, img_size_w, img_size_h, file)
#     bytes_io = io.BytesIO()
#     detection_image.save(bytes_io, format="png")
#     return Response(bytes_io.getvalue(), media_type="image/png")


@app.get("/healthcheck")
def get_api_status():
    return {"Status": "ok"}
