from fastapi import FastAPI, File, UploadFile
from converter import parse_schedule
from ics_generator import generate_ics
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    rows = parse_schedule(image_bytes)
    return {"rows": rows}

@app.post("/download")
async def download_ics(payload: dict):
    rows = payload["rows"]
    start_date = payload["start_date"]
    end_date = payload["end_date"]
    reminder = int(payload["reminder"])

    ics_text = generate_ics(rows, start_date, end_date, reminder)
    stream = io.BytesIO(ics_text.encode("utf-8"))

    return StreamingResponse(
        stream,
        media_type="text/calendar",
        headers={"Content-Disposition": "attachment; filename=classes.ics"}
    )