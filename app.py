from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
from recognize import recognize_face

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = recognize_face("temp.jpg")
    os.remove("temp.jpg")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result
    })

# 👇 ADD THIS PART
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)