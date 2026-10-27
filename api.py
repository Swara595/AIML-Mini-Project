# api.py - web backend for the new frontend.
# Run:  python api.py      then open http://127.0.0.1:8000
import engine
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import threading
import webbrowser
from typing import Literal
import joblib

# always work from this folder, so models/ and data/ are found
os.chdir(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI(title="UPI Shield")
eng = engine.Engine()

lightgbm_model = joblib.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "lightgbm_model.pkl"
    )
)


class Payment(BaseModel):
    sender: str
    receiver: str = Field(min_length=1, max_length=40)
    amount: float = Field(gt=0, le=10000000)
    txn_type: Literal["COLLECT_REQUEST", "MERCHANT", "P2P", "QR_PAYMENT"]
    hour: int = Field(ge=0, le=23)
    new_device: bool = False
    recent: int = Field(default=0, ge=0, le=100)
    message: str = Field(default="", max_length=500)


class Feedback(BaseModel):
    sender: str
    receiver: str
    features: dict[str, float]
    system_said: Literal["GREEN", "YELLOW", "RED"]
    was_right: bool


@app.get("/api/senders")
def senders():
    return eng.sender_list()


@app.get("/api/demos")
def demos():
    return eng.demos()


@app.post("/api/check")
def check(p: Payment):
    try:
        return eng.score(p.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/feedback")
def feedback(fb: Feedback):
    eng.save_feedback(fb.model_dump())
    return eng.feedback_info()


@app.get("/api/feedback")
def feedback_info():
    return eng.feedback_info()


@app.post("/api/retrain")
def run_retrain():
    return eng.run_retrain()


@app.get("/api/feed")
def feed(seed: int = 1):
    return eng.feed(seed)


@app.get("/api/mules")
def mules():
    return eng.mule_list()


@app.get("/api/mules/{account}")
def mule(account: str):
    if account not in eng.mules:
        raise HTTPException(
            status_code=404, detail="%s is not a flagged mule account." % account)
    return eng.mule_detail(account)


@app.get("/api/report")
def report():
    return eng.report()


@app.get("/files/shap_summary.png")
def shap_image():
    if not os.path.exists("shap_summary.png"):
        raise HTTPException(
            status_code=404, detail="Run explain.py to create shap_summary.png.")
    return FileResponse("shap_summary.png")


# the web page itself (must come last)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    port = 8000
    print("\nUPI Shield is running at http://127.0.0.1:%d  (press Ctrl+C to stop)\n" % port)
    threading.Timer(1.5, lambda: webbrowser.open(
        "http://127.0.0.1:%d" % port)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
