from fastapi import FastAPI, Request
from onnx_infer import run_inference
import numpy as np

app = FastAPI()

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    input_data = np.array(data["input"])
    prediction = run_inference(input_data)
    return {"prediction": prediction.tolist()}
