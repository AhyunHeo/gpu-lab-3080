from fastapi import FastAPI, Request
from onnx_infer import run_inference
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient
import gradio as gr
import requests
import time
import matplotlib.pyplot as plt
import pandas as pd
from bson import ObjectId
# api_server.py 또는 onnx_infer.py 상단에 추가
import pathlib
from export_onnx import export_resnet

if not pathlib.Path("resnet50.onnx").exists():
    export_resnet()

app = FastAPI()

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017")
db = client["gpu_scheduler"]
jobs_collection = db["jobs"]
nodes_collection = db["nodes"]

class JobRequest(BaseModel):
    input: List
    metadata: Optional[dict] = {}
    priority: Optional[int] = 1

class NodeStatus(BaseModel):
    node_id: str
    gpu_util: float
    memory_util: float
    timestamp: Optional[datetime] = None

class ResultSubmit(BaseModel):
    job_id: str
    result: List

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    input_data = np.array(data["input"])
    prediction = run_inference(input_data)
    return {"prediction": prediction.tolist()}

@app.post("/register_job")
async def register_job(job: JobRequest):
    doc = {
        "input": job.input,
        "metadata": job.metadata,
        "priority": job.priority,
        "status": "queued",
        "history": [{"status": "queued", "timestamp": datetime.utcnow()}],
        "created_at": datetime.utcnow()
    }
    result = jobs_collection.insert_one(doc)
    return {"job_id": str(result.inserted_id)}

@app.post("/status")
async def report_status(status: NodeStatus):
    status.timestamp = status.timestamp or datetime.utcnow()
    nodes_collection.update_one(
        {"node_id": status.node_id},
        {"$set": status.dict()},
        upsert=True
    )
    return {"status": "updated"}

@app.get("/get_job")
async def get_job():
    job = jobs_collection.find_one_and_update(
        {"status": "queued"},
        {"$set": {"status": "assigned", "assigned_at": datetime.utcnow()},
         "$push": {"history": {"status": "assigned", "timestamp": datetime.utcnow()}}},
        sort=[("priority", -1), ("created_at", 1)]
    )
    if job:
        return {"job_id": str(job["_id"]), "input": job["input"]}
    return {"message": "No job available"}

@app.post("/submit_result")
async def submit_result(result: ResultSubmit):
    jobs_collection.update_one(
        {"_id": ObjectId(result.job_id)},
        {"$set": {"status": "completed", "result": result.result, "completed_at": datetime.utcnow()},
         "$push": {"history": {"status": "completed", "timestamp": datetime.utcnow()}}}
    )
    return {"status": "result saved"}

@app.get("/jobs")
async def get_all_jobs():
    jobs = list(jobs_collection.find({}, {"_id": 0}))
    return {"jobs": jobs}

@app.get("/nodes")
async def get_all_nodes():
    nodes = list(nodes_collection.find({}, {"_id": 0}))
    return {"nodes": nodes}