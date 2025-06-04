import gradio as gr
import requests
import numpy as np

API_URL = "http://localhost:8000"

def submit_job_and_get_result():
    dummy_input = np.random.rand(1, 3, 224, 224).tolist()
    job_resp = requests.post(f"{API_URL}/register_job", json={"input": dummy_input, "priority": 1})
    job_id = job_resp.json().get("job_id")
    job_fetch = requests.get(f"{API_URL}/get_job").json()
    if "input" not in job_fetch:
        return {"message": "No job available"}
    input_data = np.array(job_fetch["input"])
    infer_resp = requests.post(f"{API_URL}/infer", json={"input": input_data.tolist()})
    prediction = infer_resp.json().get("prediction")
    requests.post(f"{API_URL}/submit_result", json={"job_id": job_id, "result": prediction})
    return {"job_id": job_id, "prediction": prediction}

def report_node_status(node_id, gpu_util, memory_util):
    payload = {
        "node_id": node_id,
        "gpu_util": float(gpu_util),
        "memory_util": float(memory_util)
    }
    resp = requests.post(f"{API_URL}/status", json=payload)
    return resp.json()

def view_all_jobs():
    return requests.get(f"{API_URL}/jobs").json()

def view_all_nodes():
    return requests.get(f"{API_URL}/nodes").json()

def benchmark_jobs(n: int):
    results = []
    for _ in range(n):
        start = time.time()
        result = submit_job_and_get_result()
        duration = time.time() - start
        results.append({"job_id": result.get("job_id"), "latency_sec": duration})
    return results

def plot_node_status():
    data = requests.get(f"{API_URL}/nodes").json()["nodes"]
    if not data:
        return "No node data available"
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    df.plot(kind="bar", x="node_id", y=["gpu_util", "memory_util"], ax=ax)
    ax.set_title("Node Resource Utilization")
    return fig

def plot_job_status():
    data = requests.get(f"{API_URL}/jobs").json()["jobs"]
    if not data:
        return "No job data available"
    df = pd.DataFrame(data)
    status_counts = df["status"].value_counts()
    fig, ax = plt.subplots()
    status_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Job Status Distribution")
    return fig

def plot_history_heatmap():
    data = requests.get(f"{API_URL}/jobs").json()["jobs"]
    if not data:
        return "No job data"
    records = []
    for job in data:
        for h in job.get("history", []):
            records.append({"job_id": job.get("job_id", "-"), "status": h["status"], "time": h["timestamp"]})
    df = pd.DataFrame(records)
    if df.empty:
        return "No history data"
    df["time"] = pd.to_datetime(df["time"])
    pivot = df.pivot_table(index="time", columns="status", aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(ax=ax)
    ax.set_title("Job Status Over Time")
    return fig

with gr.Blocks() as dashboard:
    gr.Markdown("""# GPU Job Scheduler Dashboard""")

    with gr.Tab("Submit Inference Job"):
        submit_btn = gr.Button("Submit Job")
        submit_output = gr.JSON()
        submit_btn.click(submit_job_and_get_result, outputs=submit_output)

    with gr.Tab("Report Node Status"):
        node_id = gr.Textbox(label="Node ID", value="node-1")
        gpu_util = gr.Slider(0, 100, step=1, label="GPU Utilization (%)")
        mem_util = gr.Slider(0, 100, step=1, label="Memory Utilization (%)")
        report_btn = gr.Button("Report Status")
        report_output = gr.JSON()
        report_btn.click(report_node_status, inputs=[node_id, gpu_util, mem_util], outputs=report_output)

    with gr.Tab("View All Jobs"):
        jobs_btn = gr.Button("Refresh Job List")
        jobs_output = gr.JSON()
        jobs_btn.click(view_all_jobs, outputs=jobs_output)

    with gr.Tab("View All Nodes"):
        nodes_btn = gr.Button("Refresh Node List")
        nodes_output = gr.JSON()
        nodes_btn.click(view_all_nodes, outputs=nodes_output)

    with gr.Tab("Benchmark Test"):
        job_count = gr.Number(label="Number of Jobs", value=5, precision=0)
        bench_btn = gr.Button("Run Benchmark")
        bench_output = gr.JSON()
        bench_btn.click(benchmark_jobs, inputs=[job_count], outputs=bench_output)

    with gr.Tab("Job Status Report"):
        job_report_btn = gr.Button("Generate Job Report")
        job_report_plot = gr.Plot()
        job_report_btn.click(plot_job_status, outputs=job_report_plot)

    with gr.Tab("Node Utilization Report"):
        node_report_btn = gr.Button("Generate Node Report")
        node_report_plot = gr.Plot()
        node_report_btn.click(plot_node_status, outputs=node_report_plot)

    with gr.Tab("Status History Over Time"):
        hist_btn = gr.Button("Show History")
        hist_plot = gr.Plot()
        hist_btn.click(plot_history_heatmap, outputs=hist_plot)

dashboard.launch(server_name="0.0.0.0", server_port=7860)
