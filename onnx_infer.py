import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("resnet50.onnx")

def run_inference(input_array):
    input_array = input_array.astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    return outputs[0]
