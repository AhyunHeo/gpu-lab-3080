import numpy as np
import json

# ONNX 입력에 맞는 더미 데이터 생성 (1, 3, 224, 224)
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

# JSON 형식으로 저장
with open("test_input.json", "w") as f:
    json.dump({"input": dummy_input.tolist()}, f)
