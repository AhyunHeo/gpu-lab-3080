import torch
import torchvision.models as models
import os

# 모델 준비
model = models.resnet50(pretrained=True)
model.eval()

# 더미 입력 정의
dummy_input = torch.randn(1, 3, 224, 224)

# 저장 디렉토리 지정
output_path = os.path.join("api", "resnet50.onnx")

# ONNX 내보내기
torch.onnx.export(
    model,
    dummy_input,
    output_path,  # ✅ ./api/resnet50.onnx로 저장
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print(f"ONNX 모델이 다음 위치에 저장되었습니다: {output_path}")
