import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",  # ✅ 생성될 ONNX 모델 파일명
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
