# export_onnx.py
import torch
import torchvision.models as models

def export_resnet(output_path="resnet50.onnx"):
    model = models.resnet50(pretrained=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["input"], output_names=["output"], opset_version=11
    )
    print(f"[OK] ONNX 모델이 생성되었습니다: {output_path}")
