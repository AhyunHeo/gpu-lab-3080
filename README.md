# gpu-lab-3080

## 이미지 빌드
`docker build -t onnx-infer-api .`


## 컨테이너 실행 (GPU 할당 포함)
`docker run --rm -it --gpus all -p 8000:8000 onnx-infer-api`
