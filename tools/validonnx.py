import argparse
import numpy as np
import onnxruntime as ort

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='onnx model file')
    args = parser.parse_args()
    
    ort_session = ort.InferenceSession(args.model)
    output = ort_session.run(None, {'data':np.random.randn(1, 3, 416, 416).astype(np.float32)})
    
    for i, o in enumerate(output):
        print(f"head{i} output size is {o.shape}")