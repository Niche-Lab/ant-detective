import argparse
from ultralytics import YOLO
import torch
from memory_profiler import profile

@profile
def train_model(args):
    i = args.iters
    model = YOLO("yolov8n.pt")
    # add psudo torch tensor
    dummy_input = torch.randn(1, 3, 640, 640)
    model.eval()
    model(dummy_input)
    # max memory allocated
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory_allocated / (1024 ** 2):.2f} MB")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iters", type=str, default="0")
    args = parser.parse_args()
    start = torch.cuda.memory_allocated()
    # run specific block
    train_model(args)
    end = torch.cuda.memory_allocated()
    print(f"Memory allocated during training: {(end - start) / (1024 ** 2):.2f} MB")
