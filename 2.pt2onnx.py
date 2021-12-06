# -*- coding: UTF-8 -*-
from models.net import Net
import torch.onnx
import os


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch pt file to ONNX file')
    parser.add_argument('-i', '--input', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dummy_input = torch.randn(1, 1, 28, 28)
    model = Net()

    print("Loading state dict to cpu")
    model.load_state_dict(torch.load(args.input, map_location=torch.device('cpu')))
    name = os.path.join(os.path.dirname(args.input), os.path.basename(args.input).split('.')[0] + ".onnx")

    print("Onnx files at:", name)
    torch.onnx.export(model, dummy_input, name)


if __name__ == '__main__':
    main()
