import torch
import torch.nn as nn
import json
 
def generateTensors():

    # change parameters here
    batch_size = 2
    in_channels = 4
    height, width = 100, 100
    k_size = 3

    input_tensor = torch.randn(batch_size, in_channels, height, width)

    conv = nn.Conv2d(in_channels=in_channels, 
                    out_channels=1,
                    kernel_size=k_size,
                    bias=False)

    kernel = conv.weight.data

    output = conv(input_tensor)

    return [input_tensor, kernel, output]

def tensors_to_json(tensors):
    with open("test4.json", "w", encoding="utf-8") as f:
        data = {"tensors": []}

        for tensor in tensors:
            shape = list(tensor.shape)
            b_size, channels, h, w = shape

            tensor_dict = {
                "height": h,
                "width": w,
                "channels": channels,
                "batchSize": b_size,
                "layers": []
            }

            for b in range(b_size):
                for c in range(channels):
                    flat_layer = tensor[b, c].flatten().tolist()
                    tensor_dict["layers"].append(flat_layer)

            data["tensors"].append(tensor_dict)

        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=False)

if __name__ == "__main__":
    tensors = generateTensors()
    tensors_to_json(tensors)
