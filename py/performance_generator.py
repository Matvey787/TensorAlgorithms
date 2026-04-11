import torch
import torch.nn as nn
import json
from tqdm import tqdm
import os

def generate_and_save_chunks(
    chunk_size=100, # The maximum number of (input, kernel) pairs in a single file
    filename_prefix="test",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")

    start_size = 20
    end_size = 20
    step_size = 100

    start_batch = 2
    end_batch = 2
    step_batch = 1
    
    start_in_channels = 1000
    end_in_channels = 2000
    step_in_channels = 10

    kernel_size = 3

    num_sizes = ((end_size - start_size) // step_size) + 1
    num_batches = ((end_batch - start_batch) // step_batch) + 1
    num_in_channels = ((end_in_channels - start_in_channels) // step_in_channels) + 1

    total_pairs = num_sizes * num_batches * num_in_channels
    print(f"Total number of tensor pairs: {total_pairs}")
    print(f"It will be created about {total_pairs // chunk_size + 1} files each containig {chunk_size} pairs\n")

    chunk_idx = 1
    current_chunk = []
    pair_counter = 0

    size = start_size
    while size <= end_size:
        batch = start_batch
        while batch <= end_batch:
            in_ch = start_in_channels
            while in_ch <= end_in_channels:
                
                input_tensor = torch.randn(batch, in_ch, size, size, device=device)
                
                conv = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=1,
                    kernel_size=kernel_size,
                    bias=False,
                    device=device
                )
                kernel = conv.weight.data
                
                current_chunk.append((input_tensor, kernel))
                pair_counter += 1
                
                if len(current_chunk) >= chunk_size:
                    save_chunk(current_chunk, chunk_idx, filename_prefix, device)
                    current_chunk.clear()
                    chunk_idx += 1
                    torch.cuda.empty_cache()

                in_ch += step_in_channels
            batch += step_batch
        size += step_size

    if current_chunk:
        save_chunk(current_chunk, chunk_idx, filename_prefix, device)

    print(f"\nSuccess!")


def save_chunk(chunk, chunk_idx, prefix, device):
    filename = f"{prefix}_{chunk_idx}.json"
    print(f"Save chunk {chunk_idx} → {filename} ({len(chunk)} pairs)")
    
    data = {"tensors": []}
    pbar = tqdm(total=len(chunk), desc=f"Prepare chunk {chunk_idx}", unit="pair", leave=False)
    
    for input_tensor, kernel in chunk:
        input_cpu = input_tensor.cpu()
        kernel_cpu = kernel.cpu()
        
        # Input tensor
        b, c_in, h, w = input_cpu.shape
        input_dict = {
            "height": h,
            "width": w,
            "channels": c_in,
            "batchSize": b,
            "layers": []
        }
        
        for batch_idx in range(b):
            for ch in range(c_in):
                flat_layer = input_cpu[batch_idx, ch].flatten().tolist()
                input_dict["layers"].append(flat_layer)
        
        data["tensors"].append(input_dict)
        
        # Kernel
        out_c, in_c, kh, kw = kernel_cpu.shape
        kernel_dict = {
            "height": kh,
            "width": kw,
            "channels": in_c,
            "batchSize": out_c,
            "layers": []
        }
        
        for out_ch in range(out_c):
            for in_ch in range(in_c):
                flat_layer = kernel_cpu[out_ch, in_ch].flatten().tolist()
                kernel_dict["layers"].append(flat_layer)
        
        data["tensors"].append(kernel_dict)
        pbar.update(1)
    
    pbar.close()

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Chunk {chunk_idx} saved to {filename}")


if __name__ == "__main__":
    print(f"When selecting the maximum number of chunk pairs in a single file, please bear in mind your computer’s RAM capacity, as well as the size of the tensors themselves. I recommend checking the system monitor when you first run the programme to keep an eye on RAM usage. If the system becomes overloaded, terminate the programme immediately; otherwise, everything will freeze.")
    generate_and_save_chunks(
        chunk_size=105,
        filename_prefix="tests/performance/tensorChannelsTest/test3",
        device=None
    )
    