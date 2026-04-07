import torch
import torch.nn as nn

# ==================== РАЗНЫЕ ЗНАЧЕНИЯ ====================
N = 2          # размер батча
C_in = 3       # входных каналов
H, W = 6, 6    # высота и ширина
K = 3          # размер ядра
C_out = 2      # выходных каналов

# Создаём входной тензор со случайными значениями
input_tensor = torch.randn((N, C_in, H, W))  # разные случайные числа

# Создаём свёрточный слой
conv = nn.Conv2d(in_channels=C_in, 
                 out_channels=C_out, 
                 kernel_size=K, 
                 stride=1, 
                 padding=0)

# Заполняем веса случайными значениями
with torch.no_grad():
    conv.weight = nn.Parameter(torch.randn(C_out, C_in, K, K))  # случайные веса
    conv.bias = nn.Parameter(torch.randn(C_out))  # случайные bias

# Выполняем свёртку
output_tensor = conv(input_tensor)

# ====================== ВЫВОД ======================
print("=== СВЁРТКА СО СЛУЧАЙНЫМИ ЗНАЧЕНИЯМИ ===\n")

print(f"Входной тензор (первое изображение, первый канал):")
print(input_tensor[0, 0])
print(f"\nМинимум: {input_tensor.min():.3f}, Максимум: {input_tensor.max():.3f}")

print(f"\nВеса ядра (первый выходной канал, первый входной канал):")
print(conv.weight[0, 0])

print(f"\nBias: {conv.bias.data}")

print(f"\nВыходной тензор (первое изображение, первый канал):")
print(output_tensor[0, 0])
print(f"\nМинимум: {output_tensor.min():.3f}, Максимум: {output_tensor.max():.3f}")
