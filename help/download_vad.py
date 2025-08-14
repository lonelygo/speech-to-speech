import torch

torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)

print("VAD model downloaded.")