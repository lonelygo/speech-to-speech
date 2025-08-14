from faster_whisper import WhisperModel

model_size = "large-v3"
model_name = "faster-whisper"
print("Downloading and converting STT model, this may take a while...")

model = WhisperModel(model_size, device="cpu", compute_type="int8")

# model = model.from_pretrained(model_name)

print("Download complete. Model name: ", model_name)