import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import torchvision.io as io
import cv2
import os
from moviepy.editor import ImageSequenceClip
import requests

device = "cpu"

# Загружаем лёгкую 3D CNN модель
model = torchvision.models.video.r3d_18(weights="KINETICS400_V1").to(device)
model.eval()

# Преобразования для видео
transform = Compose([
    Resize((112, 112)),
    CenterCrop((112, 112)),
    Normalize(mean=[0.43216, 0.394666, 0.37645],
              std=[0.22803, 0.22145, 0.216989]),
])

# Загружаем видео
video_path = "input_video.mp4"
frames, _, info = io.read_video(video_path, start_pts=0, end_pts=8, pts_unit='sec')
fps = info["video_fps"]
print(f"FPS: {fps}, frames loaded: {frames.shape[0]}")

# Берём каждый 5-й кадр для ускорения
frames = frames[::5]

# Приводим форму: [T, H, W, C] → [T, C, H, W]
frames = frames.permute(0, 3, 1, 2) / 255.0

# Применяем трансформацию к каждому кадру
frames = torch.stack([transform(f) for f in frames])

# Преобразуем в форму [1, 3, T, H, W]
video_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)

# Предсказание
with torch.no_grad():
    preds = model(video_tensor)
    pred_class = preds.argmax(dim=1).item()

# Загружаем метки классов Kinetics400
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
labels = requests.get(KINETICS_URL).text.strip().split("\n")
action_label = labels[pred_class].split(": ")[-1]

print(f"\n Предсказанное действие: {action_label}")

# Визуализация
frames_dir = "r3d_frames"
os.makedirs(frames_dir, exist_ok=True)

for i in range(min(len(frames), 20)):
    # Преобразуем кадр в numpy-массив (uint8)
    frame = frames[i].permute(1, 2, 0).numpy()
    frame = (frame * 255).clip(0, 255).astype("uint8").copy()  # ← вот ключевое .copy()

    cv2.putText(frame, f"Action: {action_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(f"{frames_dir}/frame_{i}.jpg", frame)

# Создаём GIF
clip = ImageSequenceClip([f"{frames_dir}/frame_{i}.jpg" for i in range(min(len(frames), 20))], fps=5)
clip.write_gif("video_action_output.gif")
