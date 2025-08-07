import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from approach.ResEmoteNet import ResEmoteNet
from ultralytics import YOLO
from time import time
from ui import GameUI
from parser import SheetParser
import simpleaudio as sa

class FacialExpressionGame:
  def __init__(self, sheet='sheet.txt'):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.yolo_device = 0 if torch.cuda.is_available() else "cpu"
    self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
    self.model = ResEmoteNet().to(self.device)
    self.model.load_state_dict(torch.load('models/best_resemotenet_model.pth', map_location=self.device))
    self.model.eval()
    self.transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    self.face_detector = YOLO('models/yolov8n-face-lindevs.pt')
    self.video_capture = cv2.VideoCapture(0)
    self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.judgment_line = 200
    self.judgment_time = None
    self.judgment_text = None
    self.game_ui = GameUI(self.frame_width, self.frame_height, judgment_line=200)
    self.sheet_parser = SheetParser(sheet)
    self.game_start_time = 0
    self.note_speed = 110
    self.load_time_offset = (self.frame_width - self.judgment_line) / self.note_speed
    self.spawn_current_emotion = []
    self.current_emotion = ''
    self.game_state = 'menu'
    self.play_obj = None
    self.font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 40)
    self.small_font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 24)

  def detect_emotion(self, frame):
    vid_fr_tensor = self.transform(frame).unsqueeze(0).to(self.device)
    with torch.no_grad():
      outputs = self.model(vid_fr_tensor)
      probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores
  
  def get_max_emotion(self, x, y, w, h, frame):
    crop_img = frame[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = self.detect_emotion(pil_crop_img)    
    max_index = np.argmax(rounded_scores)
    max_emotion = self.emotions[max_index]
    return max_emotion, rounded_scores
  
  def detect_faces(self, frame):
    results = self.face_detector(frame, device=self.yolo_device, verbose=False)[0]
    faces = []
    for xyxy, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces
  
  def draw_face_info(self, frame, faces):
    if not faces:
      self.current_emotion = ''
    for (x, y, w, h) in faces:
      emotion, scores = self.get_max_emotion(x, y, w, h, frame)
      self.current_emotion = emotion
      color = (0, 0, 255)
      if self.game_state != 'playing':
        color = (0, 255, 0)
      if self.spawn_current_emotion and emotion == self.spawn_current_emotion[0]['emotion']:
        color = (0, 255, 0)
      cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
      cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
      if self.game_state != 'playing':
        y_offset = 30
        for i, emotion_name in enumerate(self.emotions):
          prob_text = f"{emotion_name}: {scores[i]:.2f}"
          cv2.putText(frame, prob_text, (x + w + 10, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
          y_offset += 20
  
  def update_game(self, current_time):
    spawn_notes = self.sheet_parser.get_notes_at_time(current_time + self.load_time_offset)
    for note in spawn_notes:
      self.spawn_current_emotion.append(note)
      x_position = self.judgment_line + self.note_speed * (note['time'] - current_time)
      y_position = 100 + (len(self.game_ui.emotion_icons) * 120) % 400
      self.game_ui.add_emotion_icon(note, x_position, y_position, self.note_speed)
    self.game_ui.update_icons(current_time)
    judgment = self.game_ui.check_judgment(self.current_emotion)
    if judgment:
      del self.spawn_current_emotion[0]
      self.judgment_time = time()
      self.judgment_text = judgment
  
  def draw_game_frame(self, frame):
    self.game_ui.draw_game_ui(frame)
    if self.judgment_time and time() - self.judgment_time <= 0.25:
        self.game_ui.draw_judgment_text(frame, self.judgment_text)
    else:
      self.judgment_time = None
      self.judgment_text = None

  def run(self):
    final_score = 0
    while True:
      ret, frame = self.video_capture.read()
      if not ret:
        break
      frame = cv2.flip(frame, 1)
      self.draw_face_info(frame, self.detect_faces(frame))
      if self.game_state == 'playing':
        self.update_game(time() - self.game_start_time)
        self.draw_game_frame(frame)
        if self.sheet_parser.is_finished() and len(self.game_ui.emotion_icons) == 0:
          final_score = self.game_ui.score
          self.game_state = 'finished'
          print(f"最終スコア: {final_score}")
      elif self.game_state == 'finished':
        self.draw_result_screen(frame, final_score)
      else:
        self.draw_menu_screen(frame)
      cv2.imshow("Facial Expression Game", frame)
      key = cv2.waitKey(1) & 0xFF
      if key in (27, ord('q')):
        break
      elif key == ord(' ') and self.game_state != 'playing':
        self.start_new_game()
    self.video_capture.release()
    cv2.destroyAllWindows()

  def start_new_game(self):
        self.game_state = 'playing'
        self.game_start_time = time()
        self.sheet_parser.reset()
        self.game_ui.score = 0
        self.game_ui.combo = 0
        self.spawn_current_emotion = []
        try:
          self.play_obj = sa.WaveObject.from_wave_file("musics/music.wav").play()
        except:
          pass
        print("ゲーム開始")
  
  def draw_menu_screen(self, frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((self.frame_width // 2 - 250, self.frame_height // 2 - 200), "表情筋トレーニングゲーム", fill=(0, 221, 225), font=self.font)
    draw.text((self.frame_width // 2 - 290, self.frame_height // 2 - 140), "ノートがバーに到達するまでに表情を作りましょう", fill=(255, 255, 0), font=self.small_font)
    draw.text((self.frame_width - 240, self.frame_height - 30), "開始:SPACE", fill=(0, 255, 0), font=self.small_font)
    draw.text((self.frame_width - 100, self.frame_height - 30), "終了:Q", fill=(255, 0, 255), font=self.small_font)
    frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
  
  def draw_result_screen(self, frame, final_score):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((self.frame_width//2 - 110, self.frame_height//2 - 120), "ゲーム終了", fill=(255, 255, 255), font=self.font)
    score_color = (0, 255, 255) if final_score > 500 else (255, 255, 0)
    draw.text((self.frame_width//2 - 150, self.frame_height//2 - 60), f"最終スコア: {final_score}", fill=score_color, font=self.font)
    if final_score >= self.sheet_parser.total_point:
        message = "PERFECT"
        message_color = (0, 255, 0)
    elif final_score >= self.sheet_parser.total_point * 0.4:
        message = "  GOOD "
        message_color = (0, 255, 255)
    else:
        message = "  BAD  "
        message_color = (255, 0, 0)
    draw.text((self.frame_width//2 - 80, self.frame_height//2), message, fill=message_color, font=self.font)
    draw.text((self.frame_width - 260, self.frame_height - 30), "再挑戦:SPACE", fill=(0, 255, 0), font=self.small_font)
    draw.text((self.frame_width - 100, self.frame_height - 30), "終了:Q", fill=(255, 0, 255), font=self.small_font)
    frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
  game = FacialExpressionGame()
  game.run()