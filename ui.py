import cv2
import os
import numpy as np
import simpleaudio as sa

class EmotionIcon:
    def __init__(self, note, x, y, speed, judgment_line):
        self.emotion = note['emotion']
        self.time = note['time']
        self.point = note['point']
        self.x = x
        self.y = y
        self.size = 80
        self.speed = speed
        self.judgment_line = judgment_line
        self.icon = self.load_image()
    
    def load_image(self):
        try:
            img = cv2.imread(os.path.join('icons', f'{self.emotion}.png'), cv2.IMREAD_UNCHANGED)
            return cv2.resize(img, (self.size, self.size))
        except Exception as e:
            print(f"アイコンが読み込めません {self.emotion}: {e}")
    
    def update(self, current_time):
        self.x = self.judgment_line + self.speed * (self.time - current_time)

    def draw(self, frame):
        x_pos = int(self.x - self.size // 2)
        y_pos = int(self.y - self.size // 2)
        if x_pos < 0 or y_pos < 0 or x_pos + self.size > frame.shape[1] or y_pos + self.size > frame.shape[0]:
            return
        alpha_mask = np.stack([self.icon[:, :, 3] / 255.0] * 3, axis=2)
        roi = frame[y_pos:y_pos + self.size, x_pos:x_pos + self.size]
        frame[y_pos:y_pos + self.size, x_pos:x_pos + self.size] = self.icon[:, :, :3] * alpha_mask + roi * (1 - alpha_mask)
    
    def is_at_judgment_line(self):
        return self.x < self.judgment_line

class GameUI:
    def __init__(self, frame_width, frame_height, judgment_line=200):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.emotion_icons = []
        self.score = 0
        self.combo = 0
        self.judgment_line = judgment_line

    def add_emotion_icon(self, note, x, y, speed):
        self.emotion_icons.append(EmotionIcon(note, x, y, speed, self.judgment_line))

    def update_icons(self, current_time):
        for icon in self.emotion_icons[:]:
            icon.update(current_time)

    def draw_game_ui(self, frame):
        cv2.line(frame, (self.judgment_line, 0), (self.judgment_line, self.frame_height), (255, 255, 255), 3)
        cv2.putText(frame, f"Score : {self.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Combo: {self.combo}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for icon in self.emotion_icons:
            icon.draw(frame)
    
    def check_judgment(self, current_emotion):
        for icon in self.emotion_icons[:]:
            if icon.is_at_judgment_line():
                if not current_emotion:
                    self.combo = 0
                    self.emotion_icons.remove(icon)
                    return "MISS"
                elif icon.emotion == current_emotion:
                    sa.WaveObject.from_wave_file("musics/correct.wav").play()
                    self.score += icon.point
                    self.combo += 1
                    self.emotion_icons.remove(icon)
                    return "PERFECT"
                else:
                    self.combo = 0
                    self.emotion_icons.remove(icon)
                    return "MISS"
        return None
    
    def draw_judgment_text(self, frame, judgment):
        if judgment:
            color = (0, 255, 0) if judgment == "PERFECT" else (0, 0, 255)
            cv2.putText(frame, judgment, (self.frame_width // 2 - 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)