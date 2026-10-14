from abc import ABC, abstractmethod
import numpy as np
import math

class BaseExercise(ABC):
    def __init__(self, visibility_threshold=0.6):
        # Tüm hareketler için geçerli ortak referanslar
        self.gravity_vector_2d = np.array([0.0, 1.0])
        self.visibility_threshold = visibility_threshold

    # --- ORTAK MATEMATİKSEL ARAÇLAR (Tüm hareketler kullanabilir) ---
    def calculate_angle_2d(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def calculate_gravity_angle_2d(self, p_ust, p_alt):
        v_kemik = np.array([p_alt[0] - p_ust[0], p_alt[1] - p_ust[1]])
        if np.linalg.norm(v_kemik) == 0: return 0.0
        cosine_angle = np.dot(v_kemik, self.gravity_vector_2d) / np.linalg.norm(v_kemik)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_shoulder_shrug_angle(self, omuz, karsi_omuz):
        dy = karsi_omuz[1] - omuz[1]
        dx = abs(karsi_omuz[0] - omuz[0])
        if dx == 0: return 0.0
        return math.degrees(math.atan2(dy, dx))

    # --- SOYUT METOT (MECBURİ) ---
    # Bu sınıftan türeyen her egzersiz, kendi process_frame'ini yazmak ZORUNDADIR.
    @abstractmethod
    def process_frame(self, pose_landmarks, w, h):
        pass