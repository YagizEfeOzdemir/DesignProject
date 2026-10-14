import numpy as np
from core.base_exercise import BaseExercise


class BicepsCurlPreprocessor(BaseExercise):
    def __init__(self):
        super().__init__()

    def process_frame(self, pose_landmarks, w, h):
        vis_right = pose_landmarks[12].visibility + pose_landmarks[14].visibility
        vis_left = pose_landmarks[11].visibility + pose_landmarks[13].visibility

        # Parmak_idx yerine isaret ve serce parmakları alıyoruz
        if vis_right > vis_left:
            omuz_idx, dirsek_idx, bilek_idx = 12, 14, 16
            isaret_idx, serce_idx = 20, 18
        else:
            omuz_idx, dirsek_idx, bilek_idx = 11, 13, 15
            isaret_idx, serce_idx = 19, 17

        # Parmak görünürlüğü için işaret ve serçe parmağın ikisinin de eşiği geçmesini istiyoruz
        vis = {
            "omuz": pose_landmarks[omuz_idx].visibility > self.visibility_threshold,
            "dirsek": pose_landmarks[dirsek_idx].visibility > self.visibility_threshold,
            "bilek": pose_landmarks[bilek_idx].visibility > self.visibility_threshold,
            "parmak": (pose_landmarks[isaret_idx].visibility > self.visibility_threshold and
                       pose_landmarks[serce_idx].visibility > self.visibility_threshold)
        }

        omuz = np.array([pose_landmarks[omuz_idx].x * w, pose_landmarks[omuz_idx].y * h])
        dirsek = np.array([pose_landmarks[dirsek_idx].x * w, pose_landmarks[dirsek_idx].y * h])
        bilek = np.array([pose_landmarks[bilek_idx].x * w, pose_landmarks[bilek_idx].y * h])

        # --- KANKA SENİN İSTEDİĞİN MATEMATİK BURADA ---
        # İşaret ve Serçe parmağın piksellerini alıp 2'ye bölüyoruz (Elin tam ortası)
        isaret = np.array([pose_landmarks[isaret_idx].x * w, pose_landmarks[isaret_idx].y * h])
        serce = np.array([pose_landmarks[serce_idx].x * w, pose_landmarks[serce_idx].y * h])

        parmak = (isaret + serce) / 2.0
        # ----------------------------------------------

        bilek_acisi = self.calculate_angle_2d(dirsek, bilek, parmak) if (
                    vis["dirsek"] and vis["bilek"] and vis["parmak"]) else None
        biceps_acisi = self.calculate_angle_2d(omuz, dirsek, bilek) if (
                    vis["omuz"] and vis["dirsek"] and vis["bilek"]) else None
        yercekimi_acisi = self.calculate_gravity_angle_2d(omuz, dirsek) if (vis["omuz"] and vis["dirsek"]) else None

        # Noktaları main'e yolluyoruz (Main hala sadece 'parmak' değişkeni geldiğini sanıyor, arka plandaki hesabı bilmiyor. OOP'nin gücü!)
        noktalar_px = {
            "omuz": (int(omuz[0]), int(omuz[1])), "dirsek": (int(dirsek[0]), int(dirsek[1])),
            "bilek": (int(bilek[0]), int(bilek[1])), "parmak": (int(parmak[0]), int(parmak[1]))
        }
        return bilek_acisi, biceps_acisi, yercekimi_acisi, noktalar_px, vis