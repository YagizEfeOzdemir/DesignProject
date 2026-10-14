import numpy as np
from core.base_exercise import BaseExercise

class LateralRaisePreprocessor(BaseExercise):
    def __init__(self):
        super().__init__()

    def process_frame(self, pose_landmarks, w, h):
        vis = {
            "r_omuz": pose_landmarks[12].visibility > self.visibility_threshold,
            "r_dirsek": pose_landmarks[14].visibility > self.visibility_threshold,
            "r_bilek": pose_landmarks[16].visibility > self.visibility_threshold,
            "r_kalca": pose_landmarks[24].visibility > self.visibility_threshold,
            "l_omuz": pose_landmarks[11].visibility > self.visibility_threshold,
            "l_dirsek": pose_landmarks[13].visibility > self.visibility_threshold,
            "l_bilek": pose_landmarks[15].visibility > self.visibility_threshold,
            "l_kalca": pose_landmarks[23].visibility > self.visibility_threshold
        }

        r_omuz = np.array([pose_landmarks[12].x * w, pose_landmarks[12].y * h])
        r_dirsek = np.array([pose_landmarks[14].x * w, pose_landmarks[14].y * h])
        r_bilek = np.array([pose_landmarks[16].x * w, pose_landmarks[16].y * h])
        r_kalca = np.array([pose_landmarks[24].x * w, pose_landmarks[24].y * h])

        l_omuz = np.array([pose_landmarks[11].x * w, pose_landmarks[11].y * h])
        l_dirsek = np.array([pose_landmarks[13].x * w, pose_landmarks[13].y * h])
        l_bilek = np.array([pose_landmarks[15].x * w, pose_landmarks[15].y * h])
        l_kalca = np.array([pose_landmarks[23].x * w, pose_landmarks[23].y * h])

        if vis["r_omuz"] and vis["l_omuz"] and vis["r_kalca"] and vis["l_kalca"]:
            omuz_genisligi = np.linalg.norm(r_omuz - l_omuz)
            kalca_genisligi = np.linalg.norm(r_kalca - l_kalca)
            egilme_orani = omuz_genisligi / kalca_genisligi if omuz_genisligi > 0 else 0.0
        else:
            egilme_orani = None

        r_kalkis = self.calculate_gravity_angle_2d(r_omuz, r_dirsek) if (vis["r_omuz"] and vis["r_dirsek"]) else None
        r_dirsek_aci = self.calculate_angle_2d(r_omuz, r_dirsek, r_bilek) if (vis["r_omuz"] and vis["r_dirsek"] and vis["r_bilek"]) else None
        r_govde = self.calculate_gravity_angle_2d(r_omuz, r_kalca) if (vis["r_omuz"] and vis["r_kalca"]) else None
        r_silkme = self.get_shoulder_shrug_angle(r_omuz, l_omuz) if (vis["r_omuz"] and vis["l_omuz"]) else None

        l_kalkis = self.calculate_gravity_angle_2d(l_omuz, l_dirsek) if (vis["l_omuz"] and vis["l_dirsek"]) else None
        l_dirsek_aci = self.calculate_angle_2d(l_omuz, l_dirsek, l_bilek) if (vis["l_omuz"] and vis["l_dirsek"] and vis["l_bilek"]) else None
        l_govde = self.calculate_gravity_angle_2d(l_omuz, l_kalca) if (vis["l_omuz"] and vis["l_kalca"]) else None
        l_silkme = self.get_shoulder_shrug_angle(l_omuz, r_omuz) if (vis["l_omuz"] and vis["r_omuz"]) else None

        noktalar_px = {
            "r_omuz": (int(r_omuz[0]), int(r_omuz[1])), "r_dirsek": (int(r_dirsek[0]), int(r_dirsek[1])),
            "r_bilek": (int(r_bilek[0]), int(r_bilek[1])), "r_kalca": (int(r_kalca[0]), int(r_kalca[1])),
            "l_omuz": (int(l_omuz[0]), int(l_omuz[1])), "l_dirsek": (int(l_dirsek[0]), int(l_dirsek[1])),
            "l_bilek": (int(l_bilek[0]), int(l_bilek[1])), "l_kalca": (int(l_kalca[0]), int(l_kalca[1]))
        }

        metrics_r = (r_kalkis, r_dirsek_aci, r_govde, r_silkme)
        metrics_l = (l_kalkis, l_dirsek_aci, l_govde, l_silkme)

        return metrics_r, metrics_l, egilme_orani, noktalar_px, vis