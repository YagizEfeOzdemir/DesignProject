import cv2
import mediapipe as mp
import numpy as np
import math


class LateralRaisePreprocessor:
    def __init__(self):
        self.gravity_vector_2d = np.array([0.0, 1.0])
        self.visibility_threshold = 0.6

    def _calculate_angle_2d(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _calculate_gravity_angle_2d(self, p_ust, p_alt):
        v_kemik = np.array([p_alt[0] - p_ust[0], p_alt[1] - p_ust[1]])
        if np.linalg.norm(v_kemik) == 0: return 0.0
        cosine_angle = np.dot(v_kemik, self.gravity_vector_2d) / np.linalg.norm(v_kemik)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_shoulder_shrug_angle(self, omuz, karsi_omuz):
        dy = karsi_omuz[1] - omuz[1]
        dx = abs(karsi_omuz[0] - omuz[0])
        if dx == 0: return 0.0
        return math.degrees(math.atan2(dy, dx))

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

        # --- YENİ EKLENEN: SENTETİK Z-EKSENİ (EĞİLME ORANI) ---
        if vis["r_omuz"] and vis["l_omuz"] and vis["r_kalca"] and vis["l_kalca"]:
            omuz_genisligi = np.linalg.norm(r_omuz - l_omuz)
            kalca_genisligi = np.linalg.norm(r_kalca - l_kalca)
            r_govde_boyu = np.linalg.norm(r_kalca - r_omuz)
            l_govde_boyu = np.linalg.norm(l_kalca - l_omuz)
            govde_uzunlugu = (r_govde_boyu + l_govde_boyu) / 2.0

            egilme_orani = omuz_genisligi / kalca_genisligi if omuz_genisligi > 0 else 0.0
        else:
            egilme_orani = None
        # -----------------------------------------------------

        r_kalkis = self._calculate_gravity_angle_2d(r_omuz, r_dirsek) if (vis["r_omuz"] and vis["r_dirsek"]) else None
        r_dirsek_aci = self._calculate_angle_2d(r_omuz, r_dirsek, r_bilek) if (
                    vis["r_omuz"] and vis["r_dirsek"] and vis["r_bilek"]) else None
        r_govde = self._calculate_gravity_angle_2d(r_omuz, r_kalca) if (vis["r_omuz"] and vis["r_kalca"]) else None
        r_silkme = self._get_shoulder_shrug_angle(r_omuz, l_omuz) if (vis["r_omuz"] and vis["l_omuz"]) else None

        l_kalkis = self._calculate_gravity_angle_2d(l_omuz, l_dirsek) if (vis["l_omuz"] and vis["l_dirsek"]) else None
        l_dirsek_aci = self._calculate_angle_2d(l_omuz, l_dirsek, l_bilek) if (
                    vis["l_omuz"] and vis["l_dirsek"] and vis["l_bilek"]) else None
        l_govde = self._calculate_gravity_angle_2d(l_omuz, l_kalca) if (vis["l_omuz"] and vis["l_kalca"]) else None
        l_silkme = self._get_shoulder_shrug_angle(l_omuz, r_omuz) if (vis["l_omuz"] and vis["r_omuz"]) else None

        noktalar_px = {
            "r_omuz": (int(r_omuz[0]), int(r_omuz[1])), "r_dirsek": (int(r_dirsek[0]), int(r_dirsek[1])),
            "r_bilek": (int(r_bilek[0]), int(r_bilek[1])), "r_kalca": (int(r_kalca[0]), int(r_kalca[1])),
            "l_omuz": (int(l_omuz[0]), int(l_omuz[1])), "l_dirsek": (int(l_dirsek[0]), int(l_dirsek[1])),
            "l_bilek": (int(l_bilek[0]), int(l_bilek[1])), "l_kalca": (int(l_kalca[0]), int(l_kalca[1]))
        }

        metrics_r = (r_kalkis, r_dirsek_aci, r_govde, r_silkme)
        metrics_l = (l_kalkis, l_dirsek_aci, l_govde, l_silkme)

        return metrics_r, metrics_l, egilme_orani, noktalar_px, vis


# ==========================================
# ANA KAMERA DÖNGÜSÜ
# ==========================================
mp_pose = mp.solutions.pose
preprocessor = LateralRaisePreprocessor()

pencere_adi = 'Lateral Raise AI - Egilme Testi'
cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
cv2.resizeWindow(pencere_adi, 1280, 720)

url = "http://192.168.137.40:8080/video"
cap = cv2.VideoCapture(url)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            metrics_r, metrics_l, egilme_orani, noktalar, vis = preprocessor.process_frame(
                results.pose_landmarks.landmark, w, h)

            # --- ÇİFT KOL İSKELET ÇİZİMİ ---
            if vis["r_omuz"] and vis["r_dirsek"]: cv2.line(frame, noktalar["r_omuz"], noktalar["r_dirsek"], (0, 255, 0),
                                                           5)
            if vis["r_dirsek"] and vis["r_bilek"]: cv2.line(frame, noktalar["r_dirsek"], noktalar["r_bilek"],
                                                            (0, 255, 0), 5)
            if vis["r_omuz"] and vis["r_kalca"]: cv2.line(frame, noktalar["r_omuz"], noktalar["r_kalca"], (255, 100, 0),
                                                          4)

            if vis["l_omuz"] and vis["l_dirsek"]: cv2.line(frame, noktalar["l_omuz"], noktalar["l_dirsek"], (0, 255, 0),
                                                           5)
            if vis["l_dirsek"] and vis["l_bilek"]: cv2.line(frame, noktalar["l_dirsek"], noktalar["l_bilek"],
                                                            (0, 255, 0), 5)
            if vis["l_omuz"] and vis["l_kalca"]: cv2.line(frame, noktalar["l_omuz"], noktalar["l_kalca"], (255, 100, 0),
                                                          4)

            if vis["r_omuz"] and vis["l_omuz"]: cv2.line(frame, noktalar["r_omuz"], noktalar["l_omuz"], (0, 255, 255),
                                                         2)

            for pt_key in noktalar:
                if vis[pt_key]: cv2.circle(frame, noktalar[pt_key], 6, (0, 0, 255), -1)

            # --- EKRANA DEĞERLERİ YAZDIRMA ---
            renk_sari = (0, 255, 255)
            renk_cyan = (255, 255, 0)
            renk_pembe = (255, 100, 255)


            def fmt(val):
                return f"{int(val)}" if val is not None else "-"


            str_kalkis = f"{fmt(metrics_r[0])} / {fmt(metrics_l[0])}"
            str_dirsek = f"{fmt(metrics_r[1])} / {fmt(metrics_l[1])}"
            str_govde = f"{fmt(metrics_r[2])} / {fmt(metrics_l[2])}"
            str_silkme = f"{fmt(metrics_r[3])} / {fmt(metrics_l[3])}"

            # Yeni 5. Metrik (Ondalıklı yazdırıyoruz çünkü küçük bir oran)
            str_egilme = f"{egilme_orani:.2f}" if egilme_orani is not None else "KADRAJDA YOK"

            texts = [
                (f"1. Kol Kalkis Acisi (Sag/Sol): {str_kalkis}", 40, renk_sari),
                (f"2. Dirsek Bukulme (Sag/Sol): {str_dirsek}", 80, renk_cyan),
                (f"3. Govde Salinim (Sag/Sol): {str_govde}", 120, renk_sari),
                (f"4. Omuz Silkme (Sag/Sol): {str_silkme}", 160, renk_cyan),
                (f"5. Sentetik Z-Ekseni (Diklik Orani): {str_egilme}", 200, renk_pembe)
            ]

            for text, y_pos, color in texts:
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(pencere_adi, frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()