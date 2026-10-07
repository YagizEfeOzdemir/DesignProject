import cv2
import mediapipe as mp
import numpy as np

class BicepsCurlPreprocessor:
    def _init_(self):
        # Y ekseni (Yerçekimi) referansı
        self.gravity_vector_2d = np.array([0.0, 1.0])
        # Güvenilirlik sınırı
        self.visibility_threshold = 0.6

    def _calculate_angle_2d(self, p1, p2, p3):
        """Sadece X ve Y piksellerini kullanarak KUSURSUZ 2D açı hesaplar"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _calculate_gravity_angle_2d(self, p_omuz, p_dirsek):
        """Omuz ve dirseğin aynı X hizasında (Y'ye dik) kalıp kalmadığını ölçer"""
        v_kol = np.array([p_dirsek[0] - p_omuz[0], p_dirsek[1] - p_omuz[1]])
        if np.linalg.norm(v_kol) == 0: return 0.0
        cosine_angle = np.dot(v_kol, self.gravity_vector_2d) / np.linalg.norm(v_kol)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def process_frame(self, pose_landmarks, w, h):
        """Artık w (genişlik) ve h (yükseklik) alarak pikseller üzerinden hesap yapıyor"""
        vis_right = pose_landmarks[12].visibility + pose_landmarks[14].visibility
        vis_left = pose_landmarks[11].visibility + pose_landmarks[13].visibility

        # Hangi kol daha netse onu seç
        if vis_right > vis_left:
            omuz_idx, dirsek_idx, bilek_idx, parmak_idx = 12, 14, 16, 20
        else:
            omuz_idx, dirsek_idx, bilek_idx, parmak_idx = 11, 13, 15, 19

        vis = {
            "omuz": pose_landmarks[omuz_idx].visibility > self.visibility_threshold,
            "dirsek": pose_landmarks[dirsek_idx].visibility > self.visibility_threshold,
            "bilek": pose_landmarks[bilek_idx].visibility > self.visibility_threshold,
            "parmak": pose_landmarks[parmak_idx].visibility > self.visibility_threshold
        }

        # Z EKSENİ ÇÖPE ATILDI! Doğrudan ekran piksellerine (w, h) çevrildi.
        omuz = np.array([pose_landmarks[omuz_idx].x * w, pose_landmarks[omuz_idx].y * h])
        dirsek = np.array([pose_landmarks[dirsek_idx].x * w, pose_landmarks[dirsek_idx].y * h])
        bilek = np.array([pose_landmarks[bilek_idx].x * w, pose_landmarks[bilek_idx].y * h])
        parmak = np.array([pose_landmarks[parmak_idx].x * w, pose_landmarks[parmak_idx].y * h])

        # Sadece görünürlük yüksekse hesapla (Görünmüyorsa None döner)
        bilek_acisi = self._calculate_angle_2d(dirsek, bilek, parmak) if (
                    vis["dirsek"] and vis["bilek"] and vis["parmak"]) else None
        biceps_acisi = self._calculate_angle_2d(omuz, dirsek, bilek) if (
                    vis["omuz"] and vis["dirsek"] and vis["bilek"]) else None
        yercekimi_acisi = self._calculate_gravity_angle_2d(omuz, dirsek) if (vis["omuz"] and vis["dirsek"]) else None

        # Çizim için pikselleri hazırla (Main loop'u rahatlatır)
        noktalar_px = {
            "omuz": (int(omuz[0]), int(omuz[1])),
            "dirsek": (int(dirsek[0]), int(dirsek[1])),
            "bilek": (int(bilek[0]), int(bilek[1])),
            "parmak": (int(parmak[0]), int(parmak[1]))
        }

        return bilek_acisi, biceps_acisi, yercekimi_acisi, noktalar_px, vis


# ==========================================
# ANA KAMERA DÖNGÜSÜ
# ==========================================
mp_pose = mp.solutions.pose
preprocessor = BicepsCurlPreprocessor()

# Senin IP Webcam bağlantın
url = "http://192.168.137.40:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Bağlantı hatası: IP adresini ve aynı ağda olduğunuzu kontrol edin.")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            print("Görüntü akışı kesildi.")
            break

        # --- 1. YÖNÜ DÜZELTME ---
        qframe = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # --- 2. EKRANA SIĞDIRMA VE GECİKME ÖNLEME (YENİ EKLENEN KISIM) ---
        # Görüntü yüksekliğini 800 piksele sabitleyip, genişliği orantılı küçültüyoruz
        # Bu hem ekrandan taşmayı önler hem de MediaPipe'ın FPS'ini uçurur.
        hedef_yukseklik = 800
        oran = hedef_yukseklik / frame.shape[0]
        yeni_genislik = int(frame.shape[1] * oran)
        frame = cv2.resize(frame, (yeni_genislik, hedef_yukseklik))

        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ... (Kodun geri kalanı aynı şekilde devam ediyor)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            bilek_aci, biceps_aci, yercekimi_aci, noktalar, vis = preprocessor.process_frame(
                results.pose_landmarks.landmark, w, h)

            px_omuz = noktalar["omuz"]
            px_dirsek = noktalar["dirsek"]
            px_bilek = noktalar["bilek"]
            px_parmak = noktalar["parmak"]

            # --- SADECE GÖRÜNEN KEMİKLERİ ÇİZ ---
            if vis["omuz"] and vis["dirsek"]:
                cv2.line(frame, px_omuz, px_dirsek, (0, 255, 0), 5)
            if vis["dirsek"] and vis["bilek"]:
                cv2.line(frame, px_dirsek, px_bilek, (0, 255, 0), 5)
            if vis["bilek"] and vis["parmak"]:
                cv2.line(frame, px_bilek, px_parmak, (0, 255, 0), 5)

            for pt, v in zip([px_omuz, px_dirsek, px_bilek, px_parmak],
                             [vis["omuz"], vis["dirsek"], vis["bilek"], vis["parmak"]]):
                if v: cv2.circle(frame, pt, 6, (0, 0, 255), -1)

            # --- EKRANA DEĞERLERİ YAZDIRMA ---
            renk_sari = (0, 255, 255)
            renk_cyan = (255, 255, 0)

            # Değer None ise ekrana "KADRAJDA YOK" yazdır
            str_bilek = f"{int(bilek_aci)}" if bilek_aci is not None else "KADRAJDA YOK"
            str_biceps = f"{int(biceps_aci)}" if biceps_aci is not None else "KADRAJDA YOK"
            str_yercekimi = f"{int(yercekimi_aci)}" if yercekimi_aci is not None else "KADRAJDA YOK"

            cv2.putText(frame, f"1. Isaret Parmagi/On Kol Acisi: {str_bilek}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4)
            cv2.putText(frame, f"1. Isaret Parmagi/On Kol Acisi: {str_bilek}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        renk_sari, 2)

            cv2.putText(frame, f"2. On Kol/Ust Kol Acisi: {str_biceps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4)
            cv2.putText(frame, f"2. On Kol/Ust Kol Acisi: {str_biceps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        renk_cyan, 2)

            cv2.putText(frame, f"3. Omuz/Dirsek Y Dikligi: {str_yercekimi}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4)
            cv2.putText(frame, f"3. Omuz/Dirsek Y Dikligi: {str_yercekimi}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        renk_sari, 2)

        cv2.imshow('Biceps AI Matematiksel Form Testi', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()