import cv2
import mediapipe as mp
import numpy as np
import math


class LateralRaisePreprocessor:
    def __init__(self):
        # Y ekseni (Yerçekimi) referansı - 2 Boyutlu
        self.gravity_vector_2d = np.array([0.0, 1.0])
        # Güvenilirlik sınırı (Kamera o uzvu %60'tan az görüyorsa işlemi keser)
        self.visibility_threshold = 0.6

    def _calculate_angle_2d(self, p1, p2, p3):
        """Sadece X ve Y piksellerini kullanarak KUSURSUZ 2D açı hesaplar"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _calculate_gravity_angle_2d(self, p_ust, p_alt):
        """Herhangi bir vektörün yerçekimi (Y) ekseni ile açısını ölçer"""
        v_kemik = np.array([p_alt[0] - p_ust[0], p_alt[1] - p_ust[1]])
        if np.linalg.norm(v_kemik) == 0: return 0.0
        cosine_angle = np.dot(v_kemik, self.gravity_vector_2d) / np.linalg.norm(v_kemik)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_shoulder_shrug_angle(self, omuz, karsi_omuz):
        """İki omuz arasındaki paralelliği (Trapez silkme) ölçer"""
        dy = karsi_omuz[1] - omuz[1]
        dx = abs(karsi_omuz[0] - omuz[0])
        if dx == 0: return 0.0
        return math.degrees(math.atan2(dy, dx))

    def process_frame(self, pose_landmarks, w, h):
        # 1. OTOMATİK KOL VE TARAF SEÇİMİ (Kameraya daha net bakan tarafı alır)
        vis_right = pose_landmarks[12].visibility + pose_landmarks[14].visibility
        vis_left = pose_landmarks[11].visibility + pose_landmarks[13].visibility

        if vis_right > vis_left:  # Sağ Taraf
            omuz_idx, dirsek_idx, bilek_idx, kalca_idx = 12, 14, 16, 24
            karsi_omuz_idx = 11
        else:  # Sol Taraf
            omuz_idx, dirsek_idx, bilek_idx, kalca_idx = 11, 13, 15, 23
            karsi_omuz_idx = 12

        # 2. GÖRÜNÜRLÜK KONTROLÜ
        vis = {
            "omuz": pose_landmarks[omuz_idx].visibility > self.visibility_threshold,
            "dirsek": pose_landmarks[dirsek_idx].visibility > self.visibility_threshold,
            "bilek": pose_landmarks[bilek_idx].visibility > self.visibility_threshold,
            "kalca": pose_landmarks[kalca_idx].visibility > self.visibility_threshold,
            "karsi_omuz": pose_landmarks[karsi_omuz_idx].visibility > self.visibility_threshold
        }

        # 3. Z EKSENİ SİLİNDİ, PİKSELLERE (X, Y) ÇEVRİLDİ
        omuz = np.array([pose_landmarks[omuz_idx].x * w, pose_landmarks[omuz_idx].y * h])
        dirsek = np.array([pose_landmarks[dirsek_idx].x * w, pose_landmarks[dirsek_idx].y * h])
        bilek = np.array([pose_landmarks[bilek_idx].x * w, pose_landmarks[bilek_idx].y * h])
        kalca = np.array([pose_landmarks[kalca_idx].x * w, pose_landmarks[kalca_idx].y * h])
        karsi_omuz = np.array([pose_landmarks[karsi_omuz_idx].x * w, pose_landmarks[karsi_omuz_idx].y * h])

        # 4. KİNEMATİK ZİNCİR MATEMATİĞİ (Sadece üst kol için kanıt üretiyoruz)
        ust_kol_vektoru = np.array([dirsek[0] - omuz[0], dirsek[1] - omuz[1]])
        ham_boy = np.linalg.norm(ust_kol_vektoru)

        if ham_boy > 0:
            ust_kol_birim = ust_kol_vektoru / ham_boy
        else:
            ust_kol_birim = np.array([0.0, 0.0])

        norm_boy = np.linalg.norm(ust_kol_birim) if ham_boy > 0 else 0.0

        # 5. AÇILARI HESAPLA (Modelin Öğreneceği Saf Veriler)
        # A. Üst Kolun Yere (Yerçekimine) Açısı (Kaldırma Miktarı)
        kol_kalkis_acisi = self._calculate_gravity_angle_2d(omuz, dirsek) if (vis["omuz"] and vis["dirsek"]) else None

        # B. Dirsek Açısı (Kolu Bükerek Hile Yapma)
        dirsek_acisi = self._calculate_angle_2d(omuz, dirsek, bilek) if (
                    vis["omuz"] and vis["dirsek"] and vis["bilek"]) else None

        # C. Gövde Salınım Açısı (Sallanarak / Belden Güç Alma)
        govde_salinim_acisi = self._calculate_gravity_angle_2d(omuz, kalca) if (vis["omuz"] and vis["kalca"]) else None

        # D. Omuz Silkme Açısı (Trapezden Güç Alma)
        omuz_silkme_acisi = self._get_shoulder_shrug_angle(omuz, karsi_omuz) if (
                    vis["omuz"] and vis["karsi_omuz"]) else None

        # Çizim için pikseller
        noktalar_px = {
            "omuz": (int(omuz[0]), int(omuz[1])),
            "dirsek": (int(dirsek[0]), int(dirsek[1])),
            "bilek": (int(bilek[0]), int(bilek[1])),
            "kalca": (int(kalca[0]), int(kalca[1])),
            "karsi_omuz": (int(karsi_omuz[0]), int(karsi_omuz[1]))
        }

        return kol_kalkis_acisi, dirsek_acisi, govde_salinim_acisi, omuz_silkme_acisi, noktalar_px, vis, ham_boy, norm_boy, ust_kol_birim


# ==========================================
# ANA KAMERA DÖNGÜSÜ
# ==========================================
mp_pose = mp.solutions.pose
preprocessor = LateralRaisePreprocessor()

# --- PENCERE AYARLARI (Tam Ekran / Genişletilebilir) ---
pencere_adi = 'Lateral Raise AI - Saf Veri Testi'
cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
cv2.resizeWindow(pencere_adi, 1280, 720)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            (kol_kalkis_aci, dirsek_aci, govde_salinim_aci, omuz_silkme_aci,
             noktalar, vis, ham_boy, norm_boy, birim_vektor) = preprocessor.process_frame(
                results.pose_landmarks.landmark, w, h)

            px_omuz = noktalar["omuz"]
            px_dirsek = noktalar["dirsek"]
            px_bilek = noktalar["bilek"]
            px_kalca = noktalar["kalca"]
            px_karsi = noktalar["karsi_omuz"]

            # --- SADECE GÖRÜNEN VE GEREKEN KEMİKLERİ ÇİZ ---
            # 1. Üst Kol ve Ön Kol (Yeşil)
            if vis["omuz"] and vis["dirsek"]:
                cv2.line(frame, px_omuz, px_dirsek, (0, 255, 0), 5)
            if vis["dirsek"] and vis["bilek"]:
                cv2.line(frame, px_dirsek, px_bilek, (0, 255, 0), 5)

                # 2. Gövde Hattı (Mavi) - Momentum kontrolü için
            if vis["omuz"] and vis["kalca"]:
                cv2.line(frame, px_omuz, px_kalca, (255, 100, 0), 4)

            # 3. İki Omuz Arası Paralellik Hattı (Sarı) - Trapez kontrolü için
            if vis["omuz"] and vis["karsi_omuz"]:
                cv2.line(frame, px_omuz, px_karsi, (0, 255, 255), 2)

            # Eklemleri (Kırmızı) çiz
            for pt, v in zip([px_omuz, px_dirsek, px_bilek, px_kalca, px_karsi],
                             [vis["omuz"], vis["dirsek"], vis["bilek"], vis["kalca"], vis["karsi_omuz"]]):
                if v: cv2.circle(frame, pt, 6, (0, 0, 255), -1)

            # --- EKRANA DEĞERLERİ YAZDIRMA (Sadece Saf Veri) ---
            renk_sari = (0, 255, 255)
            renk_cyan = (255, 255, 0)
            renk_pembe = (255, 100, 255)

            str_kalkis = f"{int(kol_kalkis_aci)}" if kol_kalkis_aci is not None else "KADRAJDA YOK"
            str_dirsek = f"{int(dirsek_aci)}" if dirsek_aci is not None else "KADRAJDA YOK"
            str_govde = f"{int(govde_salinim_aci)}" if govde_salinim_aci is not None else "KADRAJDA YOK"
            str_silkme = f"{int(omuz_silkme_aci)}" if omuz_silkme_aci is not None else "KADRAJDA YOK"

            cv2.putText(frame, f"1. Kol Kalkis Acisi (0=Asagi, 90=Yatay): {str_kalkis}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"1. Kol Kalkis Acisi (0=Asagi, 90=Yatay): {str_kalkis}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk_sari, 2)

            cv2.putText(frame, f"2. Dirsek Bukulme Acisi (Kiriklik): {str_dirsek}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"2. Dirsek Bukulme Acisi (Kiriklik): {str_dirsek}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, renk_cyan, 2)

            cv2.putText(frame, f"3. Govde Salinim (0=Dik Dusus): {str_govde}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4)
            cv2.putText(frame, f"3. Govde Salinim (0=Dik Dusus): {str_govde}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        renk_sari, 2)

            cv2.putText(frame, f"4. Omuz Silkme (Trapez) Sapmasi: {str_silkme}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"4. Omuz Silkme (Trapez) Sapmasi: {str_silkme}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, renk_cyan, 2)

            # --- KİNEMATİK ZİNCİR KANIT EKRANI ---
            str_ham = f"{ham_boy:.1f} piksel" if vis["omuz"] and vis["dirsek"] else "KADRAJDA YOK"
            str_norm = f"{norm_boy:.2f}" if vis["omuz"] and vis["dirsek"] else "KADRAJDA YOK"
            str_vec = f"X: {birim_vektor[0]:.2f}, Y: {birim_vektor[1]:.2f}" if vis["omuz"] and vis[
                "dirsek"] else "KADRAJDA YOK"

            cv2.putText(frame, "--- KINEMATIK ZINCIR (ESITLEME) KANITI ---", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            cv2.putText(frame, "--- KINEMATIK ZINCIR (ESITLEME) KANITI ---", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"Ham Boy: {str_ham}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, f"Ham Boy: {str_ham}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk_pembe, 2)

            cv2.putText(frame, f"AI Icin Boy: {str_norm}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, f"AI Icin Boy: {str_norm}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk_pembe, 2)

            cv2.putText(frame, f"Yapay Zekaya Giden Vektor Yonu: {str_vec}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            cv2.putText(frame, f"Yapay Zekaya Giden Vektor Yonu: {str_vec}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        renk_pembe, 2)

        cv2.imshow(pencere_adi, frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()