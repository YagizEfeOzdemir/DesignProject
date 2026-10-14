import cv2
import mediapipe as mp
from exercises.biceps_curl import BicepsCurlPreprocessor
from exercises.lateral_raise import LateralRaisePreprocessor

# ==========================================
# 1. AYARLAR
# ==========================================
SECILI_HAREKET = "BICEPS_CURL"  # "BICEPS_CURL" de yapabilirsin

if SECILI_HAREKET == "BICEPS_CURL":
    preprocessor = BicepsCurlPreprocessor()
elif SECILI_HAREKET == "LATERAL_RAISE":
    preprocessor = LateralRaisePreprocessor()

# ==========================================
# 2. KAMERA VE MEDIAPIPE BAŞLATMA
# ==========================================
# IP kameran takılıysa buraya IP adresini koyabilirsin
url ="https://192.168.137.213:8080/video"
cap = cv2.VideoCapture(url)

mp_pose = mp.solutions.pose
pencere_adi = f'Yapay Zeka Koc - OOP Mimari - {SECILI_HAREKET}'
cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
cv2.resizeWindow(pencere_adi, 1280, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Görüntü akışı kesildi.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        hedef_yukseklik = 800
        oran = hedef_yukseklik / h
        yeni_genislik = int(w * oran)
        frame = cv2.resize(frame, (yeni_genislik, hedef_yukseklik))
        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # --- HAREKET: BICEPS CURL ---
            if SECILI_HAREKET == "BICEPS_CURL":
                bilek_aci, biceps_aci, yercekimi_aci, noktalar, vis = preprocessor.process_frame(
                    results.pose_landmarks.landmark, w, h)

                px_omuz, px_dirsek, px_bilek, px_parmak = noktalar["omuz"], noktalar["dirsek"], noktalar["bilek"], \
                noktalar["parmak"]
                if vis["omuz"] and vis["dirsek"]: cv2.line(frame, px_omuz, px_dirsek, (0, 255, 0), 5)
                if vis["dirsek"] and vis["bilek"]: cv2.line(frame, px_dirsek, px_bilek, (0, 255, 0), 5)
                if vis["bilek"] and vis["parmak"]: cv2.line(frame, px_bilek, px_parmak, (0, 255, 0), 5)
                for pt, v in zip([px_omuz, px_dirsek, px_bilek, px_parmak],
                                 [vis["omuz"], vis["dirsek"], vis["bilek"], vis["parmak"]]):
                    if v: cv2.circle(frame, pt, 6, (0, 0, 255), -1)

                cv2.putText(frame, f"1. Isaret Parmagi Acisi: {int(bilek_aci) if bilek_aci else '-'}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"2. On Kol/Ust Kol Acisi: {int(biceps_aci) if biceps_aci else '-'}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"3. Omuz/Dirsek Y Dikligi: {int(yercekimi_aci) if yercekimi_aci else '-'}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- HAREKET: LATERAL RAISE ---
            elif SECILI_HAREKET == "LATERAL_RAISE":
                metrics_r, metrics_l, egilme_orani, noktalar, vis = preprocessor.process_frame(
                    results.pose_landmarks.landmark, w, h)

                if vis["r_omuz"] and vis["r_dirsek"]: cv2.line(frame, noktalar["r_omuz"], noktalar["r_dirsek"],
                                                               (0, 255, 0), 5)
                if vis["r_dirsek"] and vis["r_bilek"]: cv2.line(frame, noktalar["r_dirsek"], noktalar["r_bilek"],
                                                                (0, 255, 0), 5)
                if vis["r_omuz"] and vis["r_kalca"]: cv2.line(frame, noktalar["r_omuz"], noktalar["r_kalca"],
                                                              (255, 100, 0), 4)

                if vis["l_omuz"] and vis["l_dirsek"]: cv2.line(frame, noktalar["l_omuz"], noktalar["l_dirsek"],
                                                               (0, 255, 0), 5)
                if vis["l_dirsek"] and vis["l_bilek"]: cv2.line(frame, noktalar["l_dirsek"], noktalar["l_bilek"],
                                                                (0, 255, 0), 5)
                if vis["l_omuz"] and vis["l_kalca"]: cv2.line(frame, noktalar["l_omuz"], noktalar["l_kalca"],
                                                              (255, 100, 0), 4)

                if vis["r_omuz"] and vis["l_omuz"]: cv2.line(frame, noktalar["r_omuz"], noktalar["l_omuz"],
                                                             (0, 255, 255), 2)

                for pt_key in noktalar:
                    if vis[pt_key]: cv2.circle(frame, noktalar[pt_key], 6, (0, 0, 255), -1)


                def fmt(val):
                    return f"{int(val)}" if val is not None else "-"


                cv2.putText(frame, f"1. Kol Kalkis Acisi (Sag/Sol): {fmt(metrics_r[0])} / {fmt(metrics_l[0])}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"2. Dirsek Bukulme (Sag/Sol): {fmt(metrics_r[1])} / {fmt(metrics_l[1])}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"3. Govde Salinim (Sag/Sol): {fmt(metrics_r[2])} / {fmt(metrics_l[2])}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"4. Omuz Silkme (Sag/Sol): {fmt(metrics_r[3])} / {fmt(metrics_l[3])}", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame,
                            f"5. Sentetik Z-Ekseni: {egilme_orani:.2f}" if egilme_orani else "5. Sentetik Z-Ekseni: -",
                            (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)

        cv2.imshow(pencere_adi, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()