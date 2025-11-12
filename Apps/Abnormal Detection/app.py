# streamlit_abnormal_detector_fusion.py
# Complete ready-to-run script — shows only abnormal frames (red boxes) and logs them.

import os
import cv2
import math
import logging
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import deque

# Optional imports guarded
try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except Exception:
    YOLO = None
    HAVE_YOLO = False

try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except Exception:
    mp = None
    HAVE_MEDIAPIPE = False

try:
    from fer import FER
    HAVE_FER = True
except Exception:
    FER = None
    HAVE_FER = False

# ---------------- logging ----------------
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file_txt = os.path.join(log_directory, "activity_log.txt")
log_file_csv = os.path.join(log_directory, "activity_log.csv")

logging.basicConfig(
    filename=log_file_txt,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_to_csv(activity: str, details: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, activity, details]], columns=["Timestamp", "Activity", "Details"])
    if not os.path.exists(log_file_csv):
        entry.to_csv(log_file_csv, index=False)
    else:
        entry.to_csv(log_file_csv, mode="a", header=False, index=False)

# ---------------- config ----------------
DEFAULT_YOLO_MODEL = "yolov8n-pose.pt"
PROCESS_WIDTH = 960
MIN_BOX_AREA = 800
MATCH_DIST_FACTOR = 0.5
HISTORY_LEN = 12
GLOBAL_BUF = 400
SUSTAIN_FRAMES = 5
MIN_WRIST_MIN = 0.007
MAD_Z_THRESH = 3.0

EAR_THRESH = 0.21
EAR_CONSEC_FRAMES = 12

# ---------------- utils ----------------
def clip_box(box, W, H):
    x1 = max(0, min(W-1, int(box[0]))); y1 = max(0, min(H-1, int(box[1])))
    x2 = max(0, min(W-1, int(box[2]))); y2 = max(0, min(H-1, int(box[3])))
    return (x1, y1, x2, y2)

def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def mad_to_std(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(1e-6, 1.4826 * mad)

# EAR helpers (MediaPipe FaceMesh indices)
LM_LEFT_EYE = [33, 160, 158, 133, 153, 144]
LM_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(pts):
    p0, p1, p2, p3, p4, p5 = pts
    A = np.linalg.norm(p1 - p5)
    B = np.linalg.norm(p2 - p4)
    C = np.linalg.norm(p0 - p3)
    if C <= 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Abnormal Behavior Detector", page_icon="🎥", layout="wide")
st.title("🎥 Abnormal Behavior Detector — show ONLY abnormal frames")

st.sidebar.header("⚙️ Settings")
source = st.sidebar.radio("Input Source", ("Upload Video", "Webcam"))
backend = st.sidebar.selectbox("Skeleton Backend", ("YOLOv8 Pose", "MediaPipe Pose"), index=0 if HAVE_YOLO else 1)

# YOLO model selection
custom_model_path = None
if backend == "YOLOv8 Pose":
    st.sidebar.caption("YOLOv8 Pose (multi-person)")
    model_choice = st.sidebar.selectbox("YOLOv8 pose model", (DEFAULT_YOLO_MODEL, "yolov8s-pose.pt", "yolov8m-pose.pt", "Upload .pt"))
    if model_choice == "Upload .pt":
        up = st.sidebar.file_uploader("Upload YOLO .pt model", type=["pt"])
        if up is not None:
            tmodel = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmodel.write(up.read())
            custom_model_path = tmodel.name
    else:
        custom_model_path = model_choice

st.sidebar.subheader("Modules")
use_skeleton_abn = st.sidebar.checkbox("Skeleton Movement Anomaly", value=True)
use_face_emotion = st.sidebar.checkbox("Face Emotion (FER)", value=False, disabled=not HAVE_FER)
use_mtcnn = st.sidebar.checkbox("Use MTCNN for FER (slower, more accurate on faces)", value=False, disabled=not HAVE_FER)

use_face_drowsy = st.sidebar.checkbox("Face Drowsiness (EAR)", value=HAVE_MEDIAPIPE)

st.sidebar.subheader("Thresholds")
MAD_Z_THRESH = st.sidebar.slider("MAD z-score (movement)", 2.0, 6.0, MAD_Z_THRESH, 0.5)
SUSTAIN_FRAMES = st.sidebar.slider("Sustain frames (movement)", 3, 20, SUSTAIN_FRAMES, 1)
if use_face_drowsy:
    EAR_THRESH = st.sidebar.slider("EAR threshold (eyes closed)", 0.15, 0.30, EAR_THRESH, 0.01)
    EAR_CONSEC_FRAMES = st.sidebar.slider("Frames for drowsy", 5, 30, EAR_CONSEC_FRAMES, 1)

st.sidebar.subheader("Logging")
if os.path.exists(log_file_csv):
    with open(log_file_csv, "rb") as fh:
        st.sidebar.download_button("📥 Download Logs (CSV)", fh, file_name="activity_log.csv")
if st.sidebar.button("🧹 Clear logs"):
    for p in (log_file_csv, log_file_txt):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    st.sidebar.success("Logs cleared")

# input source widget
uploaded_file = None
if source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
else:
    st.sidebar.info("Webcam uses device index 0")

# placeholders
col1, col2 = st.columns([3,1])
frame_placeholder = col1.empty()
status_placeholder = col1.empty()
progress_bar = st.progress(0)
metrics_box = col2.container()

# ---------------- initialize backend ----------------
pose_model = None
mp_pose = None
mp_draw = None
mp_face_mesh = None
emotion_detector = None

if backend == "YOLOv8 Pose":
    if not HAVE_YOLO:
        st.error("Ultralytics not installed. pip install ultralytics")
        st.stop()
    if not custom_model_path:
        st.error("Please choose or upload a YOLOv8 pose model in sidebar")
        st.stop()
    st.info(f"Loading YOLO model: {os.path.basename(custom_model_path)} ...")
    pose_model = YOLO(custom_model_path)
else:
    if not HAVE_MEDIAPIPE:
        st.error("MediaPipe not installed. pip install mediapipe")
        st.stop()
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                              enable_segmentation=False, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
    if use_face_drowsy:
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2)

if use_face_emotion and HAVE_FER:
    emotion_detector = FER(mtcnn=use_mtcnn)

# ---------------- processing functions ----------------
def process_yolo_video(cap, W, H, total_frames):
    tracks = {}
    next_id = 0
    global_wrist_buf = deque(maxlen=GLOBAL_BUF)
    global_torso_buf = deque(maxlen=GLOBAL_BUF)
    frame_idx = 0
    abnormal_total = 0

    # per-track drowsy counters
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_proc = cv2.resize(frame, (W, H)) if (frame.shape[1] != W) else frame.copy()

        results = pose_model(frame_proc, verbose=False)

        # --- parse detections (robust handling) ---
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            kps = getattr(r, "keypoints", None)
            if boxes is None or kps is None or boxes.xyxy is None or kps.xy is None:
                continue
            xyxy = boxes.xyxy
            confs = getattr(boxes, "conf", None)
            clss = getattr(boxes, "cls", None)
            kparr = kps.xy  # (num, num_kpts, 2)

            num = min(len(xyxy), len(kparr))
            for i in range(num):
                x1, y1, x2, y2 = map(int, xyxy[i].detach().cpu().numpy())
                conf = float(confs[i].item()) if confs is not None else 1.0
                cls  = int(clss[i].item()) if clss is not None else 0
                if cls != 0 or conf < 0.25:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    continue
                kpt_i = kparr[i].detach().cpu().numpy()
                if kpt_i is None or kpt_i.shape[0] < 13:
                    continue
                detections.append(((x1, y1, x2, y2), kpt_i, conf))
        # --- end parse detections ---
        
        # now you can continue with tracking and abnormality detection...
        # update/assign tracks
        det_used, track_used = set(), set()
        for i, (bbox, kps, conf) in enumerate(detections):
            cx, cy = center_of(bbox)
            best_t, best_dist = None, float('inf')
            for t_id, t in tracks.items():
                if t_id in track_used: continue
                tx, ty = t['center_hist'][-1] if t['center_hist'] else center_of(t['bbox'])
                bw = (bbox[2]-bbox[0]); tw = (t['bbox'][2]-t['bbox'][0])
                max_dist = max(40, (bw+tw)/2.0 * MATCH_DIST_FACTOR)
                d = math.hypot(cx-tx, cy-ty)
                if d < max_dist and d < best_dist:
                    best_dist = d; best_t = t_id
            if best_t is not None:
                t = tracks[best_t]
                t['bbox'] = bbox; t['last_seen'] = frame_idx
                t['center_hist'].append((cx, cy))
                if len(t['center_hist']) > HISTORY_LEN: t['center_hist'].pop(0)

                h = max(1.0, bbox[3]-bbox[1])
                try:
                    cur_wrists = kps[[9,10], :]; cur_shoulders = kps[[5,6], :]; cur_hips = kps[[11,12], :]
                except Exception:
                    cur_wrists = np.zeros((2,2)); cur_shoulders = np.zeros((2,2)); cur_hips = np.zeros((2,2))
                prev_kps = t.get('last_kps')
                if prev_kps is not None:
                    prev_wrists = prev_kps[[9,10], :]; prev_shoulders = prev_kps[[5,6], :]; prev_hips = prev_kps[[11,12], :]
                    wrist_disp = np.nanmean(np.linalg.norm(cur_wrists - prev_wrists, axis=1))
                    torso_disp = np.nanmean(np.linalg.norm(((cur_shoulders+cur_hips)/2) - ((prev_shoulders+prev_hips)/2), axis=1))
                else:
                    wrist_disp, torso_disp = 0.0, 0.0
                wrist_norm, torso_norm = wrist_disp/h, torso_disp/h
                t['last_kps'] = kps.copy()
                t['wrist_hist'].append(wrist_norm); t['torso_hist'].append(torso_norm)
                if len(t['wrist_hist']) > HISTORY_LEN: t['wrist_hist'].pop(0)
                if len(t['torso_hist']) > HISTORY_LEN: t['torso_hist'].pop(0)
                det_used.add(i); track_used.add(best_t)
                global_wrist_buf.append(wrist_norm); global_torso_buf.append(torso_norm)
        # create new tracks
        for i, (bbox,kps,conf) in enumerate(detections):
            if i in det_used: continue
            cx, cy = center_of(bbox)
            tracks[next_id] = {
                'bbox': bbox, 'last_seen': frame_idx,
                'center_hist': [(cx,cy)], 'last_kps': kps.copy(),
                'wrist_hist': [], 'torso_hist': [],
                'abn_counter': 0, 'abnormal': False, 'abn_conf': 0,
                'drowsy_counter': 0, 'last_emotion': None
            }
            next_id += 1

        # remove stale
        stale = [t_id for t_id,t in tracks.items() if frame_idx - t['last_seen'] > 40]
        for t_id in stale: tracks.pop(t_id, None)

        # global stats
        wrist_arr = np.array(global_wrist_buf) if len(global_wrist_buf)>0 else np.array([0.0])
        torso_arr = np.array(global_torso_buf) if len(global_torso_buf)>0 else np.array([0.0])
        global_w_med, global_t_med = float(np.median(wrist_arr)), float(np.median(torso_arr))
        global_w_mad, global_t_mad = mad_to_std(wrist_arr), mad_to_std(torso_arr)

        out = frame_proc.copy()
        any_abnormal = False

        # face processing per track (FER or FaceMesh) and anomaly decision
        for t_id, t in tracks.items():
            bx1,by1,bx2,by2 = clip_box(t['bbox'], W, H)
            wrist_val = np.mean(t['wrist_hist']) if len(t['wrist_hist'])>0 else 0.0
            torso_val = np.mean(t['torso_hist']) if len(t['torso_hist'])>0 else 0.0
            z_w = (wrist_val - global_w_med) / (global_w_mad + 1e-9)
            z_t = (torso_val - global_t_med) / (global_t_mad + 1e-9)

            is_abn, conf_score = False, 0.0
            if use_skeleton_abn:
                if z_w > MAD_Z_THRESH and wrist_val > MIN_WRIST_MIN:
                    is_abn = True; conf_score = max(conf_score, min(1.0, z_w/(MAD_Z_THRESH*2)))
                if z_t > MAD_Z_THRESH:
                    is_abn = True; conf_score = max(conf_score, min(1.0, z_t/(MAD_Z_THRESH*2)))

            # face ROI
            person_roi = out[by1:by2, bx1:bx2] if (bx2>bx1 and by2>by1) else None

            # emotion (only flag certain emotions)
            emotion_flag = False
            if use_face_emotion and emotion_detector is not None and person_roi is not None and person_roi.size>0:
                try:
                    em_results = emotion_detector.detect_emotions(person_roi)
                    if em_results:
                        scores = em_results[0]['emotions']
                        dominant = max(scores, key=scores.get)
                        t['last_emotion'] = dominant
                        if dominant in ("angry","fear","disgust","sad"):
                            emotion_flag = True
                except Exception:
                    pass

            # drowsiness via FaceMesh
            drowsy_flag = False
            if use_face_drowsy and mp_face_mesh is not None and person_roi is not None and person_roi.size>0:
                rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                fm_res = mp_face_mesh.process(rgb)
                if fm_res.multi_face_landmarks:
                    lm = fm_res.multi_face_landmarks[0].landmark
                    h_roi, w_roi = person_roi.shape[:2]
                    def pick(idx_list):
                        return np.array([[lm[i].x * w_roi, lm[i].y * h_roi] for i in idx_list], dtype=np.float32)
                    left_eye = pick(LM_LEFT_EYE); right_eye = pick(LM_RIGHT_EYE)
                    ear_left = eye_aspect_ratio(left_eye); ear_right = eye_aspect_ratio(right_eye)
                    ear = (ear_left + ear_right) / 2.0
                    if ear < EAR_THRESH:
                        t['drowsy_counter'] = t.get('drowsy_counter', 0) + 1
                    else:
                        t['drowsy_counter'] = 0
                    if t['drowsy_counter'] >= EAR_CONSEC_FRAMES:
                        drowsy_flag = True
                else:
                    t['drowsy_counter'] = 0

            # fusion: decide final abnormal state
            # only consider emotion_flag/drowsy_flag if modules enabled
            fused_abn = False
            fused_reasons = []
            if is_abn:
                fused_abn = True
                fused_reasons.append("Movement")
            if emotion_flag:
                fused_abn = True
                fused_reasons.append(f"Emotion:{t.get('last_emotion')}")
            if drowsy_flag:
                fused_abn = True
                fused_reasons.append("Drowsy")

            # sustain logic for movement-only cases
            if is_abn:
                t['abn_counter'] = t.get('abn_counter',0) + 1
            else:
                # if not movement abnormal, decay counter but don't clear emotional/drowsy immediate flags
                t['abn_counter'] = max(0, t.get('abn_counter',0) - 1)

            should_alert = False
            label_lines = []
            color = None

            # If movement sustained long enough OR other flags present => alert
            if (t['abn_counter'] >= SUSTAIN_FRAMES) or emotion_flag or drowsy_flag:
                t['abn_conf'] = int(conf_score*100)
                if t['abn_conf']>=80:
                   should_alert = True
                # log only once when abnormal becomes active
                if not t.get('abnormal', False):
                    abnormal_total += 1
                    logging.info(f"Abnormal detected (ID {t_id}) reasons: {','.join(fused_reasons)}")
                    log_to_csv("Abnormal", f"ID {t_id} — {';'.join(fused_reasons)}")
                t['abnormal'] = True
                
                color = (0,0,255)
                # build meaningful labels
                if "Movement" in fused_reasons:
                    label_lines.append(f"ID{t_id} ABNORMAL ({t['abn_conf']}%)")
                for r in fused_reasons:
                    if r.startswith("Emotion:"):
                        label_lines.append(r.replace("Emotion:","Face: "))
                    elif r == "Drowsy":
                        label_lines.append("DROWSY")
            else:
                t['abnormal'] = False

            # draw only if abnormal
            if should_alert and color is not None:
                any_abnormal = True
                cv2.rectangle(out, (bx1,by1), (bx2,by2), color, 2)
                y0 = max(14, by1 - 6)
                for k, text in enumerate(label_lines):
                    cv2.putText(out, text, (bx1, y0 - 16*k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # display only if any_abnormal detected
        frame_placeholder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")
        if any_abnormal:
            
            status_placeholder.info(f"⚠️ Abnormal detected on frame {frame_idx} (total alerts: {abnormal_total})")
        else:
            # skip showing normal frame — show small message instead
            
            status_placeholder.info(f"Frame {frame_idx}: normal (skipped)")

        if total_frames and frame_idx % 3 == 0:
            progress_bar.progress(min(max(frame_idx / total_frames, 0.0), 1.0))

    # end of while loop

    # --- cleanup ---
    cap.release()
    if 'face_mesh' in locals() and face_mesh:
        face_mesh.close()

    metrics_box.metric("🚨 Abnormal Events", abnormal_total)
    status_placeholder.success("Processing finished.")
    progress_bar.progress(1.0)




def process_mediapipe_video(cap, W, H, total_frames):
    frame_idx = 0
    wrist_buf = deque(maxlen=HISTORY_LEN)
    torso_buf = deque(maxlen=HISTORY_LEN)
    global_wrist_buf = deque(maxlen=GLOBAL_BUF)
    global_torso_buf = deque(maxlen=GLOBAL_BUF)
    abnormal_total = 0
    drowsy_counter = 0

    # Init FaceMesh if needed
    face_mesh = None
    if use_face_drowsy and HAVE_MEDIAPIPE:
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_proc = cv2.resize(frame, (W, H)) if (frame.shape[1] != W) else frame.copy()
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)
        out = frame_proc.copy()

        any_abnormal = False

        if results.pose_landmarks:
            mp_draw.draw_landmarks(out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

            h, w = out.shape[:2]
            lm = results.pose_landmarks.landmark
            pts = lambda idxs: np.array([[lm[i].x * w, lm[i].y * h] for i in idxs], dtype=np.float32)
            wrists = pts([15, 16]); shoulders = pts([11, 12]); hips = pts([23, 24])

            all_xy = np.vstack([wrists, shoulders, hips])
            x1, y1 = np.min(all_xy, axis=0).astype(int); x2, y2 = np.max(all_xy, axis=0).astype(int)
            x1 = max(0, x1 - 20); y1 = max(0, y1 - 40); x2 = min(w - 1, x2 + 20); y2 = min(h - 1, y2 + 40)
            hbox = max(1.0, (y2 - y1))

            # wrist/torso displacement
            if 'prev_wrists' in st.session_state:
                wrist_disp = np.nanmean(np.linalg.norm(wrists - st.session_state.prev_wrists, axis=1))
                torso_disp = np.nanmean(np.linalg.norm(((shoulders + hips) / 2) -
                                                       ((st.session_state.prev_shoulders + st.session_state.prev_hips) / 2), axis=1))
            else:
                wrist_disp, torso_disp = 0.0, 0.0

            st.session_state.prev_wrists = wrists.copy()
            st.session_state.prev_shoulders = shoulders.copy()
            st.session_state.prev_hips = hips.copy()

            wrist_norm = wrist_disp / hbox
            torso_norm = torso_disp / hbox
            wrist_buf.append(wrist_norm); torso_buf.append(torso_norm)
            global_wrist_buf.append(wrist_norm); global_torso_buf.append(torso_norm)

            # stats
            w_arr = np.array(global_wrist_buf) if len(global_wrist_buf) > 0 else np.array([0.0])
            t_arr = np.array(global_torso_buf) if len(global_torso_buf) > 0 else np.array([0.0])
            w_med, t_med = float(np.median(w_arr)), float(np.median(t_arr))
            w_mad, t_mad = mad_to_std(w_arr), mad_to_std(t_arr)
            w_val = float(np.mean(wrist_buf)) if len(wrist_buf) else 0.0
            t_val = float(np.mean(torso_buf)) if len(torso_buf) else 0.0
            z_w = (w_val - w_med) / (w_mad + 1e-9)
            z_t = (t_val - t_med) / (t_mad + 1e-9)

            is_abn = False; conf_score = 0.0
            if use_skeleton_abn:
                if z_w > MAD_Z_THRESH and w_val > MIN_WRIST_MIN:
                    is_abn = True; conf_score = max(conf_score, min(1.0, z_w / (MAD_Z_THRESH * 2)))
                if z_t > MAD_Z_THRESH:
                    is_abn = True; conf_score = max(conf_score, min(1.0, z_t / (MAD_Z_THRESH * 2)))

            # face ROI
            person_roi = out[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

            # emotion
            emotion_flag = False
            last_emotion = None
            if use_face_emotion and emotion_detector is not None and person_roi is not None and person_roi.size > 0:
                try:
                    em_results = emotion_detector.detect_emotions(person_roi)
                    if em_results:
                        scores = em_results[0]['emotions']
                        last_emotion = max(scores, key=scores.get)
                        if last_emotion in ("angry", "fear", "disgust", "sad"):
                            emotion_flag = True
                except Exception:
                    pass

            # drowsiness
            drowsy_flag = False
            if use_face_drowsy and face_mesh is not None and person_roi is not None and person_roi.size > 0:
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                fm_res = face_mesh.process(rgb_roi)
                if fm_res.multi_face_landmarks:
                    lm2 = fm_res.multi_face_landmarks[0].landmark
                    hh, ww = person_roi.shape[:2]
                    def pick(ids):
                        return np.array([[lm2[i].x * ww, lm2[i].y * hh] for i in ids], dtype=np.float32)
                    le = pick(LM_LEFT_EYE); re = pick(LM_RIGHT_EYE)
                    ear = (eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0
                    if ear < EAR_THRESH:
                        drowsy_counter += 1
                    else:
                        drowsy_counter = 0
                    if drowsy_counter >= EAR_CONSEC_FRAMES:
                        drowsy_flag = True
                else:
                    drowsy_counter = 0

            # --------- Fusion ---------
            alerting_now = False
            fused_reasons = []
            labels = []
            color = None

            if is_abn and int(conf_score * 100) >= 80:
                alerting_now = True
                fused_reasons.append("Movement")

            if emotion_flag:
                alerting_now = True
                fused_reasons.append(f"Emotion:{last_emotion}")

            if drowsy_flag:
                alerting_now = True
                fused_reasons.append("Drowsy")

            prev_alerting = st.session_state.get("mp_prev_alerting", False)

            if alerting_now and not prev_alerting:
                abnormal_total += 1
                logging.info(f"Abnormal (MediaPipe) frame {frame_idx}: {','.join(fused_reasons)}")
                log_to_csv("Abnormal", f"ID0 — {','.join(fused_reasons)}")
                color = (0, 0, 255)
                if "Movement" in fused_reasons:
                    labels.append(f"ID0 ABNORMAL ({int(conf_score*100)}%)")
                for r in fused_reasons:
                    if r.startswith("Emotion:"):
                        labels.append(r.replace("Emotion:", "Face: "))
                    elif r == "Drowsy":
                        labels.append("DROWSY")

                # draw abnormal box
                cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
                for k, text in enumerate(labels):
                    cv2.putText(out, text, (x1, max(14, y1-6) - 16*k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                any_abnormal = True

            st.session_state.mp_prev_alerting = alerting_now

        # ----- display -----
        frame_placeholder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")
        if any_abnormal:
            
            status_placeholder.info(f"⚠️ Abnormal detected on frame {frame_idx} (alerts: {abnormal_total})")
        else:
            
            status_placeholder.info(f"Frame {frame_idx}: normal (skipped)")

        if total_frames and frame_idx % 3 == 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    metrics_box.metric("🚨 Abnormal Events", abnormal_total)
    status_placeholder.success("Processing finished.")

# ---------------- Run ----------------
start_clicked = st.sidebar.button("▶️ Start")

if source == "Upload Video":
    if uploaded_file is None:
        st.info("📂 Please upload a video file to begin.")
        st.stop()
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    tmpf.write(uploaded_file.read())
    cap = cv2.VideoCapture(tmpf.name)
else:
    cap = cv2.VideoCapture(0)

if not start_clicked:
    st.warning("Click ▶️ Start to begin processing.")
    st.stop()

if not cap or not cap.isOpened():
    st.error("❌ Cannot open video source")
    st.stop()

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or PROCESS_WIDTH)
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (PROCESS_WIDTH*9//16))
try:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
except Exception:
    total_frames = None
scale = PROCESS_WIDTH / orig_w if orig_w > PROCESS_WIDTH else 1.0
W = int(orig_w * scale); H = int(orig_h * scale)

st.success("✅ Processing started...")

if backend == "YOLOv8 Pose":
    process_yolo_video(cap, W, H, total_frames)
else:
    process_mediapipe_video(cap, W, H, total_frames)

st.success("✅ All done!")

# cleanup only for uploaded file
if source == "Upload Video":
    try:
        os.unlink(tmpf.name)  # delete temp file
    except Exception:
        pass
