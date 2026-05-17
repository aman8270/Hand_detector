import cv2
import math
import os
import mediapipe as mp
import mediapipe.tasks as mp_tasks

# ── New Tasks API ─────────────────────────────────────────────────────────────
BaseOptions       = mp_tasks.BaseOptions
HandLandmarker    = mp_tasks.vision.HandLandmarker
HandLandmarkerOpt = mp_tasks.vision.HandLandmarkerOptions
RunningMode       = mp_tasks.vision.RunningMode

# ── Drawing helpers (still bundled under tasks.vision in 0.10.32) ────────────
drawing_utils  = mp_tasks.vision.drawing_utils
drawing_styles = mp_tasks.vision.drawing_styles

# ── Hand connections from tasks.vision ───────────────────────────────────────
HandLandmarksConnections = mp_tasks.vision.HandLandmarksConnections

# Landmark indices
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

# Hand-connection list needed by draw_landmarks
HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
])


class HandDetector:
    """
    Real-time hand detection using MediaPipe HandLandmarker (Tasks API).
    Compatible with mediapipe >= 0.10.

    Each detected hand is returned as a dict:
        {
            "lmList"  : [[x, y, z], ...],   # 21 points in pixel coords
            "bbox"    : (x, y, w, h),        # bounding box
            "center"  : (cx, cy),
            "type"    : "Left" | "Right",    # mirror-corrected
            "score"   : float,
        }
    """

    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hand_landmarker.task")

    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.5,
                 model_path=None):
        self.maxHands     = maxHands
        self.detectionCon = float(detectionCon)
        self.trackCon     = float(trackCon)
        model_path        = model_path or self.MODEL_PATH

        options = HandLandmarkerOpt(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_hand_presence_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.landmarker   = HandLandmarker.create_from_options(options)
        self._ts_ms       = 0  # monotonically increasing timestamp

    # ─────────────────────────────────────────────────────────────────────────
    def findHands(self, img, draw=True, flipType=True):
        """
        Detect hands in a BGR frame.
        Returns (list_of_hand_dicts, annotated_img).
        """
        self._ts_ms += 1
        h, w, _ = img.shape

        # MediaPipe needs an mp.Image object
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result  = self.landmarker.detect_for_video(mp_img, self._ts_ms)

        allHands = []

        if result.hand_landmarks:
            for lms, handedness in zip(result.hand_landmarks,
                                       result.handedness):
                # ── Pixel landmarks ──────────────────────────────────────────
                lmList = [[int(lm.x * w), int(lm.y * h), int(lm.z * w)]
                          for lm in lms]

                # ── Bounding box ─────────────────────────────────────────────
                xs = [p[0] for p in lmList]
                ys = [p[1] for p in lmList]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                bw, bh  = x2 - x1, y2 - y1
                center  = (x1 + bw // 2, y1 + bh // 2)

                # ── Handedness (mirror-corrected) ────────────────────────────
                label = handedness[0].category_name      # "Left" or "Right"
                if flipType:
                    label = "Left" if label == "Right" else "Right"

                allHands.append({
                    "lmList" : lmList,
                    "bbox"   : (x1, y1, bw, bh),
                    "center" : center,
                    "type"   : label,
                    "score"  : round(handedness[0].score, 2),
                })

                # ── Draw ─────────────────────────────────────────────────────
                if draw:
                    self._draw_landmarks(img, lms, w, h)
                    pad = 15
                    cv2.rectangle(img,
                                  (x1 - pad, y1 - pad),
                                  (x2 + pad, y2 + pad),
                                  (255, 0, 255), 2)
                    cv2.putText(img, f"{label}",
                                (x1 - pad, y1 - pad - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 0, 255), 2)

        return allHands, img

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_landmarks(self, img, lms, w, h):
        """Draw hand skeleton directly using cv2 (no protobuf dependency)."""
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Draw connections
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 200, 255), 2)

        # Draw landmark circles
        for i, pt in enumerate(pts):
            r = 6 if i not in TIP_IDS else 9
            color = (0, 255, 0) if i in TIP_IDS else (255, 255, 255)
            cv2.circle(img, pt, r, color, cv2.FILLED)
            cv2.circle(img, pt, r, (0, 0, 0), 1)

    # ─────────────────────────────────────────────────────────────────────────
    def fingersUp(self, hand: dict) -> list:
        """
        Returns [Thumb, Index, Middle, Ring, Pinky] — 1=up, 0=down.
        """
        lm        = hand["lmList"]
        hand_type = hand["type"]
        fingers   = []

        # Thumb (x-axis comparison, differs per hand)
        if hand_type == "Right":
            fingers.append(1 if lm[TIP_IDS[0]][0] > lm[TIP_IDS[0]-1][0] else 0)
        else:
            fingers.append(1 if lm[TIP_IDS[0]][0] < lm[TIP_IDS[0]-1][0] else 0)

        # Other four fingers (tip y < pip y  → finger is up)
        for i in range(1, 5):
            fingers.append(1 if lm[TIP_IDS[i]][1] < lm[PIP_IDS[i]][1] else 0)

        return fingers

    # ─────────────────────────────────────────────────────────────────────────
    def findDistance(self, p1, p2, img=None, draw=True, r=12, t=3):
        """
        Euclidean distance between two (x, y) points.
        Returns (length, (x1,y1,x2,y2,cx,cy), img).
        """
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
        length  = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (cx, cy), r, (0, 255, 255), cv2.FILLED)

        return length, (x1, y1, x2, y2, cx, cy), img
