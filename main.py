import cv2
import time
from hand_detector import HandDetector


def main():
    # ── Camera setup ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that it is connected.")
        return

    detector = HandDetector(maxHands=2, detectionCon=0.5, trackCon=0.5)

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    pTime = 0

    print("[INFO] Starting. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("[WARN] Empty frame – skipping.")
            continue

        # Optional: flip for natural mirror view
        img = cv2.flip(img, 1)

        # ── Detect ───────────────────────────────────────────────────────────
        hands, img = detector.findHands(img, draw=True)

        if hands:
            hand1 = hands[0]
            fingers1 = detector.fingersUp(hand1)
            total_up1 = sum(fingers1)

            # Show finger count for hand 1
            label1 = hand1["type"]
            cv2.putText(img,
                        f"{label1}: {total_up1} finger(s) up",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Which specific fingers are up
            up_names = [finger_names[i] for i, v in enumerate(fingers1) if v]
            cv2.putText(img,
                        ", ".join(up_names) if up_names else "Fist",
                        (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            if len(hands) == 2:
                hand2 = hands[1]
                fingers2 = detector.fingersUp(hand2)
                total_up2 = sum(fingers2)
                label2 = hand2["type"]

                cv2.putText(img,
                            f"{label2}: {total_up2} finger(s) up",
                            (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Distance between index finger tips of both hands
                p1 = hand1["lmList"][8][:2]   # index tip (x, y)
                p2 = hand2["lmList"][8][:2]
                length, info, img = detector.findDistance(p1, p2, img, draw=True)

                cv2.putText(img,
                            f"Index dist: {int(length)} px",
                            (10, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        # ── FPS ───────────────────────────────────────────────────────────────
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Hand Detector", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
