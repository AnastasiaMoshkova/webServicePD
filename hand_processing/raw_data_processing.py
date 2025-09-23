import os
import re
import re
import numpy as np
import json
import cv2
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreProcessing:
    def __init__(self):

        self.config = 1

    def MPjson(self, path_to_dir):  # функция для формирования json по анологии с leapmotion
        translate_hand = {
            "Left": "right hand",
            "Right": "left hand",
        }  # из-за того что видосы на фронталку, приходится руку вот так инвертировать
        POINTS = [
            "CENTRE",
            "THUMB_MCP",
            "THUMB_PIP",
            "THUMB_DIP",
            "THUMB_TIP",
            "FORE_MCP",
            "FORE_PIP",
            "FORE_DIP",
            "FORE_TIP",
            "MIDDLE_MCP",
            "MIDDLE_PIP",
            "MIDDLE_DIP",
            "MIDDLE_TIP",
            "RING_MCP",
            "RING_PIP",
            "RING_DIP",
            "RING_TIP",
            "LITTLE_MCP",
            "LITTLE_PIP",
            "LITTLE_DIP",
            "LITTLE_TIP",
        ]
        path_to_file = os.path.join(path_to_dir, "test_video.mp4")
        path_to_out = os.path.join(path_to_dir, "handMP")  # FIXME config
        if not os.path.isdir(path_to_out):
            os.mkdir(path_to_out)
        if path_to_file.split(".")[1] in ["mp4", "mkv", "MOV", "mov"]:
            res = []
            timestamp = 0
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
            )
            logger.info("\n")
            cap = cv2.VideoCapture(path_to_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"\n\nProcessing file: {path_to_file}\n")
            count_frame = 1
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                dict_points = {}
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        hand_label = handedness.classification[0].label
                        for id, landmark in enumerate(hand_landmarks.landmark):

                            if id == 0:
                                cords = {
                                    "X": round(landmark.x, 3),
                                    "Y": round(landmark.y, 3),
                                    "Z": round(landmark.z, 3),
                                    "X1": round(landmark.x, 3),
                                    "Y1": round(landmark.y, 3),
                                    "Z1": round(landmark.z, 3),
                                    "W": 0,
                                    "Wx": 0,
                                    "Wy": 0,
                                    "Wz": 0,
                                    "Angle": 0,
                                }
                            else:
                                cords = {
                                    "X1": round(landmark.x, 3),
                                    "Y1": round(landmark.y, 3),
                                    "Z1": round(landmark.z, 3),
                                    "X": round(landmark.x, 3),
                                    "Y": round(landmark.y, 3),
                                    "Z": round(landmark.z, 3),
                                    "W": 0,
                                    "Angle": 0,
                                }
                            dict_points.update({POINTS[id]: cords})

                            dict_points.update(
                                {
                                    "info": {
                                        "confidence": np.NaN,
                                        "id_frame": np.NaN,
                                        "visible_time": np.NaN,
                                        "pinch_distance": np.NaN,
                                        "pinch_strength": np.NaN,
                                        "grab_angle": np.NaN,
                                        "grab_strength": np.NaN,
                                        "palm_width": np.NaN,
                                        "timestamp": timestamp * 1000000,
                                        "frame_id": np.NaN,
                                        "tracking_frame_id": np.NaN,
                                        "framerate": fps,
                                        "version": np.NaN,
                                    }
                                }
                            )

                    res.append({translate_hand[hand_label]: dict_points, "frame": count_frame})
                    # print("frame:", count_frame, end="\r", flush=True)
                    count_frame += 1
                    timestamp += 1 / fps
            # logger.info("frame:", count_frame, end="\n")
            cap.release()
            with open(
                os.path.join(path_to_out, "test_file.json"),
                "w",
            ) as outfile:
                json.dump(res, outfile)
            return fps

    def processing(self, path_to_dir):
        return self.MPjson(path_to_dir)
