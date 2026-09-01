import numpy as np
import math
from itertools import combinations


class FeatureExtraction:
    def __init__(self):
        # Улыбка 1: Большая скуловая мышца (Zygomaticus Major)
        self.smile_1_l = [61, 186, 216, 207, 187, 123]
        self.smile_1_r = [308, 410, 436, 427, 411, 352]

        # Улыбка 2: Малая скуловая (Zygomaticus Minor)
        self.smile_2_l = [50, 203, 165, 39]
        self.smile_2_r = [280, 423, 391, 269]

        # Улыбка 3: Мышца, поднимающая верхнюю губу (Levator Labii Superioris)
        self.smile_3_l = [50, 205, 206, 92, 40]
        self.smile_3_r = [280, 425, 426, 322, 270]

        # Брови (внутренняя часть - Corrugator)
        self.left_in_brow = [55, 65, 52, 53, 46]
        self.right_in_brow = [285, 295, 282, 283, 276]

        # Брови (верхняя часть - Frontalis)
        self.left_up_brow = [107, 66, 105, 63, 70]
        self.right_up_brow = [336, 296, 334, 293, 300]

        # Глаза (Orbicularis Oculi)
        self.blink_left = [159, 145, 133, 33]
        self.blink_right = [386, 374, 362, 263]

        # Нейтраль
        self.neutral_points = [0, 17, 61, 291]

        # Опорные точки носа — якорь для бровных моделей
        # Порядок ВАЖЕН: должен совпадать с порядком при обучении
        self.nose_refs_brow  = [4, 10, 168]
        self.nose_refs_frown = [4, 6, 8, 10, 109, 151, 168, 338]

    def get_3d_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def get_eye_distance(self, landmarks):
        if len(landmarks) > 473:
            p_left  = landmarks[468]
            p_right = landmarks[473]
            dist = self.get_3d_distance(p_left, p_right)
            return dist if dist > 0 else 1.0
        return 1.0

    def calculate_pairs_in_group(self, landmarks, point_indices, eye_scale, prefix=""):
        """Расстояние для каждой пары точек в группе. Ключ: 'i-j' в порядке списка."""
        pairs_data = {}
        for i, j in combinations(point_indices, 2):
            if i < len(landmarks) and j < len(landmarks):
                dist = self.get_3d_distance(landmarks[i], landmarks[j]) / eye_scale
                pairs_data[f"{i}-{j}"] = dist
        return pairs_data

    def extract_features(self, landmarks, exercise_type):
        if not landmarks:
            return {}

        eye_scale    = self.get_eye_distance(landmarks)
        result_pairs = {}

        if exercise_type == "smile":
            for group in [self.smile_1_l, self.smile_1_r,
                          self.smile_2_l, self.smile_2_r,
                          self.smile_3_l, self.smile_3_r]:
                result_pairs.update(self.calculate_pairs_in_group(landmarks, group, eye_scale))

        elif exercise_type == "brows_up":
            # Модель обучена на combinations(left_in + left_up + nose_refs_brow, 2)
            # и  combinations(right_in + right_up + nose_refs_brow, 2)
            # Порядок конкатенации определяет имена признаков — не менять!
            left_group  = self.left_in_brow  + self.left_up_brow  + self.nose_refs_brow
            right_group = self.right_in_brow + self.right_up_brow + self.nose_refs_brow
            result_pairs.update(self.calculate_pairs_in_group(landmarks, left_group,  eye_scale))
            result_pairs.update(self.calculate_pairs_in_group(landmarks, right_group, eye_scale))

        elif exercise_type == "frown":
            # Модель обучена на combinations(left_in + right_in + nose_refs_frown, 2)
            frown_group = self.left_in_brow + self.right_in_brow + self.nose_refs_frown
            result_pairs.update(self.calculate_pairs_in_group(landmarks, frown_group, eye_scale))

        elif exercise_type == "blink":
            for group in [self.blink_left, self.blink_right]:
                result_pairs.update(self.calculate_pairs_in_group(landmarks, group, eye_scale))

        elif exercise_type in ("neutral", "neutral_all"):
            # Все группы для улыбки/бровей
            for group in [self.smile_1_l, self.smile_1_r,
                          self.smile_2_l, self.smile_2_r,
                          self.smile_3_l, self.smile_3_r,
                          self.left_in_brow,  self.right_in_brow,
                          self.left_up_brow,  self.right_up_brow]:
                result_pairs.update(self.calculate_pairs_in_group(landmarks, group, eye_scale))
            # Те же cross-группы с носовыми якорями, что используются в brows_up и frown
            left_brow_group  = self.left_in_brow  + self.left_up_brow  + self.nose_refs_brow
            right_brow_group = self.right_in_brow + self.right_up_brow + self.nose_refs_brow
            frown_group      = self.left_in_brow + self.right_in_brow + self.nose_refs_frown
            result_pairs.update(self.calculate_pairs_in_group(landmarks, left_brow_group,  eye_scale))
            result_pairs.update(self.calculate_pairs_in_group(landmarks, right_brow_group, eye_scale))
            result_pairs.update(self.calculate_pairs_in_group(landmarks, frown_group,      eye_scale))

        else:
            result_pairs.update(self.calculate_pairs_in_group(landmarks, self.neutral_points, eye_scale))

        return result_pairs
