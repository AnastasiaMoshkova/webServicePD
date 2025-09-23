import os
import pandas as pd
import json
import math
import numpy as np
import json
import os
import shutil

# from pprint import pprint
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
from data_base.hand2D import HandDataAngle
from hand_processing.adaptive import Adaptive
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMarking:
    def __init__(self):
        self.config = 1

    # подготовка датафрема с путями до папок, где будет применен алгоритм авто разметки
    def data_parser_dataset(self, dataset):
        path_to_dir = self.config["automarking"][dataset]["path_to_directory"]
        folders = os.listdir(self.config["automarking"][dataset]["path_to_directory"])
        id = []
        dataset_folders = []
        for folder in folders:
            folders_r = os.listdir(os.path.join(path_to_dir, folder))
            for r in folders_r:
                id.append(folder)
                dataset_folders.append(os.path.join(path_to_dir, folder, r))
        df = pd.DataFrame(dataset_folders, columns=["path"])
        df["folders"] = id
        folder_name = self.config["automarking"][dataset]["folder_name"]
        df["number"] = df["folders"].str.split(folder_name).str[1].apply(int)
        df["dataset"] = dataset
        return df.loc[df["number"].isin(self.config["automarking"][dataset]["number"])]

    def data_processing(self):
        df_result = []
        for dataset in self.config["automarking"]["dataset_type"]:
            df_result.append(self.data_parser_dataset(dataset))
            """
            if ((dataset == 'PD') | (dataset == 'Students')):
                #df_result.append(self.data_parser(dataset))
                df_result.append(self.data_parser_dataset(dataset))
            if (dataset == 'Healthy'):
                df_result.append(self.data_parser_dataset(dataset))
            """
        df = pd.concat(df_result, axis=0).reset_index(drop=True)
        return df

    # запись точек в файл
    def write_point_hand(
        self,
        path,
        maxP,
        minP,
        maxA,
        minA,
    ):
        datapoint = []
        for i in range(len(maxP)):
            datapoint.append(
                {"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": float(maxP[i]), "Y": maxA[i]}
            )
        for i in range(len(minP)):
            datapoint.append(
                {"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": float(minP[i]), "Y": minA[i]}
            )
        datapoint = sorted(datapoint, key=lambda k: k["X"])
        file_point = "min_max_points.json"
        if len(datapoint) != 0:
            if not os.path.isdir(path):
                os.mkdir(path)
            with open(os.path.join(path, file_point), "w") as f:
                json.dump(datapoint, f)

    def write_point_hand_timestamps(
        self, path, file, maxP, minP, maxA, minA, frac, order_min, order_max
    ):
        datapoint = []
        for i in range(len(maxP)):
            datapoint.append(
                {"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": float(maxP[i]), "Y": maxA[i]}
            )
        for i in range(len(minP)):
            datapoint.append(
                {"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": float(minP[i]), "Y": minA[i]}
            )
        datapoint = sorted(datapoint, key=lambda k: k["X"])
        file_point = (
            file.split(".json")[0]
            + "_".join(["_point", str(frac), str(order_min), str(order_max)])
            + ".json"
        )
        if len(datapoint) != 0:
            if not os.path.isdir(path):
                os.mkdir(path)
            with open(os.path.join(path, file_point), "w") as f:
                json.dump(datapoint, f)

    # получение точек авто разметки по сигналам двигательной активности рук
    def auto_point_hand(self, values, frame, fps):
        auto_alg_class = Adaptive()
        maxP, minP, maxA, minA, frac, order_min, order_max = auto_alg_class.get_point(
            values, frame, fps
        )
        return np.array(frame)[maxP], np.array(frame)[minP], maxA, minA, frac, order_min, order_max
        # return maxP, minP, maxA, minA, frac, order_min, order_max

    def hand_processing_auto_point(self, path_to_dir, hand, exercise, fps):
        signal_class = HandDataAngle()  # FIXME config
        exercise_dict = {"FT": "1", "OC": "2", "PS": "3"}
        folder_to_save = os.path.join(path_to_dir, "auto_algoritm_MP")  # FIXME config
        input_file = os.path.join(path_to_dir, "handMP", "test_file.json")
        logger.info(f"Размечаемый файл { input_file}")

        values, frames, palm_width = signal_class.signal_hand(
            input_file, exercise_dict[exercise], hand
        )  # FIXME
        if len(values) > 50:  # FIXME config
            maxP, minP, maxA, minA, frac, order_min, order_max = self.auto_point_hand(
                values, frames, fps
            )
            # maxP, minP, maxA, minA = self.signalPoint(maxP, minP, maxA, minA)
            if True:  # FIXME save peacture config
                output_dir = os.path.join(path_to_dir, "images")
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                path_to_save_image = os.path.join(
                    output_dir,
                    "signal_picture.png",
                )
                signal_class.plot_image(
                    values,
                    frames,
                    maxP,
                    minP,
                    maxA,
                    minA,
                    path_to_save_image,
                    "",
                )
            self.write_point_hand(
                folder_to_save,
                maxP,
                minP,
                maxA,
                minA,
            )
            return (
                maxP,
                minP,
                maxA,
                minA,
                values,
                frames,
            )

    def processing(self, path_to_dir, exercise, fps):
        hand = "L"
        return self.hand_processing_auto_point(path_to_dir, hand, exercise, fps)
