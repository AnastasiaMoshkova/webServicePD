import pandas as pd
import numpy as np
import os
import json
import math
import glob
from functools import reduce
from scipy.signal import find_peaks
from hydra.utils import instantiate


class FE:
    def __init__(self, config_feature):
        self.config_fe = ""
        self.config_feature = config_feature
        self.feature_type = [
            "NumA",
            "AvgFrq",
            "VarFrq",
            "AvgVopen",
            "AvgVclose",
            "AvgA",
            "VarA",
            "VarVopen",
            "VarVclose",
            "DecA",
            "DecV",
            "DecLin",
        ]
        self.threshold_aplitude = {
            "FT": 5,
            "OC": 5,
            "PS": 10,
        }
        self.algorithm_filtering = "by_low_amplitude"
        self.FEATURE_NORMS = {
            "NumA": 40,
            "AvgFrq": 4,
            "VarFrq": 2,
            "AvgVopen": 2,
            "AvgVclose": 2,
            "AvgA": 50,
            "VarA": 15,
            "VarVopen": 30,
            "VarVclose": 30,
            "DecA": 2,
            "DecV": 1.5,
            "DecLin": 0.1,
        }

    def loadfileInterval_hand(self, datapoint, start, stop):
        counter = 0
        maxPointX = []
        minPointX = []
        maxPointY = []
        minPointY = []
        newlist = []
        newlistSortedAll = sorted(datapoint, key=lambda k: k["X"])
        for point in newlistSortedAll:
            if (point["X"] >= start) & (point["X"] <= stop):
                newlist.append(point)
        for point in newlist:
            counter = counter + 1
            if point["Type"] == 1:
                if (counter != 1) and (counter != len(newlist)):
                    maxPointX.append(point["X"])
                    maxPointY.append(point["Y"])
            if point["Type"] == 0:
                minPointX.append(point["X"])
                minPointY.append(point["Y"])
        return maxPointX, maxPointY, minPointX, minPointY

    def deleterAmplitude(self, maxPointX, maxPointY, minPointX, minPointY, threshhold):
        resultMax = []
        resultMin = []
        for i in range(len(maxPointX)):
            if (maxPointY[i] - minPointY[i + 1]) < threshhold:
                resultMax.append(i)
                resultMin.append(i + 1)

        if len(resultMax) != 0:
            resultMax.reverse()
            for k in resultMax:
                del maxPointX[k]
                del maxPointY[k]
        if len(resultMin) != 0:
            resultMin.reverse()
            for k in resultMin:
                del minPointX[k]
                del minPointY[k]
        return maxPointX, maxPointY, minPointX, minPointY

    # def _norm_coefficient(self, path_to_folder, exercise, mode):
    #     if self.config_fe["feature_extractor"][mode]["norm_coeff"]:
    #         norm_coeff_name = self.config_fe["feature_extractor"][mode]["norm_coeff_name"]
    #         path = os.path.normpath(path_to_folder)
    #         # path = os.path.join(*path.split(os.sep)[:-1])
    #         path = os.path.join("\\".join(path.split(os.sep)[:-1]))
    #         norm_coeff_file = json.load(open(os.path.join(path, "coefficients", "palm_width.json")))
    #         norm_coeff = norm_coeff_file[norm_coeff_name]
    #     else:
    #         norm_coeff = 1
    #     if exercise == "PS":
    #         norm_coeff = 1
    #     return norm_coeff

    def feature_calculation_hand(
        self,
        path,
        file,
        exercise,
    ):

        path_to_file = os.path.join(path, file)
        datapoint = json.load(open(path_to_file))
        start = 100
        stop = 1700
        maxPointX, maxPointY, minPointX, minPointY = self.loadfileInterval_hand(
            datapoint, start, stop
        )
        algorithm_filtering = self.algorithm_filtering
        if algorithm_filtering == "by_low_amplitude":
            maxPointX, maxPointY, minPointX, minPointY = self.deleterAmplitude(
                maxPointX,
                maxPointY,
                minPointX,
                minPointY,
                self.threshold_aplitude[exercise],
            )
        norm_coeff = 1
        res = {}
        for feature in self.feature_type:
            if len(maxPointX) > 1:
                feature_class = instantiate(
                    self.config_feature[feature],
                    maxPointX,
                    maxPointY,
                    minPointX,
                    minPointY,
                    norm_coeff,
                    datapoint,
                )
                res[feature] = feature_class.calc()
            else:
                res[feature] = -1
        return res

    def norm_feature(self, result):
        res_norm = {}
        for key, value in result.items():
            res_norm[key] = value / self.FEATURE_NORMS[key]
        return res_norm

    def processing(self, local_dir, exercise):
        res = self.feature_calculation_hand(local_dir, "min_max_points.json", exercise)
        res_norm = self.norm_feature(res)
        return res, res_norm
