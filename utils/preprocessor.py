import torch
import numpy as np


class Preprocessor:
    def __init__(self, args):
        self.args = args

    def preprocess(self, observations):
        # observation: dict
        result, positions = {}, {}
        for key, value in observations.items():
            image = value.transpose(2, 0, 1)
            position = value[0, 0, 7:]

            result[key] = torch.from_numpy(image)
            positions[key] = position

        return result, positions


class GraphBuilder:
    def __init__(self):
        pass

    def build_graph(self, positions):
        pass
