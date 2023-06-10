from dataclasses import dataclass
import json
from typing import Tuple
import copy


@dataclass
class Hparams:
    batch_size: int = 128
    epochs: int = 128

    lr: float = 1e-3
    beta1: float = 0.9
    grad_clipping: float = 1.0

    knn: int = 32
    features: int = 256

    layers: int = 3

    diffusion_steps: int = 100
    x_betas: Tuple[float, float] = (1e-6, 2e-3)

    def from_json(self, file_name):
        with open(file_name, "r") as fp:
            hparams = json.load(fp)

        for key, value in hparams.items():
            assert (key in self.__dict__) or (key in ("x_betas_min", "x_betas_max"))

            if key == "x_betas_min":
                self.__dict__["x_betas"] = (value, self.__dict__["x_betas"][1])
            elif key == "x_betas_max":
                self.__dict__["x_betas"] = (self.__dict__["x_betas"][0], value)
            else:
                self.__dict__[key] = value

    def to_json(self, file_name):
        with open(file_name, "w") as fp:
            json.dump(self.__dict__, fp, indent=4)

    def dict(self):
        result = copy.deepcopy(self.__dict__)
        result["x_betas_min"], result["x_betas_max"] = result["x_betas"]
        del result["x_betas"]
        return result
