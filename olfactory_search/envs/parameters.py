import dataclasses

import numpy as np
import scipy


@dataclasses.dataclass
class ParametersIsotropic:
    grid_size: int  # length of each dimension
    h_max: int  # maximum number of hits where h in [0, 1, ..., h_max]
    T_max: int  # maximum episode length
    lambda_over_delta_x: float  # dispersion length-scale of particles in medium (lambda) / cell size (delta_x)
    R_times_delta_t: float  # source emission rate (R) * sniff time (delta_t)
    delta_x_over_a: float  # cell size (delta_x) / agent radius (a)

    def __post_init__(self):
        assert self.grid_size > 0
        assert self.h_max > 0
        assert self.T_max > 0
        assert self.lambda_over_delta_x > 0
        assert self.R_times_delta_t > 0
        assert self.delta_x_over_a > 0

        self.lambda_over_a = self.lambda_over_delta_x * self.delta_x_over_a
        self.mu0_Poisson = (
            1 / np.log(self.lambda_over_a) * scipy.special.k0(1)
        ) * self.R_times_delta_t
        self.h = np.arange(0, self.h_max + 1)


@dataclasses.dataclass
class ParametersWindy(ParametersIsotropic):
    V_times_delta_t: float  # wind speed (V) * sniff time (delta_t) (optional)
    tau_bar: float  # mean particle lifetime (tau_bar) (optional)

    def __post_init__(self):
        super().__post_init__()

        if self.V_times_delta_t is not None and self.tau_bar is not None:
            self.lambda_bar = np.sqrt(
                (self.tau_bar / self.V_times_delta_t**2) / (1 + self.tau_bar / 4)
            )


SMALLER_ISOTROPIC_DOMAIN = ParametersIsotropic(
    grid_size=19,
    h_max=2,
    T_max=642,
    lambda_over_delta_x=1.0,
    R_times_delta_t=1.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
)

LARGER_ISOTROPIC_DOMAIN = ParametersIsotropic(
    grid_size=53,
    h_max=3,
    T_max=2188,
    lambda_over_delta_x=3.0,
    R_times_delta_t=2.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
)

SMALLER_WINDY_DOMAIN = ParametersWindy(
    grid_size=19,
    h_max=2,
    T_max=642,
    lambda_over_delta_x=1.0,
    R_times_delta_t=2.5,  # with detections
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
    V_times_delta_t=2.0,
    tau_bar=150.0,
)

LARGER_WINDY_DOMAIN = ParametersWindy(
    grid_size=53,
    h_max=3,
    T_max=2188,
    lambda_over_delta_x=3.0,
    R_times_delta_t=2.5,  # with detections
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
    V_times_delta_t=2.0,
    tau_bar=150.0,
)
