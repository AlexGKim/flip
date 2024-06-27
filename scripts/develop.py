import os

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from flip import fisher, utils
from flip.covariance import covariance

def main():
    flip_base = resource_filename("flip", ".")
    data_path = os.path.join(flip_base, "data")

    ### Load data
    sn_data = pd.read_parquet(os.path.join(data_path, "velocity_data.parquet"))

    sn_data = sn_data[np.array(sn_data["status"]) != False]
    sn_data = sn_data[np.array(sn_data["status"]) != None]

    coordinates_velocity = np.array([sn_data["ra"], sn_data["dec"], sn_data["rcom_zobs"]])

    data_velocity = sn_data.to_dict("list")
    for key in data_velocity.keys():
        data_velocity[key] = np.array(data_velocity[key])
    data_velocity["velocity"] = data_velocity.pop("vpec")
    data_velocity["velocity_error"] = np.zeros_like(data_velocity["velocity"])


    ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
    kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
    kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))

    sigmau_fiducial = 15

    power_spectrum_dict = {"vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]]}

    ### Compute covariance
    size_batch = 10_000
    number_worker = 16

    covariance_fit = covariance.CovMatrix.init_from_flip(
        "agkim24",
        # 'carreres23',
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
    )


    ### Load fitter

    fisher_properties = {
        "inversion_method": "inverse",
        "velocity_type": "scatter",
    }

    variant = None  # can be replaced by growth_index

    parameter_dict = {
        "fs8": 0.4,
        "sigv": 200,
        "sigma_M": 0.12,
    }

    Fisher = fisher.FisherMatrix.init_from_covariance(
        covariance_fit,
        data_velocity,
        parameter_dict,
        fisher_properties=fisher_properties,
    )

    parameter_name_list, fisher_matrix = Fisher.compute_fisher_matrix(
        parameter_dict, variant=variant
    )
    return parameter_name_list, fisher_matrix


if __name__ == "__main__":
    parameter_name_list, fisher_matrix = main()
    print(fisher_matrix)