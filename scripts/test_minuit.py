import os
import numpy as np
import pandas as pd
from flip import fitter, plot_utils, utils
from flip.covariance import covariance, contraction
from pkg_resources import resource_filename
import matplotlib.pyplot as plt

def main(mode = None):

    flip_base = resource_filename("flip", ".")
    data_path = os.path.join(flip_base, "data")

    ### Load data
    sn_data = pd.read_parquet(os.path.join(data_path, "velocity_data.parquet"))
    sn_data = sn_data[np.array(sn_data["status"]) != False]
    sn_data = sn_data[np.array(sn_data["status"]) != None]

    if mode == 'l':
        sn_data = sn_data[np.array(sn_data["rcom_zobs"]) < np.median(sn_data["rcom_zobs"])]
    elif mode == 'u':
        sn_data = sn_data[np.array(sn_data["rcom_zobs"]) >= np.median(sn_data["rcom_zobs"])]    

    coordinates_velocity = np.array([sn_data["ra"], sn_data["dec"], sn_data["rcom_zobs"]])

    data_velocity = sn_data.to_dict("list")
    for key in data_velocity.keys():
        data_velocity[key] = np.array(data_velocity[key])
    data_velocity["velocity"] = data_velocity.pop("vpec")
    data_velocity["velocity_error"] = np.zeros_like(data_velocity["velocity"])
    # plt.scatter(sn_data["rcom_zobs"],data_velocity["velocity"])
    # plt.show()
    # we

    ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
    kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
    kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))

    sigmau_fiducial = 15

    power_spectrum_dict = {"vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]]}

    ### Compute covariance
    size_batch = 10_000
    number_worker = 16
    # print(sn_data['zobs'].min(), sn_data['zobs'].max())

    # print(coordinates_velocity[2].min(), coordinates_velocity[2].max())
    # wef
    covariance_fit = covariance.CovMatrix.init_from_flip(
        "carreres23",
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
    )


    ### Load fitter
    likelihood_type = "multivariate_gaussian"
    likelihood_properties = {"inversion_method": "cholesky", "velocity_type": "scatter"}


    parameter_dict = {
        "fs8": {
            "value": 0.4,
            "limit_low": 0.0,
            "fixed": False,
        },
        "sigv": {
            "value": 200,
            "limit_low": 0.0,
            "fixed": True,
        },
        "sigma_M": {
            "value": 0.12,
            "limit_low": 0.0,
            "fixed": True,
        },
    }

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        covariance_fit,
        data_velocity,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    )

    ### Fit
    ans= minuit_fitter.run(migrad=False, hesse=True, n_iter=1)

if __name__ == '__main__':
    fit=main()
    fit_l=main(mode='l')
    fit_u=main(mode='u')

