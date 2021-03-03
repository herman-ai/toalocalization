import numpy as np
from math import pi
from scipy.optimize import minimize
NUM_SENSORS = 5

scale_lst_sq_obj_fn = 10**23
scale_neg_log_likelihood = 10**-3

H = 200
F = 100


T_S = 10.0 ** (-3)
PS_0 = 10  # 19 dBm
MU_0 = 4 * pi * 10 ** (-7)
MU_A = MU_0
MU_S = 1.0084 * MU_0
LAMBDA = 0.7
N_0 = -110.0  #dBm
FREQUENCY = 833 * 10 ** 6
BETA_SQ = FREQUENCY ** 2
z_scale = 100.

EPSILON_0 = 8.85 * 10 ** (-12)
EPSILON_SOIL = np.asarray((2.361970519728 + 0.096670930496j,
                            5.08697889259241 + 0.4413468113884j,
                            16.4109595611802 + 2.36876126519685j,
                            24.4855102012741 + 3.85704851601056j)) * EPSILON_0

epsilon_index = 0
EPSILON_S_REAL = np.real(EPSILON_SOIL[epsilon_index])
EPSILON_S_IMG = np.imag([epsilon_index])
EPSILON_AIR = EPSILON_0

speed_soil = ( \
    np.sqrt( \
    (MU_S * EPSILON_S_REAL / 2 ) * (np.sqrt(1 + (EPSILON_S_IMG/EPSILON_S_REAL) ** 2) + 1)\
    ) ) ** (-1)

OMEGA = 2 * pi * FREQUENCY

ALPHA_SOIL = OMEGA * \
             np.sqrt((MU_S * EPSILON_S_REAL/2) *
                  (np.sqrt(1 + (EPSILON_S_IMG/EPSILON_S_REAL) ** 2) - 1))

# Reflection coefficient
RHO = ((np.sqrt(MU_A / EPSILON_AIR) - np.sqrt(MU_S / EPSILON_S_REAL)) / \
    (np.sqrt(MU_A / EPSILON_AIR) + np.sqrt(MU_S / EPSILON_S_REAL))) ** 2


def p_average_soil2soil(d, ALPHA_SOIL):
    pathLoss = 20 * np.log10(4 * pi * d / LAMBDA)
    return PS_0 - d * ALPHA_SOIL -pathLoss #in dBm

def sigma_soil2Soil(d, ALPHA_SOIL):
  specularPowerdB = p_average_soil2soil(d, ALPHA_SOIL)
  specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
  TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
  sigma = np.sqrt(8 * (pi ** 2) * specular_power * T_S * BETA_SQ / TOTAL_NOISE) ** -1
  return sigma


def toa_squared_error_soil2soil(xyz, toa_observed_from_other_sensors):
    xyz = xyz.reshape(-1,3)*(1,1,1/z_scale)
    sq_error = 0.
    for i in range(xyz.shape[-2]):
        for j in range(xyz.shape[-2]):
            if not toa_observed_from_other_sensors[i,j] > 0:
                continue
            dist = np.sqrt(np.sum((xyz[i] - xyz[j]) ** 2))
            mean_toa =dist/speed_soil
            sq_error += ((toa_observed_from_other_sensors[i,j]-mean_toa)**2)*scale_lst_sq_obj_fn
            # sigma_toa = sigma_soil2Soil(dist, ALPHA_SOIL)
    return sq_error

def p_average_soil2soil_reflected(d, ALPHA_SOIL):
    pathLoss = 20 * np.log10(4 * pi * d / LAMBDA)
    return PS_0 - d * ALPHA_SOIL - pathLoss - RHO #in dBm

def sigma_soil2Soil_reflected(d, ALPHA_SOIL):
  specularPowerdB = p_average_soil2soil_reflected(d, ALPHA_SOIL)
  specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
  TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
  sigma = np.sqrt(8 * (pi ** 2) * specular_power * T_S * BETA_SQ / TOTAL_NOISE) ** -1
  return sigma

def sigma_soil2soil_cross(d_dir,d_ref, ALPHA_SOIL):
    specularPowerDirdB = p_average_soil2soil(d_dir, ALPHA_SOIL)
    specular_power_dir = 10 ** ((specularPowerDirdB - 30) / 10.0)

    specularPowerRefdB = p_average_soil2soil_reflected(d_ref, ALPHA_SOIL)
    specular_power_ref = 10 ** ((specularPowerRefdB - 30) / 10.0)
    TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)

    sigma = (8 * (pi ** 2) * np.sqrt(specular_power_dir*specular_power_ref) * T_S * BETA_SQ / TOTAL_NOISE) ** -1
    # sigma=np.array([0])
    return sigma


def toa_neg_log_likelihood_soil2soil(xyz, toa_observed_from_other_sensors_dir,
                                     toa_observed_from_other_sensors_reflected):
    xyz = xyz.reshape(-1, 3) * (1, 1, 1 / z_scale)
    neg_log_likelihood = 0.
    #for direct path
    for i in range(xyz.shape[-2]):
        for j in range(xyz.shape[-2]):
            if i == j:
                continue
            if toa_observed_from_other_sensors_dir[i, j] > 0 and toa_observed_from_other_sensors_reflected[i,j] > 0:
                dist_dir = np.sqrt(np.sum((xyz[i] - xyz[j]) ** 2))
                mean_toa_dir = dist_dir / speed_soil
                sigma2_toa_dir = sigma_soil2Soil(dist_dir, ALPHA_SOIL)**2
                # neg_log_likelihood += ((toa_observed_from_other_sensors_dir[i, j] - mean_toa) ** 2) /sigma2_toa
                # neg_log_likelihood += 0.5*np.log(sigma2_toa)

                if xyz[i,2] == 0 or xyz[j,2] == 0:
                    continue
                #
                import warnings
                warnings.filterwarnings('error')
                try:

                    d1 = np.sqrt((xyz[i, 2]) ** 2 + \
                                 ((xyz[i, 0] - xyz[j, 0]) ** 2 + \
                                  (xyz[i, 1] - xyz[j, 1]) ** 2) *
                                 ((xyz[i, 2] / (xyz[i, 2] + xyz[j, 2])) ** 2))

                    d2 = np.sqrt((xyz[j, 2]) ** 2 + \
                                 ((xyz[i, 0] - xyz[j, 0]) ** 2 + \
                                  (xyz[i, 1] - xyz[j, 1]) ** 2) *
                                 ((xyz[j, 2] / (xyz[i, 2] + xyz[j, 2])) ** 2))
                except Warning:
                    print("xyz[i,2] = {}, xyz[j,2]={}".format(xyz[i, 2], xyz[j, 2]))
                    print(xyz[i, 2] == 0 or xyz[j, 2] == 0)
                    print("warning")

                dist_ref = d1 + d2
                mean_toa_ref = dist_ref / speed_soil
                sigma2_toa_ref = sigma_soil2Soil_reflected(dist_ref, ALPHA_SOIL) ** 2
                sigma2_cross = sigma_soil2soil_cross(dist_dir, dist_ref, ALPHA_SOIL)
                # neg_log_likelihood += ((toa_observed_from_other_sensors_reflected[i, j] - mean_toa_ref) ** 2) / sigma2_toa_ref
                # neg_log_likelihood += 0.5 * np.log(sigma2_toa_ref)

                t = np.asarray([toa_observed_from_other_sensors_dir[i, j], toa_observed_from_other_sensors_reflected[i, j]]).reshape(2,1)
                t_bar = np.asarray([mean_toa_dir, mean_toa_ref])
                sigma = np.asarray([[1/sigma2_toa_dir[0], sigma2_cross], [sigma2_cross,1/sigma2_toa_ref[0]]])

                a = sigma.dot(t-t_bar)
                b = (t-t_bar).T
                # neg_log_likelihood += ((t-t_bar).dot(sigma)).dot((t-t_bar).T)
                neg_log_likelihood += b.dot(a)
                neg_log_likelihood += 0.5*np.log(np.linalg.det(sigma))
                # neg_log_likelihood += 0.5 * np.log(sigma2_toa_dir)
                # neg_log_likelihood += 0.5 * np.log(sigma2_toa_ref)

            else:
                raise Exception("test")

            # if toa_observed_from_other_sensors_reflected[i,j] > 0:
                # print("i={},j={}".format(i,j))


    return neg_log_likelihood*scale_neg_log_likelihood


#direct path
if __name__=="__main__":

    actual_s_locations = np.loadtxt("data/act_sensor_loc.csv")
    est_s_locations_stage1 = np.loadtxt("data/est_stage1.csv")

    print("estimated locations (stage1) = \n{}".format(est_s_locations_stage1))
    print("actual locations = \n{}".format(actual_s_locations))

    toa_observed_from_other_sensors_dir = np.zeros(shape=(NUM_SENSORS, NUM_SENSORS))
    toa_observed_from_other_sensors_reflected = np.zeros(shape=(NUM_SENSORS, NUM_SENSORS))

    for i in range(actual_s_locations.shape[-2]):
        for j in range(actual_s_locations.shape[-2]):
            if i == j:
                continue
            dist_dir = np.sqrt(np.sum((actual_s_locations[i]-actual_s_locations[j])**2))

            power_dir = p_average_soil2soil(dist_dir, ALPHA_SOIL)

            d1 = np.sqrt((actual_s_locations[i, 2]) ** 2 + \
                         ((actual_s_locations[i, 0] - actual_s_locations[j, 0]) ** 2 + \
                          (actual_s_locations[i, 1] - actual_s_locations[j, 1]) ** 2) *
                         ((actual_s_locations[i, 2] / (actual_s_locations[i, 2] + actual_s_locations[j, 2])) ** 2))

            d2 = np.sqrt((actual_s_locations[j, 2]) ** 2 + \
                         ((actual_s_locations[i, 0] - actual_s_locations[j, 0]) ** 2 + \
                          (actual_s_locations[i, 1] - actual_s_locations[j, 1]) ** 2) *
                         ((actual_s_locations[j, 2] / (actual_s_locations[i, 2] + actual_s_locations[j, 2])) ** 2))

            dist_ref = d1 + d2

            power_ref = p_average_soil2soil_reflected(dist_ref, ALPHA_SOIL)

            # print("power = {} dB".format(power))

            if power_dir > N_0 and power_ref > N_0:
                mean_toa_dir = dist_dir / speed_soil
                sigma_toa_dir = sigma_soil2Soil(dist_dir, ALPHA_SOIL=ALPHA_SOIL)

                mean_toa_ref = dist_ref / speed_soil
                sigma_toa_ref = sigma_soil2Soil_reflected(dist_ref, ALPHA_SOIL=ALPHA_SOIL)

                sigma2_cross = sigma_soil2soil_cross(dist_dir, dist_ref, ALPHA_SOIL)

                mean = (mean_toa_dir[0], mean_toa_ref[0])
                cov = [[sigma_toa_dir[0], sigma2_cross[0]],[sigma2_cross[0], sigma_toa_ref[0]]]
                cov = [[sigma_toa_dir[0], 0], [0, sigma_toa_ref[0]]]

                ob = np.random.multivariate_normal(mean, cov, 1)

                ob_dir = ob[0][0]
                ob_ref = ob[0][1]

                ob_dir = np.random.normal(loc=mean_toa_dir, scale=sigma_toa_dir, size=1)[0]
                ob_ref = np.random.normal(loc=mean_toa_ref, scale=sigma_toa_ref, size=1)[0]

                toa_observed_from_other_sensors_reflected[i, j] = ob_dir
                toa_observed_from_other_sensors_dir[i,j] = ob_ref
            else:
                print("****************power alert***********")





    print("observed toa from other sensors (dir)=\n{}".format(toa_observed_from_other_sensors_dir))
    print("observed toa from other sensors (reflected) =\n{}".format(toa_observed_from_other_sensors_reflected))

    # xyz0 = np.random.random_sample(size=(NUM_SENSORS, 3)) * (F, F, -H / H)  # X, Y, Z coordinates
    xyz0 = est_s_locations_stage1


    sq_error_at_actual = toa_neg_log_likelihood_soil2soil(actual_s_locations*(1,1,z_scale),
                                                          toa_observed_from_other_sensors_dir,
                                                          toa_observed_from_other_sensors_reflected)
    print("squared error at actual = {}".format(sq_error_at_actual))


    sq_error_at_seed = toa_neg_log_likelihood_soil2soil(xyz0*(1,1,z_scale),
                                                           toa_observed_from_other_sensors_dir,
                                                        toa_observed_from_other_sensors_reflected)
    print("squared error at seed = {}".format(sq_error_at_seed))

    bnds = (((None, None),)*2+((-100,-1),))*NUM_SENSORS
    lower = est_s_locations_stage1 - np.asarray([0.01, 0.01, 0.0001])
    upper = est_s_locations_stage1 + np.asarray([0.01, 0.01, 0.0001])
    print(bnds)
    bnds = list(((bb[0],cc[0]), (bb[1],cc[1]), (bb[2]*z_scale, cc[2]*z_scale)) for bb,cc in zip(lower,upper))
    bnds = np.asarray(bnds).reshape(-1, 2)
    print(bnds)

    result = minimize(toa_neg_log_likelihood_soil2soil,
                      xyz0*(1,1,z_scale),
                      args=(toa_observed_from_other_sensors_dir,
                            toa_observed_from_other_sensors_reflected),
                      # method="L-BFGS-B",
                      method="TNC",
                      # method='SLSQP',
                      bounds=bnds,
                      options={"maxiter":int(1e6)})
    print(result.success)
    est_s_locations_stage2 = result.x.reshape(-1, 3) * (1, 1, 1 / z_scale)
    print("actual =\n{}".format(actual_s_locations))
    print("stage 1=\n{}".format(est_s_locations_stage1))
    print("stage 2 =\n{}".format(est_s_locations_stage2))
    np.savetxt("data/est_stage2.csv", est_s_locations_stage2)
    sq_error_at_est = toa_neg_log_likelihood_soil2soil(est_s_locations_stage2*(1,1,z_scale),
                                                          toa_observed_from_other_sensors_dir,
                                                          toa_observed_from_other_sensors_reflected)
    print("squared error at est_s_locations_stage2 = {}".format(sq_error_at_est))


