import numpy as np
from math import pi
from scipy.optimize import minimize
from math import log10

speed = 3. * 10**8
NUM_RUNS = 1000
MU_0 = 4 * pi * 10 ** (-7)
EPSILON_0 = 8.85 * 10 ** (-12)

LAMBDA = 0.7
PA_0 = 38 # 38 dBm
PS_0 = 10  # 19 dBm
N_0 = -110.0  #dBm
T_S = 10.0 ** (-3)
z_scale = 100.

# Scaling constants for optimization stability
scale_lst_sq_obj_fn = 10**14
scale_neg_log_likelihood_soil2soil = 10 ** -3

class ToaLocalizer:

    EPSILON_SOIL = np.asarray((2.361970519728 + 0.096670930496j,
                                5.08697889259241 + 0.4413468113884j,
                                16.4109595611802 + 2.36876126519685j,
                                24.4855102012741 + 3.85704851601056j)) * EPSILON_0
    
    def __init__(self, epsilon_index, f):
        self.e_s_real = np.real(ToaLocalizer.EPSILON_SOIL[epsilon_index])
        self.e_s_img = np.imag([epsilon_index])
        e_a = EPSILON_0
        mu_a = MU_0
        self.mu_s = 1.0084 * MU_0
        self.omega = 2 * pi * f
        self.anchor_locations = np.asarray([[0., 0., H],
                             [F, 0., H],
                             [0., F, H],
                             [F, F, H]
                             ])
        self.beta_sq = f ** 2
        self.alpha_s = self.get_alpha_soil()

        self.rho = ((np.sqrt(mu_a / e_a) - np.sqrt(self.mu_s / self.e_s_real)) / \
            (np.sqrt(mu_a / e_a) + np.sqrt(self.mu_s / self.e_s_real))) ** 2   # Reflection coefficient

        self.tau = 1 - self.rho   # Transmission coefficient

        self.speed_soil = self.get_speed_soil()

    def get_speed_soil(self, e_s_img=None, e_s_real=None, mu_s=None):
        if e_s_img is None:
            e_s_img = self.e_s_img
        if e_s_real is None:
            e_s_real = self.e_s_real
        if mu_s is None:
            mu_s = self.mu_s
        return (np.sqrt( \
                      (mu_s * e_s_real / 2) * (np.sqrt(1 + (e_s_img / e_s_real) ** 2) + 1) \
                      )) ** (-1)

    def get_alpha_soil(self, e_s_img=None, e_s_real=None, mu_s=None):
        if e_s_img is None:
            e_s_img = self.e_s_img
        if e_s_real is None:
            e_s_real = self.e_s_real
        if mu_s is None:
            mu_s = self.mu_s

        return self.omega * \
        np.sqrt((mu_s * e_s_real / 2) *
                (np.sqrt(1 + (e_s_img / e_s_real) ** 2) - 1))

    def p_average_air2soil(self, d_a, d_s, ALPHA_SOIL=None):
      pathLossAir = 20 * np.log10(4 * pi * d_a * self.tau / LAMBDA)
      if ALPHA_SOIL is None:
        ALPHA_SOIL = self.alpha_s
      pathLossSoil = ALPHA_SOIL * d_s
      return PA_0 - pathLossAir - pathLossSoil

    def sigma_air2Soil(self, d_a, d_s, ALPHA_SOIL=None):
      if ALPHA_SOIL is None:
        ALPHA_SOIL = self.alpha_s
      specularPowerdB = self.p_average_air2soil(d_a, d_s, ALPHA_SOIL=ALPHA_SOIL)
      specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
      TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
      sigma = np.sqrt(8 * (pi ** 2) * specular_power * T_S * self.beta_sq / TOTAL_NOISE) ** -1
      return sigma

    def p_average_soil2soil(self, d, ALPHA_SOIL=None):
        if ALPHA_SOIL is None:
            ALPHA_SOIL = self.alpha_s
        pathLoss = 20 * np.log10(4 * pi * d / LAMBDA)
        return PS_0 - d * ALPHA_SOIL - pathLoss  # in dBm

    def sigma_soil2Soil(self, d, ALPHA_SOIL=None):
        if ALPHA_SOIL is None:
            ALPHA_SOIL = self.alpha_s
        specularPowerdB = self.p_average_soil2soil(d, ALPHA_SOIL)
        specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
        TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
        sigma = np.sqrt(8 * (pi ** 2) * specular_power * T_S * self.beta_sq / TOTAL_NOISE) ** -1
        return sigma

    def p_average_soil2soil_reflected(self, d, ALPHA_SOIL=None, RHO=None):
        if ALPHA_SOIL is None:
            ALPHA_SOIL = self.alpha_s
        if RHO is None:
            RHO = self.rho
        pathLoss = 20 * np.log10(4 * pi * d / LAMBDA)
        return PS_0 - d * ALPHA_SOIL - pathLoss - RHO  # in dBm

    def sigma_soil2Soil_reflected(self, d, ALPHA_SOIL=None):
        specularPowerdB = self.p_average_soil2soil_reflected(d, ALPHA_SOIL)
        specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
        TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
        sigma = np.sqrt(8 * (pi ** 2) * specular_power * T_S * self.beta_sq / TOTAL_NOISE) ** -1
        return sigma

    def sigma_soil2soil_cross(self, d_dir, d_ref, ALPHA_SOIL=None):
        if ALPHA_SOIL is None:
            ALPHA_SOIL = self.alpha_s
        specularPowerDirdB = self.p_average_soil2soil(d_dir, ALPHA_SOIL)
        specular_power_dir = 10 ** ((specularPowerDirdB - 30) / 10.0)

        specularPowerRefdB = self.p_average_soil2soil_reflected(d_ref, ALPHA_SOIL)
        specular_power_ref = 10 ** ((specularPowerRefdB - 30) / 10.0)
        TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)

        sigma = (8 * (pi ** 2) * np.sqrt(specular_power_dir * specular_power_ref) * T_S * self.beta_sq / TOTAL_NOISE) ** -1
        # sigma=np.array([0])
        return sigma

    # Least squares objective function
    def toa_squared_error_air2soil(self, xyz, toa_observed_from_anchors, speed_soil):

        xyz=xyz.reshape(-1,3)*(1, 1, 1 / z_scale)
        sq_error = 0.
        for i in range(xyz.shape[-2]):
            distance_air = np.sqrt(np.sum((xyz[i,:2]-self.anchor_locations[:,:2])**2, axis=1)+self.anchor_locations[:,2]**2)
            distance_soil = abs(xyz[i,2])
            mean_toa = distance_air / speed + distance_soil / speed_soil
            sq_error += np.sum((toa_observed_from_anchors[i,:]-mean_toa)**2)*scale_lst_sq_obj_fn
        return sq_error

    def toa_neg_log_likelihood_air2soil(self,
                                        xyz,
                                        toa_observed_from_anchors,
                                        e_s_img=None,
                                        e_s_real=None,
                                        mu_s=None):
        xyz = xyz.reshape(-1, 3) * (1, 1, 1 / z_scale)
        neg_log_likelihood = 0
        for i in range(xyz.shape[-2]):
            distance_air = np.sqrt(
                np.sum((xyz[i, :2] - self.anchor_locations[:, :2]) ** 2, axis=1) + self.anchor_locations[:, 2] ** 2)
            distance_soil = abs(xyz[i, 2])
            speed_soil = self.get_speed_soil(e_s_img, e_s_real, mu_s)
            alpha_soil = self.get_alpha_soil(e_s_img, e_s_real, mu_s)

            mean_toa = distance_air / speed + distance_soil / speed_soil
            sigma2_toa = self.sigma_air2Soil(distance_air, distance_soil, alpha_soil)**2

            neg_log_likelihood += np.sum(((toa_observed_from_anchors[i,:]-mean_toa)**2)/sigma2_toa)
            neg_log_likelihood += np.sum(np.log(sigma2_toa))
        return neg_log_likelihood


    def toa_neg_log_likelihood(self, xyz,
                               toa_observed_from_anchors,
                               e_s_img=None,
                               e_s_real=None,
                               mu_s=None):
        # TODO add soil to soil here
        return self.toa_neg_log_likelihood_air2soil(xyz, toa_observed_from_anchors, e_s_img, e_s_real, mu_s)



    ##### soil to soil
    # def p_average_soil2soil(self, d, ALPHA_SOIL=None):
    #     pathLoss = 20 * np.log10(4 * pi * d / LAMBDA)
    #     return PS_0 - d * ALPHA_SOIL - pathLoss  # in dBm

    def toa_squared_error_soil2soil(self, xyz, toa_observed_from_other_sensors):
        xyz = xyz.reshape(-1, 3) * (1, 1, 1 / z_scale)
        sq_error = 0.
        for i in range(xyz.shape[-2]):
            for j in range(xyz.shape[-2]):
                if not toa_observed_from_other_sensors[i, j] > 0:
                    continue
                dist = np.sqrt(np.sum((xyz[i] - xyz[j]) ** 2))
                mean_toa = dist / speed_soil
                sq_error += ((toa_observed_from_other_sensors[i, j] - mean_toa) ** 2) * scale_lst_sq_obj_fn
                # sigma_toa = sigma_soil2Soil(dist, ALPHA_SOIL)
        return sq_error


    # def sigma_soil2Soil_reflected(self, d, ALPHA_SOIL):
    #     specularPowerdB = self.p_average_soil2soil_reflected(d, ALPHA_SOIL)
    #     specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
    #     TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
    #     sigma = np.sqrt(8 * (pi ** 2) * specular_power * T_S * self.beta_sq / TOTAL_NOISE) ** -1
    #     return sigma

    # def sigma_soil2soil_cross(self, d_dir, d_ref, ALPHA_SOIL):
    #     specularPowerDirdB = self.p_average_soil2soil(d_dir, ALPHA_SOIL)
    #     specular_power_dir = 10 ** ((specularPowerDirdB - 30) / 10.0)
    #
    #     specularPowerRefdB = self.p_average_soil2soil_reflected(d_ref, ALPHA_SOIL)
    #     specular_power_ref = 10 ** ((specularPowerRefdB - 30) / 10.0)
    #     TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
    #
    #     sigma = (8 * (pi ** 2) * np.sqrt(specular_power_dir * specular_power_ref) * T_S * self.beta_sq / TOTAL_NOISE) ** -1
    #     # sigma=np.array([0])
    #     return sigma

    def toa_neg_log_likelihood_soil2soil(self, xyz, toa_observed_from_other_sensors_dir,
                                         toa_observed_from_other_sensors_reflected):
        xyz = xyz.reshape(-1, 3) * (1, 1, 1 / z_scale)
        neg_log_likelihood = 0.
        # for direct path
        for i in range(xyz.shape[-2]):
            for j in range(xyz.shape[-2]):
                if i == j:
                    continue
                if toa_observed_from_other_sensors_dir[i, j] > 0 and toa_observed_from_other_sensors_reflected[
                    i, j] > 0:
                    dist_dir = np.sqrt(np.sum((xyz[i] - xyz[j]) ** 2))
                    mean_toa_dir = dist_dir / speed_soil
                    sigma2_toa_dir = self.sigma_soil2Soil(dist_dir, self.alpha_s) ** 2
                    # neg_log_likelihood += ((toa_observed_from_other_sensors_dir[i, j] - mean_toa) ** 2) /sigma2_toa
                    # neg_log_likelihood += 0.5*np.log(sigma2_toa)

                    if xyz[i, 2] == 0 or xyz[j, 2] == 0:
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
                    sigma2_toa_ref = self.sigma_soil2Soil_reflected(dist_ref, self.alpha_s) ** 2
                    sigma2_cross = self.sigma_soil2soil_cross(dist_dir, dist_ref, self.alpha_s)
                    # neg_log_likelihood += ((toa_observed_from_other_sensors_reflected[i, j] - mean_toa_ref) ** 2) / sigma2_toa_ref
                    # neg_log_likelihood += 0.5 * np.log(sigma2_toa_ref)

                    t = np.asarray([toa_observed_from_other_sensors_dir[i, j],
                                    toa_observed_from_other_sensors_reflected[i, j]]).reshape(2, 1)
                    t_bar = np.asarray([mean_toa_dir, mean_toa_ref])
                    sigma = np.asarray([[1 / sigma2_toa_dir[0], sigma2_cross], [sigma2_cross, 1 / sigma2_toa_ref[0]]])

                    a = sigma.dot(t - t_bar)
                    b = (t - t_bar).T
                    # neg_log_likelihood += ((t-t_bar).dot(sigma)).dot((t-t_bar).T)
                    neg_log_likelihood += b.dot(a)
                    neg_log_likelihood += 0.5 * np.log(np.linalg.det(sigma))
                    # neg_log_likelihood += 0.5 * np.log(sigma2_toa_dir)
                    # neg_log_likelihood += 0.5 * np.log(sigma2_toa_ref)

                else:
                    raise Exception("test")

                    # if toa_observed_from_other_sensors_reflected[i,j] > 0:
                    # print("i={},j={}".format(i,j))

        return 100*(neg_log_likelihood * scale_neg_log_likelihood_soil2soil)

if __name__ == "__main__":
    H = 200
    F = 100
    NUM_SENSORS = 5
    num_samples_1 = 0
    xy_error1 = 0
    z_error1 = 0
    num_samples_2 = 0
    xy_error2 = 0
    z_error2 = 0
    # np.random.seed(0)

    localizer = ToaLocalizer(epsilon_index=0, f=833 * 10 ** 6)    # epsilon index = 0
    speed_soil = localizer.speed_soil
    for i in range(NUM_RUNS):

        actual_s_locations = np.random.random_sample(size=(NUM_SENSORS, 3)) * (F, F, -H / H)  # X, Y, Z coordinates
        # print("actual_s_locations.shape = {}".format(actual_s_locations.shape))
        print("Actual sensor locations = \n{}".format(actual_s_locations))
        xyz0 = np.random.random_sample(size=(NUM_SENSORS, 3)) * (F, F, -H / H)  # X, Y, Z coordinates
        # print("Guessed sensor location = {}".format(xyz0))

        toa_observed_from_anchors = []
        for xyzOneSensorActual in actual_s_locations:
            toa_from_anchors = []
            for anchor in localizer.anchor_locations[:4]:  # For anchors above soil
                # assert anchor.shape == xyzOneSensorActual.shape
                distance = np.sqrt(np.sum((anchor - xyzOneSensorActual) ** 2))

                # power = p_average_soil2soil(distance, ALPHA_SOIL)   # Received signal power for soil anchors
                distanceAir = np.sqrt((anchor[0] - xyzOneSensorActual[0]) ** 2 + \
                                   (anchor[1] - xyzOneSensorActual[1]) ** 2 + \
                                   (anchor[2]) ** 2)
                distanceSoil = abs(xyzOneSensorActual[2])
                power = localizer.p_average_air2soil(distanceAir, distanceSoil)
                # print("power = {} dB".format(power))
                if (power > -110.0):
                    mean = distanceAir / speed + distanceSoil / speed_soil
                    sigma = localizer.sigma_air2Soil(distanceAir, distanceSoil)
                    # print("Sampling t with mean = {} and sigma = {}".format(mean, sigma))
                    ob = np.random.normal(loc=mean, scale=sigma, size=1)[0] #- 5e-9  TODO Add synch error here
                    toa_from_anchors.append(ob)
                else:
                    raise AssertionError("Invalid power")
                    toa_from_anchors.append(np.nan)
            toa_observed_from_anchors.append(toa_from_anchors)

        toa_observed_from_anchors = np.asarray(toa_observed_from_anchors)

        toa_observed_from_other_sensors_dir = np.zeros(shape=(NUM_SENSORS, NUM_SENSORS))
        toa_observed_from_other_sensors_reflected = np.zeros(shape=(NUM_SENSORS, NUM_SENSORS))

        for i in range(actual_s_locations.shape[-2]):
            for j in range(actual_s_locations.shape[-2]):
                if i == j:
                    continue
                dist_dir = np.sqrt(np.sum((actual_s_locations[i] - actual_s_locations[j]) ** 2))

                power_dir = localizer.p_average_soil2soil(dist_dir)

                d1 = np.sqrt((actual_s_locations[i, 2]) ** 2 + \
                             ((actual_s_locations[i, 0] - actual_s_locations[j, 0]) ** 2 + \
                              (actual_s_locations[i, 1] - actual_s_locations[j, 1]) ** 2) *
                             ((actual_s_locations[i, 2] / (actual_s_locations[i, 2] + actual_s_locations[j, 2])) ** 2))

                d2 = np.sqrt((actual_s_locations[j, 2]) ** 2 + \
                             ((actual_s_locations[i, 0] - actual_s_locations[j, 0]) ** 2 + \
                              (actual_s_locations[i, 1] - actual_s_locations[j, 1]) ** 2) *
                             ((actual_s_locations[j, 2] / (actual_s_locations[i, 2] + actual_s_locations[j, 2])) ** 2))

                dist_ref = d1 + d2

                power_ref = localizer.p_average_soil2soil_reflected(dist_ref)

                # print("power = {} dB".format(power))

                if power_dir > N_0 and power_ref > N_0:
                    mean_toa_dir = dist_dir / speed_soil
                    sigma_toa_dir = localizer.sigma_soil2Soil(dist_dir)

                    mean_toa_ref = dist_ref / speed_soil
                    sigma_toa_ref = localizer.sigma_soil2Soil_reflected(dist_ref)

                    sigma2_cross = localizer.sigma_soil2soil_cross(dist_dir, dist_ref)

                    mean = (mean_toa_dir[0], mean_toa_ref[0])
                    cov = [[sigma_toa_dir[0], sigma2_cross[0]], [sigma2_cross[0], sigma_toa_ref[0]]]
                    cov = [[sigma_toa_dir[0], 0], [0, sigma_toa_ref[0]]]

                    ob = np.random.multivariate_normal(mean, cov, 1)

                    ob_dir = ob[0][0]
                    ob_ref = ob[0][1]

                    ob_dir = np.random.normal(loc=mean_toa_dir, scale=sigma_toa_dir, size=1)[0]
                    ob_ref = np.random.normal(loc=mean_toa_ref, scale=sigma_toa_ref, size=1)[0]

                    toa_observed_from_other_sensors_reflected[i, j] = ob_dir
                    toa_observed_from_other_sensors_dir[i, j] = ob_ref
                else:
                    print("****************power alert***********")

        # print("toa_observed_from_anchors = {}".format(toa_observed_from_anchors))


        # neg_likelihood = localizer.toa_neg_log_likelihood(actual_s_locations * (1, 1, z_scale),
        #                                                  toa_observed_from_anchors,
        #                                                  speed_soil)
        #
        # # print("neg_likelihood at minima = {}".format(neg_likelihood))
        # neg_likelihood = localizer.toa_neg_log_likelihood(xyz0 * (1, 1, z_scale),
        #                                                  toa_observed_from_anchors,
        #                                                  speed_soil)

        # print("neg_likelihood at other = {}".format(neg_likelihood))


        bnds = (((None, None),)*2+((-100,0),))*NUM_SENSORS
        result = minimize(localizer.toa_neg_log_likelihood,
                          xyz0 * (1, 1, z_scale),
                          args=(toa_observed_from_anchors),
                          # method="Nelder-Mead",
                          method="L-BFGS-B",
                          bounds=bnds,
                          options={"maxiter":2e2})
        print("stage 1 success ? {}".format(result.success))
        est_s_locations_stage1 = result.x.reshape(-1, 3) * (1, 1, 1 / z_scale)
        #
        # np.savetxt("data/est_stage1.csv", estimated_locations)
        # np.savetxt("data/act_sensor_loc.csv", actual_s_locations)

        if result.success == True:
            print("num_samples 1= {}".format(num_samples_1))
            print("estimated locations 1= \n{}".format(est_s_locations_stage1))
            # print("actual locations 1 = \n{}".format(actual_s_locations))
            num_samples_1 += 1
            xy_error1 += np.sum(np.sqrt(np.sum((est_s_locations_stage1[:, :2] - actual_s_locations[:, :2]) ** 2, axis=1)))
            z_error1  += np.sum(np.abs(est_s_locations_stage1[:, 2] - actual_s_locations[:, 2]))
            # print("z_error = {}".format(np.sum(np.abs(estimated_locations[:,2]-actual_s_locations[:,2]))))



        ### Stage 2
        # print("observed toa from other sensors (dir)=\n{}".format(toa_observed_from_other_sensors_dir))
        # print("observed toa from other sensors (reflected) =\n{}".format(toa_observed_from_other_sensors_reflected))

        # xyz0 = np.random.random_sample(size=(NUM_SENSORS, 3)) * (F, F, -H / H)  # X, Y, Z coordinates
        xyz0 = est_s_locations_stage1

        # sq_error_at_actual = localizer.toa_neg_log_likelihood_soil2soil(actual_s_locations * (1, 1, z_scale),
        #                                                       toa_observed_from_other_sensors_dir,
        #                                                       toa_observed_from_other_sensors_reflected)
        # print("squared error at actual = {}".format(sq_error_at_actual))
        #
        # sq_error_at_seed = localizer.toa_neg_log_likelihood_soil2soil(xyz0 * (1, 1, z_scale),
        #                                                     toa_observed_from_other_sensors_dir,
        #                                                     toa_observed_from_other_sensors_reflected)
        # print("squared error at seed = {}".format(sq_error_at_seed))

        bnds = (((None, None),) * 2 + ((-100, -1),)) * NUM_SENSORS
        lower = est_s_locations_stage1 - np.asarray([0.01, 0.01, 0.0001])
        upper = est_s_locations_stage1 + np.asarray([0.01, 0.01, 0.0001])
        # print(bnds)
        bnds = list(((bb[0], cc[0]), (bb[1], cc[1]), (bb[2] * z_scale, cc[2] * z_scale)) for bb, cc in zip(lower, upper))
        bnds = np.asarray(bnds).reshape(-1, 2)
        # print(bnds)

        result = minimize(localizer.toa_neg_log_likelihood_soil2soil,
                          xyz0 * (1, 1, z_scale),
                          args=(toa_observed_from_other_sensors_dir,
                                toa_observed_from_other_sensors_reflected),
                          # method="L-BFGS-B",
                          method="TNC",
                          # method='SLSQP',
                          bounds=bnds,
                          options={"maxiter": int(2e2)})
        print("stage 2 success ? {}".format(result.success))
        est_s_locations_stage2 = result.x.reshape(-1, 3) * (1, 1, 1 / z_scale)
        # print("actual =\n{}".format(actual_s_locations))
        # print("stage 1=\n{}".format(est_s_locations_stage1))
        # print("stage 2 =\n{}".format(est_s_locations_stage2))
        # np.savetxt("data/est_stage2.csv", est_s_locations_stage2)
        # sq_error_at_est = localizer.toa_neg_log_likelihood_soil2soil(est_s_locations_stage2 * (1, 1, z_scale),
        #                                                    toa_observed_from_other_sensors_dir,
        #                                                    toa_observed_from_other_sensors_reflected)
        # print("squared error at est_s_locations_stage2 = {}".format(sq_error_at_est))
        if result.success == True:
            print("num_samples 2= {}".format(num_samples_2))
            print("estimated locations 2= \n{}".format(est_s_locations_stage2))
            # print("actual locations 2 = \n{}".format(actual_s_locations))
            num_samples_2 += 1
            xy_error2 += np.sum(np.sqrt(np.sum((est_s_locations_stage2[:, :2] - actual_s_locations[:, :2]) ** 2, axis=1)))
            z_error2 += np.sum(np.abs(est_s_locations_stage2[:, 2] - actual_s_locations[:, 2]))

    print("xy error = {}".format(xy_error1 / (num_samples_1 * NUM_SENSORS)))
    print("z error = {}".format(z_error1 / (num_samples_1 * NUM_SENSORS)))
    print("xy error 2 = {}".format(xy_error2 / (num_samples_2 * NUM_SENSORS)))
    print("z error 2= {}".format(z_error2 / (num_samples_2 * NUM_SENSORS)))


