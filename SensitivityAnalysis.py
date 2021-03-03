
from scipy.optimize import minimize
from math import sqrt
from random import random
from random import seed
import numpy
import UtilsSensitivityAnalysis
reload(UtilsSensitivityAnalysis)
from UtilsSensitivityAnalysis import *

seed()

xyzAllActual = ()
for i in range(NUM_SENSORS):
    xyzAllActual = xyzAllActual + ((random() * F, random() * F, random() * (-H/200)), )

epsilon_index = 3

xEstSamples = [[] for _ in range(NUM_SENSORS)]
yEstSamples = [[] for _ in range(NUM_SENSORS)]
zEstSamples = [[] for _ in range(NUM_SENSORS)]
errors = []


for ctr in range(1):
    #print "********************************************"
    #print "EPSILON_INDEX = ", epsilon_index
    #print "********************************************"
    EPSILON_S_REAL = EPSILON_SOIL[epsilon_index]["real"]
    EPSILON_S_IMG = EPSILON_SOIL[epsilon_index]["img"]


    ALPHA_SOIL = OMEGA * sqrt( (MU_S * EPSILON_S_REAL/2) * (sqrt(1 + (EPSILON_S_IMG/EPSILON_S_REAL) ** 2) - 1))
    RHO = ((sqrt(MU_A / EPSILON_AIR) - sqrt(MU_S / EPSILON_S_REAL)) / \
        (sqrt(MU_A / EPSILON_AIR) + sqrt(MU_S / EPSILON_S_REAL))) ** 2

    TAU = 1 - RHO

    speed_soil = ( \
        sqrt( \
        (MU_S * EPSILON_S_REAL / 2 ) * (sqrt(1 + (EPSILON_S_IMG/EPSILON_S_REAL) ** 2) + 1)\
        ) \
    ) \
        ** (-1) * (10 ** (-7))


    AnchorsXYZ = {"above": [(0., 0., H), (F, 0., H), (0., F, H), (F, F, H)], \
        "below": [(0., 0., -H/4), (F, 0., -H/4), (0., F, -H/4), (F, F, -H/4)]}
    #AnchorsXYZ = {"above": [(0., 0., H), (F, 0., H), (0., F, H), (F, F, H)]}

    obsTAnchors = {"above": (), "below": ()}

    #Anchor to sensor observations
    for xyzOneSensorActual in xyzAllActual:
        observed = {}
        i = 0
        for anchor in AnchorsXYZ["below"]:
            distance = sqrt((anchor[0] - xyzOneSensorActual[0]) ** 2 +\
                    (anchor[1] - xyzOneSensorActual[1]) ** 2 +\
                    (anchor[2] - xyzOneSensorActual[2]) ** 2)
            power = p_average_soil2soil(distance, ALPHA_SOIL)
            if (power > -110.0):
                mean = distance / speed_soil + random()/40
                sigma = sigma_soil2Soil(distance, ALPHA_SOIL)
                ob = numpy.random.normal(loc = mean, scale = sigma, size = 1)[0]
                observed[i] = numpy.random.uniform(ob-DELTA_T, ob+DELTA_T)
            i = i + 1
        obsTAnchors["below"] = obsTAnchors["below"] + (observed, )

        observed = {}
        i = 0
        for anchor in AnchorsXYZ["above"]:
                distanceAir = sqrt((anchor[0] - xyzOneSensorActual[0]) ** 2 +\
                    (anchor[1] - xyzOneSensorActual[1]) ** 2 +\
                    (anchor[2]) ** 2)
                distanceSoil = abs(xyzOneSensorActual[2])
                signalPowerdB = p_average_air2soil(distanceAir, distanceSoil, ALPHA_SOIL, TAU)
                if signalPowerdB > -110.0:
                    mean = distanceAir / speed + distanceSoil / speed_soil
                    sigma = sigma_air2Soil(distanceAir, distanceSoil, ALPHA_SOIL, TAU)
                    ob = numpy.random.normal(loc = mean, scale = sigma, size = 1)[0]
                    observed[i] = numpy.random.uniform(ob-DELTA_T, ob+DELTA_T)
                i = i + 1
                #observed = observed + ((distanceAir + distanceSoil) / speed + random()/400, )
        obsTAnchors["above"] = obsTAnchors["above"] + (observed, )

    observedTime3DSS = {}

    # Sensor to sensor observations
    f = open("SensorNeighbors_" + str(ctr) + "_" + str(NUM_SENSORS) + ".csv", "w")
    for i in range(len(xyzAllActual)):
        f.write(str(i))
        for j in range(i+1, len(xyzAllActual)):
            xyzi = xyzAllActual[i]
            xyzj = xyzAllActual[j]
            dist = sqrt((xyzi[0] - xyzj[0]) ** 2 + (xyzi[1] - xyzj[1]) ** 2 +(xyzi[2] - xyzj[2]) ** 2)
            signalPowerdB = p_average_soil2soil(dist, ALPHA_SOIL)

            if signalPowerdB > -110.0:
                f.write("," + str(j))
                mean = dist / speed_soil
                sigma = sigma_soil2Soil(dist, ALPHA_SOIL)
                observed = numpy.random.normal(loc = mean, scale = sigma, size = 1)[0]
                observedTime3DSS[str(i) + '-' + str(j)] = numpy.random.uniform(observed-DELTA_T, observed+DELTA_T)
        f.write("\n")
    f.close()


    # Estimation
    f = open("sensitivity_analysis" + str(ctr) + "_" + str(NUM_SENSORS) +".csv", "w")
    f.write("X,Y,Z,X (est1),Y(est1),Z(est1),X(est),Y(est),Z(est)\n")
    for epsilon_ctr in range(0, 101, 5):
        success = True
        EPSILON_S_REAL_EST = EPSILON_S_REAL * (100.0 + epsilon_ctr) / 100.0
        EPSILON_S_IMG_EST = EPSILON_S_REAL * (100.0 + epsilon_ctr) / 100.0
        MU_S_EST = MU_S * (100.0 + epsilon_ctr) / 100.0

        speed_soil_est = ( \
            sqrt( \
                (MU_S * EPSILON_S_REAL / 2 ) * (sqrt(1 + (EPSILON_S_IMG/EPSILON_S_REAL) ** 2) + 1)\
                ) \
            ) ** (-1) * (10 ** (-7))


        xyzFirstEst = ()
        for i in range(NUM_SENSORS):
            xyzInitialEstimate = (random() * F, random() * F, random() * (-H/200))
            res = minimize(timeOfArrivalMatcherAnchorToOneSensor, xyzInitialEstimate, args=(i, obsTAnchors, AnchorsXYZ, observedTime3DSS, speed_soil_est), method='Nelder-Mead', \
                options = {"maxiter":1e6})
            success = success & res.success
            e = res.x
            if (e[2] > 0 ):
               e[2] = -e[2]
            #print res.success
            xyzFirstEst = xyzFirstEst  + ((e[0], e[1], e[2]),)

            #print xyzAllActual[i]
            #print e
            #print "*****"
        #print xyzAllActual
        #print "xyzFirstEst = ", xyzFirstEst
        print "success (step 1) = ", success
        #res = minimize(timeOfArrialMatcher3DX, [element for tupl in xyzFirstEst for element in tupl], \
        #    args=(obsTAnchors, AnchorsXYZ, observedTime3DSS, speed_soil_est), method='Nelder-Mead', options = {"maxiter":1e6})
        res = minimize(timeOfArrialMatcher3DX, xyzFirstEst, \
            args=(obsTAnchors, AnchorsXYZ, observedTime3DSS, speed_soil_est), method='Nelder-Mead', options = {"maxiter":1e6})
        success = success & res.success
        print "success (step 2) = ", success

        e = [(res.x[i * 3: (i+1) * 3]) for i in range(NUM_SENSORS)]

        #for i in range(len(xyzAllActual)):
        #    print "||||||"
        #    print xyzAllActual[i]
        #    print e[i]

        params_text = r"\noindent$\epsilon_s'$ = " + '{0:.3g}'.format(EPSILON_S_REAL/EPSILON_0) + r"\\\\" \
        + r"$\epsilon_s''$ = " + '{0:.3g}'.format(EPSILON_S_IMG/EPSILON_0) + r"\\\\" \
        + r"$\alpha^{(s)}$ = " + '{0:.3g}'.format(ALPHA_SOIL) + r" N/m\\\\"

        xActual = [xyz[0] for xyz in xyzAllActual]
        yActual = [xyz[1] for xyz in xyzAllActual]
        zActual = [xyz[2] for xyz in xyzAllActual]

        xFirstEst = [xyz[0] for xyz in xyzFirstEst]
        yFirstEst = [xyz[1] for xyz in xyzFirstEst]
        zFirstEst = [xyz[2] for xyz in xyzFirstEst]

        xEst = [xyz[0] for xyz in e]
        yEst = [xyz[1] for xyz in e]
        zEst = [xyz[2] for xyz in e]

        err = 0.0
        for i in range(len(xActual)):
            d = (xEst[i] - xActual[i]) ** 2 + \
                (yEst[i] - yActual[i]) ** 2 + \
                (zEst[i] - zActual[i]) ** 2
            err = err + sqrt(d)
        #if err > 10.0:
        #    continue

        #print "DELTA_T = ", DELTA_T, "error = ", err
        #print "T_S = ", T_S, "error = ", err
        print epsilon_ctr, "% epsilon_index = ", epsilon_index, "error = ", err / NUM_SENSORS
        print "res.fun = ", res.fun
        errors.append(err)
        for i in range(NUM_SENSORS):
            xEstSamples[i].append(e[i][0])
            yEstSamples[i].append(e[i][1])
            zEstSamples[i].append(e[i][2])


        #for i in range(NUM_SENSORS):
        #    f.write(str(xyzAllActual[i][0]) + "," + str(xyzAllActual[i][1]) + "," + str(xyzAllActual[i][2]))
        #    f.write("," + str(xyzFirstEst[i][0]) + "," + str(xyzFirstEst[i][1]) + "," + str(xyzFirstEst[i][2]))
        #    f.write("," + str(e[i][0]) + "," + str(e[i][1]) + "," + str(e[i][2]) + "\n")
        #f.close()

#print "max error = ", numpy.max(errors) / NUM_SENSORS
