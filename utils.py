from math import pi
from math import log10
from math import sqrt

import matplotlib.pyplot as plt


from matplotlib import rc
import os
import numpy as np

rc('text', usetex=True)
rc('font', family='serif')

rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

PA_0 = 38 # 38 dBm
PS_0 = 10  # 19 dBm
LAMBDA = 0.7
N_0 = -110.0  #dBm
T_S = 10.0 ** (-3)

FREQUENCY = 833 * 10 ** 6
BETA_SQ = FREQUENCY ** 2
OMEGA = 2 * pi * FREQUENCY

speed = 3. * 10

F = 50.   #Field size, in m
H = 100.    # Maximum dept of sensor node, in m
NUM_SENSORS = 1

MU_0 = 4 * pi * 10 ** (-7)
MU_A = MU_0
MU_S = 1.0084 * MU_0
EPSILON_0 = 8.85 * 10 ** (-12)
EPSILON_AIR = EPSILON_0

EPSILON_SOIL = np.asarray((2.361970519728 + 0.096670930496j,
                            5.08697889259241 + 0.4413468113884j,
                            16.4109595611802 + 2.36876126519685j,
                            24.4855102012741 + 3.85704851601056j)) * EPSILON_0


# EPSILON_SOIL = {0: {"real": 2.361970519728 * EPSILON_0, "img": 0.096670930496 * EPSILON_0},
#            1: {"real": 5.08697889259241 * EPSILON_0, "img": 0.4413468113884 * EPSILON_0},
#             2: {"real": 16.4109595611802 * EPSILON_0, "img": 2.36876126519685 * EPSILON_0},
#             3: {"real": 24.4855102012741 * EPSILON_0, "img": 3.85704851601056 * EPSILON_0}}


"""
returns average air to soil power in dB
"""
def p_average_air2soil(d_a, d_s, ALPHA_SOIL, TAU):
  pathLossAir = 20 * log10(4 * pi * d_a / LAMBDA)
  pathLossSoil = ALPHA_SOIL * d_s
  return PA_0 - pathLossAir - pathLossSoil


def sigma_air2Soil(d_a, d_s, ALPHA_SOIL, TAU):
  specularPowerdB = p_average_air2soil(d_a, d_s, ALPHA_SOIL, TAU)
  specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
  TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
  sigma = sqrt(8 * (pi ** 2) * specular_power * T_S * BETA_SQ / TOTAL_NOISE) ** -1
  return sigma

def p_average_soil2soil(d, ALPHA_SOIL):
    pathLoss = 20 * log10(4 * pi * d / LAMBDA)
    return PS_0 - d * ALPHA_SOIL -pathLoss #in dBm


def sigma_soil2Soil(d, ALPHA_SOIL):
  specularPowerdB = p_average_soil2soil(d, ALPHA_SOIL)
  specular_power = 10 ** ((specularPowerdB - 30) / 10.0)
  TOTAL_NOISE = 10 ** ((N_0 - 30.0) / 10.0)
  sigma = (8 * (pi ** 2) * specular_power * T_S * BETA_SQ / TOTAL_NOISE) ** -1
  return sigma

# for soil to soil
def differenceX1(est, obs):
    val = 0.0
    for key in sorted(est):
        val = val + (est[key] - obs[key]) ** 2
    return val

#Anchor to sensor
def difference(est, obs):
    val = 0.0
    for key in obs:  #above or below
        #print "len = ", len(obs[key])
        if key == "above" and len(obs[key]) != 4:
            raise Exception('Not all anchor nodes signals were received')
        for anchorNum in obs[key]:
            val = val + (est[key][anchorNum] - obs[key][anchorNum]) ** 2
    return val

def differenceAnchorsToAllSensor(est, obs):
    val = 0.0
    for key in obs:  #above or below
        i = 0
        #print "obs[key] = ", obs[key], "key = ", key
        #print "est[key] = ", est[key], "key = ", key
        for observedSensorData in obs[key]: # get the dictionary for a sensor
            #print "obs[key][i] = ", obs[key][i], "i = ", i
            #print "est[key][i] = ", est[key][i], "i = ", i
            for anchorNum in observedSensorData:
                val = val + (obs[key][i][anchorNum] - est[key][i][anchorNum]) ** 2
            i = i + 1
    return val

def timeOfArrialMatcher3DX(arg, tObs, anchors, tObsSoil2Soil, speed_soil):

    xyzAll = ()
    for i in range(len(arg)/3):
        xyz = arg[i * 3: (i + 1) * 3]
        xyzAll = xyzAll + (xyz,)

    estimatedTime3D = {"above": (), "below": ()}

    #Anchor to Sensor
    i = 0
    for xyzA in xyzAll:
        #print "i = ", i
        #i = i + 1
        #print "len(xyzAll) = ", len(xyzAll)
        estimated = {}
        anchorId = 0
        for anchor in anchors["below"]:
            distance = sqrt((anchor[0] - xyzA[0]) ** 2 +\
                (anchor[1] - xyzA[1]) ** 2 +\
                (anchor[2] - xyzA[2]) ** 2)
            estimated[anchorId] = distance / speed_soil
            anchorId = anchorId + 1
        estimatedTime3D["below"] = estimatedTime3D["below"] + (estimated, )

        estimated = {}
        anchorId = 0
        for anchor in anchors["above"]:
            distanceAir = sqrt((anchor[0] - xyzA[0]) ** 2 + (anchor[1] - xyzA[1]) ** 2 + (anchor[2]) ** 2)
            distanceSoil = abs(xyzA[2])
            estimated[anchorId] = distanceAir / speed + distanceSoil / speed_soil
            anchorId = anchorId + 1
        estimatedTime3D["above"] = estimatedTime3D["above"] + (estimated, )

    #sensor to sensor
    estSS = {}

    for i in range(len(xyzAll)):
        for j in range(i+1, len(xyzAll)):
            if tObsSoil2Soil.get(str(i)+str(j)) == None:
                continue
            xyzi = xyzAll[i]
            xyzj = xyzAll[j]
            dist = sqrt((xyzi[0] - xyzj[0]) ** 2 + (xyzi[1] - xyzj[1]) ** 2 +(xyzi[2] - xyzj[2]) ** 2)
            est = dist / speed_soil
            estSS[str(i) + str(j)] = est
    return differenceAnchorsToAllSensor(estimatedTime3D, tObs) + differenceX1(estSS, tObsSoil2Soil)

def timeOfArrivalMatcherAnchorsToSensors(xyz, toa_observed_from_anchors, anchor_locations, speed_soil):

    distance_air = np.sqrt((xyz[:,:2]-toa_observed_from_anchors[:,:2])**2 + toa_observed_from_anchors[:,2]**2)
    distance_soil = abs(xyz[:,2])

    estimated_toa = distance_air / speed + distance_soil / speed_soil
    return (toa_observed_from_anchors-estimated_toa)**2


#for estimating location of one sensor
def timeOfArrivalMatcherAnchorToOneSensor(arg, sensorId, tObs, anchors, speed_soil):

    xyzA = arg
    estimatedTime3D = {"above": (), "below": ()}

    #Anchor to Sensor
    for anchorKey in sorted(anchors):
        estimated = {}
        i = 0
        for anchor in anchors["below"]:
            distance = sqrt((anchor[0] - xyzA[0]) ** 2 + (anchor[1] - xyzA[1]) ** 2 + (anchor[2] - xyzA[2]) ** 2)
            estimated[i] = distance / speed_soil
            i = i + 1
        estimatedTime3D["below"] = estimated
        estimated = {}
        i = 0
        # print "Number of anchors above = {}".format(len(anchors["above"]))
        for anchor in anchors["above"]:
            distanceAir = sqrt((anchor[0] - xyzA[0]) ** 2 + (anchor[1] - xyzA[1]) ** 2 + (anchor[2]) ** 2)
            distanceSoil = abs(xyzA[2])
            estimated[i] = distanceAir / speed + distanceSoil / speed_soil
            i = i + 1
        estimatedTime3D["above"] = estimated

    return difference(estimatedTime3D, {key:tObs[key][sensorId] for key in tObs})


def plot(xActual, yActual, xFirstEst, yFirstEst, xEst, yEst, EPSILON_S_REAL, EPSILON_S_IMG, params_text, xLabel, yLabel, limits):
    plt.plot(xActual, yActual, 'b.', label = "Actual")
    plt.plot(xFirstEst, yFirstEst, 'rx', label = "Estimated using Anchor to Sensor model only")
    plt.plot(xEst, yEst, 'gx', label = "Estimated")
    plt.axis(limits)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.text(plt.xlim()[1]*1.3, plt.ylim()[0],  params_text,
        horizontalalignment='center',
        verticalalignment='bottom')

    FILENAME_PREFIX = "TOA_" + "EPSILON_S_REAL_" + \
        '{0:.3g}'.format(EPSILON_S_REAL/EPSILON_0) + \
        "EPSILON_S_IMG_" + \
        '{0:.3g}'.format(EPSILON_S_IMG/EPSILON_0)

    plt.savefig(FILENAME_PREFIX + '_' + xLabel + yLabel + '.pdf', bbox_inches='tight', pad_inches = 0.5)
    plt.clf()
