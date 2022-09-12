#!/usr/bin/python

# This code is only to be used for research purposes.

# Please also send us your name, research group, and references to any publications that may result.

""" a simulation of MBTI personalities """
# sample command to run script: python mbti_parallel.py s-1 r1 t50 n0.1 i m2
# this means we are running random seed (s-1), 1 run (r1), 50 timesteps per run (t50) and 10% noise (n0.1) and introverts team experiment (i)

# SN functions
# FT functions
# multiple agents
# multiple runs
# EI functions: determines distance threshold
# use gaussian process for intuitives
# ability to specify team composition
# plot each step for each agent as subplots
# add option to specify starting region
# add new functions
# add ability to specify mbti and starting point for each agent
# agent average performance
# add in T functions (previously they were all F)
# clean up weights for judging function
# investigating 8 functions
# use 5 nearest neighbour
# make is so that neighbours information is always from the previous timestep
# add perceived fitness
# use sample standard deviation
# fix bug so that e is used
# change velocity clamping method so that it smoothes out the curve
# change auxiliary clamping method so that it never exceeds dominant function

import copy
import csv
import math
import random
import time
import numpy as np
from sklearn import gaussian_process
import sys
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import argparse

seed_number = int(sys.argv[1][1:])
number_of_runs = int(sys.argv[2][1:])
number_of_timesteps = int(sys.argv[3][1:])
noise_level = float(sys.argv[4][1:])
experiment = sys.argv[5]
number_of_teams = int(sys.argv[6][1:])

# constants
RANDOM_SEED = seed_number  # int, actual seed or -1 for currentseed_number = int(sys.argv[1][1:])
number_of_runs = int(sys.argv[2][1:])
number_of_timesteps = int(sys.argv[3][1:])
noise_level = float(sys.argv[4][1:])
experiment = sys.argv[5]
number_of_teams = int(sys.argv[6][1:])

# constants
RANDOM_SEED = seed_number  # int, actual seed or -1 for current time
NUM_RUNS = number_of_runs  # int
NUM_TIMESTEPS = number_of_timesteps  # int
NOISE = noise_level  # 0.2 # float between 0 and 1. if NOISE = 0, then perception is 100% accurate. if NOISE = 1, then perception is random. time
NUM_RUNS = number_of_runs  # int
NUM_TIMESTEPS = number_of_timesteps  # int
NOISE = noise_level  # 0.2 # float between 0 and 1. if NOISE = 0, then perception is 100% accurate. if NOISE = 1, then perception is random.
NUM_TEAMS = number_of_teams  # 2 #2 teams and each work on one dimension x and y respectively

TEAM_NUMBERS = range(NUM_TEAMS) #team indices

MIN = -100  # int, minimum x and y point
MAX = 100  # int, maximum x and y point
VELOCITY_CAP = 5.0  # float, maximum velocity allowed
FAIL_CRITERIA = (MAX-MIN)*0.1 # float, the percentage allowed for team to be apart from each other

MEETING_POINT = [40,NUM_TIMESTEPS] # meeting takes one timestep
ALL_ATTEND_MEETING = False

PLOT = False  # boolean, whether perception plots should be drawn
PATH_PLOT = False  # boolean, whether path plots should be drawn

if NUM_RUNS > 5:
    PLOT = False  # fail safe. don't plot when there are more than 5 runs
    PATH_PLOT = False

# derived globals
MAX_TIMESTEP = NUM_TIMESTEPS + 1
START_TIMESTEP = 1  # timestep 0 is used to set up agents
VELOCITY_CAP_SQUARE = VELOCITY_CAP * VELOCITY_CAP

if PLOT:  # create the x and y arrays for the heatmap
    COLORMAP = 'jet'  # 'PiYG'
    NBINS = 20
    # predxmesh, predymesh = np.meshgrid(np.arange(MIN, MAX), np.arange(MIN, MAX), indexing='ij')
    PLOT_TIMESTEPS = [1, 11, 21, 31, 41, 50]  # specify the list of timesteps to plot heatmaps
    # CMAP = plt.get_cmap(COLORMAP)
    # LEVELS = MaxNLocator(nbins=NBINS).tick_values(-1, 1)
    # NORM = BoundaryNorm(LEVELS, ncolors=CMAP.N, clip=True)

if PATH_PLOT:
    AXIS_EXTRA = 0  # int, for plotting wonderers

ANALYSE = True  # boolean, whether results should be analysed
RECORD = False # boolean, whether results should be recorded

USE_MBTI_COMPOSITION = True  # boolean, whether MBTI_COMPOSITION is used to make agents (if false, AGENT_COMPOSITION will be used instead)


def assign_teams(agent_composition):
    """ divide solution space into subspaces that are consistent with the number of agents  """
    num_agent = len(agent_composition)
    teams_size_temp, modulo = divmod(num_agent, NUM_TEAMS)
    teams_size_list = [teams_size_temp] * NUM_TEAMS
    modulo1, modulo2 = divmod(modulo, 2)
    ind = 0
    while modulo1 > 0:
        teams_size_list[ind] = teams_size_temp + 2
        ind += 1
        modulo1 -= 1
    teams_size_list[modulo1] = teams_size_temp + modulo2
    members_team_index = []
    for i in range(NUM_TEAMS):
        members_team_index += ([i] * teams_size_list[i])
    random.shuffle(members_team_index)
    return teams_size_list, members_team_index


if USE_MBTI_COMPOSITION:
    START_X_MIN = -100  # int
    START_X_MAX = 100  # int
    START_Y_MIN = -100  # int
    START_Y_MAX = 100  # int

    MBTI_COMPOSITION = OrderedDict()
    MBTI_COMPOSITION['ISTJ'] = 0
    MBTI_COMPOSITION['ISFJ'] = 0
    MBTI_COMPOSITION['INFJ'] = 0
    MBTI_COMPOSITION['INTJ'] = 0
    MBTI_COMPOSITION['ISTP'] = 0
    MBTI_COMPOSITION['ISFP'] = 0
    MBTI_COMPOSITION['INFP'] = 0
    MBTI_COMPOSITION['INTP'] = 0
    MBTI_COMPOSITION['ESTP'] = 0
    MBTI_COMPOSITION['ESFP'] = 0
    MBTI_COMPOSITION['ENFP'] = 0
    MBTI_COMPOSITION['ENTP'] = 0
    MBTI_COMPOSITION['ESTJ'] = 0
    MBTI_COMPOSITION['ESFJ'] = 0
    MBTI_COMPOSITION['ENFJ'] = 0
    MBTI_COMPOSITION['ENTJ'] = 0
    MBTI_COMPOSITION['Si'] = 0
    MBTI_COMPOSITION['Ni'] = 0
    MBTI_COMPOSITION['Ti'] = 0
    MBTI_COMPOSITION['Fi'] = 0
    MBTI_COMPOSITION['Se'] = 0
    MBTI_COMPOSITION['Ne'] = 0
    MBTI_COMPOSITION['Te'] = 0
    MBTI_COMPOSITION['Fe'] = 0

    if experiment == 'all':
        MBTI_COMPOSITION['ISTJ'] = 1
        MBTI_COMPOSITION['ISFJ'] = 1
        MBTI_COMPOSITION['INFJ'] = 1
        MBTI_COMPOSITION['INTJ'] = 1
        MBTI_COMPOSITION['ISTP'] = 1
        MBTI_COMPOSITION['ISFP'] = 1
        MBTI_COMPOSITION['INFP'] = 1
        MBTI_COMPOSITION['INTP'] = 1
        MBTI_COMPOSITION['ESTP'] = 1
        MBTI_COMPOSITION['ESFP'] = 1
        MBTI_COMPOSITION['ENFP'] = 1
        MBTI_COMPOSITION['ENTP'] = 1
        MBTI_COMPOSITION['ESTJ'] = 1
        MBTI_COMPOSITION['ESFJ'] = 1
        MBTI_COMPOSITION['ENFJ'] = 1
        MBTI_COMPOSITION['ENTJ'] = 1
    elif experiment == 'i':
        MBTI_COMPOSITION['ISTJ'] = 1
        MBTI_COMPOSITION['ISFJ'] = 1
        MBTI_COMPOSITION['INFJ'] = 1
        MBTI_COMPOSITION['INTJ'] = 1
        MBTI_COMPOSITION['ISTP'] = 1
        MBTI_COMPOSITION['ISFP'] = 1
        MBTI_COMPOSITION['INFP'] = 1
        MBTI_COMPOSITION['INTP'] = 1
    elif experiment == 'e':
        MBTI_COMPOSITION['ESTP'] = 1
        MBTI_COMPOSITION['ESFP'] = 1
        MBTI_COMPOSITION['ENFP'] = 1
        MBTI_COMPOSITION['ENTP'] = 1
        MBTI_COMPOSITION['ESTJ'] = 1
        MBTI_COMPOSITION['ESFJ'] = 1
        MBTI_COMPOSITION['ENFJ'] = 1
        MBTI_COMPOSITION['ENTJ'] = 1
    elif experiment == 's':
        MBTI_COMPOSITION['ISTJ'] = 1
        MBTI_COMPOSITION['ISFJ'] = 1
        MBTI_COMPOSITION['ISTP'] = 1
        MBTI_COMPOSITION['ISFP'] = 1
        MBTI_COMPOSITION['ESTP'] = 1
        MBTI_COMPOSITION['ESFP'] = 1
        MBTI_COMPOSITION['ESTJ'] = 1
        MBTI_COMPOSITION['ESFJ'] = 1
    elif experiment == 'n':
        MBTI_COMPOSITION['INFJ'] = 1
        MBTI_COMPOSITION['INTJ'] = 1
        MBTI_COMPOSITION['INFP'] = 1
        MBTI_COMPOSITION['INTP'] = 1
        MBTI_COMPOSITION['ENFP'] = 1
        MBTI_COMPOSITION['ENTP'] = 1
        MBTI_COMPOSITION['ENFJ'] = 1
        MBTI_COMPOSITION['ENTJ'] = 1
    elif experiment == 'f':
        MBTI_COMPOSITION['ISFJ'] = 1
        MBTI_COMPOSITION['INFJ'] = 1
        MBTI_COMPOSITION['ISFP'] = 1
        MBTI_COMPOSITION['INFP'] = 1
        MBTI_COMPOSITION['ESFP'] = 1
        MBTI_COMPOSITION['ENFP'] = 1
        MBTI_COMPOSITION['ESFJ'] = 1
        MBTI_COMPOSITION['ENFJ'] = 1
    elif experiment == 't':
        MBTI_COMPOSITION['ISTJ'] = 1
        MBTI_COMPOSITION['INTJ'] = 1
        MBTI_COMPOSITION['ISTP'] = 1
        MBTI_COMPOSITION['INTP'] = 1
        MBTI_COMPOSITION['ESTP'] = 1
        MBTI_COMPOSITION['ENTP'] = 1
        MBTI_COMPOSITION['ESTJ'] = 1
        MBTI_COMPOSITION['ENTJ'] = 1
    elif experiment == 'j':
        MBTI_COMPOSITION['ISTJ'] = 1
        MBTI_COMPOSITION['ISFJ'] = 1
        MBTI_COMPOSITION['INFJ'] = 1
        MBTI_COMPOSITION['INTJ'] = 1
        MBTI_COMPOSITION['ESTJ'] = 1
        MBTI_COMPOSITION['ESFJ'] = 1
        MBTI_COMPOSITION['ENFJ'] = 1
        MBTI_COMPOSITION['ENTJ'] = 1
    elif experiment == 'p':
        MBTI_COMPOSITION['ISTP'] = 1
        MBTI_COMPOSITION['ISFP'] = 1
        MBTI_COMPOSITION['INFP'] = 1
        MBTI_COMPOSITION['INTP'] = 1
        MBTI_COMPOSITION['ESTP'] = 1
        MBTI_COMPOSITION['ESFP'] = 1
        MBTI_COMPOSITION['ENFP'] = 1
        MBTI_COMPOSITION['ENTP'] = 1
    elif experiment == 'b1':
        MBTI_COMPOSITION['ISTJ'] = 1
        MBTI_COMPOSITION['INFJ'] = 1
        MBTI_COMPOSITION['ISTP'] = 1
        MBTI_COMPOSITION['INFP'] = 1
        MBTI_COMPOSITION['ENTP'] = 1
        MBTI_COMPOSITION['ESFP'] = 1
        MBTI_COMPOSITION['ENTJ'] = 1
        MBTI_COMPOSITION['ESFJ'] = 1
    elif experiment == 'b2':
        MBTI_COMPOSITION['ESTJ'] = 1
        MBTI_COMPOSITION['ENFJ'] = 1
        MBTI_COMPOSITION['ESTP'] = 1
        MBTI_COMPOSITION['ENFP'] = 1
        MBTI_COMPOSITION['INTP'] = 1
        MBTI_COMPOSITION['ISFP'] = 1
        MBTI_COMPOSITION['INTJ'] = 1
        MBTI_COMPOSITION['ISFJ'] = 1
    else:
        MBTI_COMPOSITION[experiment.upper()] = 16

    # calculate AGENT_COMPOSITION from MBTI_COMPOSITION
    AGENT_COMPOSITION = []
    MBTI_CODES = []
    for mbti_code, count in MBTI_COMPOSITION.items():
        if count > 0:
            MBTI_CODES.append(mbti_code)
    TEAMS_SIZE_LIST, members_team_index = assign_teams(MBTI_CODES)
    agent_index = 0
    for mbti_code, count in MBTI_COMPOSITION.items():
        while count > 0:
            AGENT_COMPOSITION.append({
                'mbti': mbti_code,
                'my_team': members_team_index[agent_index],
                'start_x_min': START_X_MIN,
                'start_x_max': START_X_MAX,
                'start_y_min': START_Y_MIN,
                'start_y_max': START_Y_MAX,
            })
            count = count - 1
            agent_index += 1
else:
    AGENT_COMPOSITION = [
        # {
        # 'mbti': 'INTJ',
        # 'start_x_min': 0, # int
        # 'start_x_max': 10, # int
        # 'start_y_min': 0, # int
        # 'start_y_max': 10, # int
        # },
        # {
        # 'mbti': 'ISTJ',
        # 'start_x_min': 90,
        # 'start_x_max': 100,
        # 'start_y_min': -20,
        # 'start_y_max': -10,
        # },
        # {
        # 'mbti': 'ISTJ',
        # 'start_x_min': -20,
        # 'start_x_max': 0,
        # 'start_y_min': 10,
        # 'start_y_max': 20,
        # },
        # {
        # 'mbti': 'ISTJ',
        # 'start_x_min': -100,
        # 'start_x_max': -90,
        # 'start_y_min': 70,
        # 'start_y_max': 80,
        # },
    ]
    MBTI_CODES = list(set(object['mbti'] for object in AGENT_COMPOSITION))  # get unique mbti code

NUM_AGENTS = len(AGENT_COMPOSITION)

DOMINANT_FUNCTION = {  # fixed mapping of mbti code to their dominant function
    'ISTJ': 'Si',
    'ISFJ': 'Si',
    'INFJ': 'Ni',
    'INTJ': 'Ni',
    'ISTP': 'Ti',
    'ISFP': 'Fi',
    'INFP': 'Fi',
    'INTP': 'Ti',
    'ESTP': 'Se',
    'ESFP': 'Se',
    'ENFP': 'Ne',
    'ENTP': 'Ne',
    'ESTJ': 'Te',
    'ESFJ': 'Fe',
    'ENFJ': 'Fe',
    'ENTJ': 'Te',
    'Si': 'Si',  # only investigating 8 functions (no auxiliary)
    'Ni': 'Ni',  # only investigating 8 functions (no auxiliary)
    'Ti': 'Ti',  # only investigating 8 functions (no auxiliary)
    'Fi': 'Fi',  # only investigating 8 functions (no auxiliary)
    'Se': 'Se',  # only investigating 8 functions (no auxiliary)
    'Ne': 'Ne',  # only investigating 8 functions (no auxiliary)
    'Te': 'Te',  # only investigating 8 functions (no auxiliary)
    'Fe': 'Fe',  # only investigating 8 functions (no auxiliary)
}

AUXILIARY_FUNCTION = {  # fixed mapping of mbti code to their auxiliary function
    'ISTJ': 'Te',
    'ISFJ': 'Fe',
    'INFJ': 'Fe',
    'INTJ': 'Te',
    'ISTP': 'Se',
    'ISFP': 'Se',
    'INFP': 'Ne',
    'INTP': 'Ne',
    'ESTP': 'Ti',
    'ESFP': 'Fi',
    'ENFP': 'Fi',
    'ENTP': 'Ti',
    'ESTJ': 'Si',
    'ESFJ': 'Si',
    'ENFJ': 'Ni',
    'ENTJ': 'Ni',
    'Si': 'XX',  # only investigating 8 functions (no auxiliary)
    'Ni': 'XX',  # only investigating 8 functions (no auxiliary)
    'Ti': 'XX',  # only investigating 8 functions (no auxiliary)
    'Fi': 'XX',  # only investigating 8 functions (no auxiliary)
    'Se': 'XX',  # only investigating 8 functions (no auxiliary)
    'Ne': 'XX',  # only investigating 8 functions (no auxiliary)
    'Te': 'XX',  # only investigating 8 functions (no auxiliary)
    'Fe': 'XX',  # only investigating 8 functions (no auxiliary)
}

MAX_NEIGHBOURS = 5 # neighbours are from same team
NUM_BEST_CANDIDATES = 3  # only the best 3 candidates are used to calculate acceleration

MIN_FLOAT = float(MIN)
MAX_FLOAT = float(MAX)


def check_parameters():
    if NUM_AGENTS <= 0:
        sys.exit('Error: Number of agents has to be one or more.')


def normalise(x):
    return (x - MIN_FLOAT) / (MAX_FLOAT - MIN_FLOAT)


def unnormalise(x):
    return (x * (MAX_FLOAT - MIN_FLOAT)) + MIN_FLOAT


# X input for gaussian predictor
testX = []
for i in range(MIN, MAX):
    i_norm = normalise(i)
    testX.append([i_norm])
testX = np.array(testX)

# globals
agents = []
group_best = []  # group's best fitness so far, recorded for each timestep

run_group_best_fitness = [[]for x in range(NUM_RUNS)]  # best fitness per run
run_group_best_fitness_timestep = [[] for x in range(NUM_RUNS)]  # best fitness timestep per run
run_group_best_fitness_agent = [[] for x in range(NUM_RUNS)] # best fitness agent per run
run_project_fitness = {} # fitness of the project per run if two teams are coherent
run_project_fail = {} # failed runs
current_group_best_fitness = [0] * NUM_TEAMS # highest sub task fitness at the current timestep for each team
current_group_best_agent = [0] * NUM_TEAMS # agent reached the highest sub task fitness at the current timestep

def creat_attend_list():
    attendees = OrderedDict()
    attendees['ISTJ'] = 0
    attendees['ISFJ'] = 0
    attendees['INFJ'] = 0
    attendees['INTJ'] = 0
    attendees['ISTP'] = 0
    attendees['ISFP'] = 0
    attendees['INFP'] = 0
    attendees['INTP'] = 0
    attendees['ESTP'] = 0
    attendees['ESFP'] = 0
    attendees['ENFP'] = 0
    attendees['ENTP'] = 0
    attendees['ESTJ'] = 0
    attendees['ESFJ'] = 0
    attendees['ENFJ'] = 0
    attendees['ENTJ'] = 0
    attendees['Si'] = 0
    attendees['Ni'] = 0
    attendees['Ti'] = 0
    attendees['Fi'] = 0
    attendees['Se'] = 0
    attendees['Ne'] = 0
    attendees['Te'] = 0
    attendees['Fe'] = 0
    return attendees

count_attendees = creat_attend_list()
record_attendees = {k: {} for k in range(NUM_RUNS)}
fail_attendees = creat_attend_list()


def project_function(x,y):
    """ the project objective function """
    return (((math.sqrt(math.pow(x, 2) + math.pow(y, 2))) * -1.0) + (141.421356237309512415697))/(141.421356237309512415697)

def function(x):
    """ the sub task objective function """
    return ((math.pow(x, 2) * -1.0) + 10000) / 10000


def distort(fitness):
    """ add noise to the perceived fitness """
    return fitness + random.uniform(-NOISE, NOISE)


# def timing(f):
#     """ calculate the timing of a function """
#     def wrap(*args):
#         time1 = time.time()
#         ret = f(*args)
#         time2 = time.time()
#         # print('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
#         print('%s function took %0.3f s' % (f.func_name, (time2-time1)))
#         return ret
#     return wrap


def get_nearby_points(agent):
    """ given x and y, calculate nearby four points """
    mu = 1
    sigma = 0.01  # top of curve is at 40
    diff = random.gauss(mu, sigma)

    team = agent['my_team']

    coor_forwards = copy.copy(agent['coordinate'])
    coor_backwards = copy.copy(agent['coordinate'])

    # diff = 1
    coor_forwards[team] += diff
    coor_backwards[team] -= diff

    return [
        {'x': coor_forwards[team],'coordinate': coor_forwards, 'fitness': distort(function(coor_forwards[team]))},
        {'x': coor_backwards[team],'coordinate': coor_backwards, 'fitness': distort(function(coor_backwards[team]))},
    ]


def get_neighbours_positions(run, agent):
    """ get the positions of an agent's neighbours """
    candidates = []
    for neighbour in agent['neighbours']:
        candidates.append({
            'x': neighbour['x'],
            'coordinate': copy.copy(neighbour['coordinate']),
            'fitness': neighbour['fitness'],
        })

    return candidates


def get_own_positions(run, agent):
    """ get all previously travelled points and nearby points of current point """
    return agent['path'] + get_nearby_points(agent)


def distinct_dict(l):
    """ given a dictionary, remove items with duplicated values """
    return [dict(t) for t in set([tuple(d.items()) for d in l])]


def normalise_predicted_fitness(x, min_pred, max_pred):
    return (x - min_pred) / float(max_pred - min_pred)


def get_prediction(timestep, candidates, agent, ax):
    """ apply intuitive function on candidates """
    trainX = []
    trainY = []
    team = agent['my_team']

    for item in (candidates):
        trainX.append([normalise(item['coordinate'][team])])
        trainY.append(item['fitness'])

    gpr = gaussian_process.GaussianProcessRegressor()
    gpr.fit(np.array(trainX), np.array(trainY))

    pred = gpr.predict(testX)

    z_min, z_max = pred.min(), pred.max()

    # if agent['mbti'] == 'INFP' and timestep == 11:
    #     print(trainX, trainY, agent['mbti'])
    #     print(testX, pred.reshape(MAX - MIN), agent['mbti'])


    if PLOT and (timestep in PLOT_TIMESTEPS):
        plot_num = PLOT_TIMESTEPS.index(timestep)
        pred_reshaped = pred.reshape(MAX - MIN)  # turn into 1D matrix
        ax[plot_num, agent['id']].plot(testX, pred_reshaped, '-')

    # sort pred values in order to select the best
    permutation = pred.argsort()  # produces the permutation required to sort pred in increasing order
    unsorted_testX = testX
    sorted_testX = unsorted_testX[permutation][
                   -NUM_BEST_CANDIDATES:]  # sort the corresponding x,y values and take the last NUM_BEST_CANDIDATES items (best fitness)
    pred_sorted = pred[permutation][
                  -NUM_BEST_CANDIDATES:]  # sort using the same permutation and take the last NUM_BEST_CANDIDATES items (best fitness)

    new_candidates = []

    for i, item in enumerate(sorted_testX):
        unnormalised_x = unnormalise(item[0])
        new_candidates.append({
            'x': unnormalised_x,
            'fitness': normalise_predicted_fitness(pred_sorted[i], z_min, z_max),
            # 'fitness': pred_sorted[i],
        })

    new_candidates.reverse()  # for consistency, reverse candidates so they are sorted by decreasing fitness
    return new_candidates


def initialise_subplot(subplot):
    """ specify how the subplots should look """

    subplot.axis([MIN, MAX, 0, 1])
    subplot.set_xticks([], [])
    subplot.set_yticks([], [])
    subplot.axhline(0, linewidth=0.3, color='grey')
    subplot.axvline(0, linewidth=0.3, color='grey')

    # subplot.axis('off') # no axis
    subplot.spines['bottom'].set_color('grey')
    subplot.spines['bottom'].set_linewidth(0.5)
    subplot.spines['top'].set_color('grey')
    subplot.spines['top'].set_linewidth(0.5)
    subplot.spines['right'].set_color('grey')
    subplot.spines['right'].set_linewidth(0.5)
    subplot.spines['left'].set_color('grey')
    subplot.spines['left'].set_linewidth(0.5)


def get_candidates(run, agent, ax, timestep):
    """ get candidates for perception function """
    candidates = []
    if PLOT and (timestep in PLOT_TIMESTEPS):
        plot_num = PLOT_TIMESTEPS.index(timestep)
        # title = '#' + str(agent['id']) + ', ' + agent['mbti']
        title = agent['mbti']
        if plot_num == 0:  # if this is the first plot row
            ax[plot_num, agent['id']].set_title(title, fontsize=5)
        initialise_subplot(ax[plot_num, agent['id']])

    # perception (S or N)
    if (agent['dominant'] == 'Si') or (agent['auxiliary'] == 'Si') or (agent['dominant'] == 'Ni') or (
            agent['auxiliary'] == 'Ni'):  # introverted
        candidates = get_own_positions(run, agent)
    elif (agent['dominant'] == 'Se') or (agent['auxiliary'] == 'Se') or (agent['dominant'] == 'Ne') or (
            agent['auxiliary'] == 'Ne'):  # extraverted
        candidates = get_neighbours_positions(run, agent)


    if PLOT and (timestep in PLOT_TIMESTEPS):
        if (agent['dominant'][0] == 'S') or (agent['auxiliary'][0] == 'S'):
            plot_num = PLOT_TIMESTEPS.index(timestep)
            plot_x = []
            plot_z = []
            for candidate in candidates:
                plot_x.append(candidate['x'])
                plot_z.append(candidate['fitness'])
            # ax[plot_num, agent['id']].scatter(plot_x, plot_z, cmap=CMAP, marker='o', s=0.4, vmin=-1, vmax=1)
            ax[plot_num, agent['id']].scatter(plot_x, plot_z, marker='o', s=0.4, vmin=-1, vmax=1)

    # for intuitives: apply gaussian function on the "sensed" candidates
    if (agent['dominant'][0] == 'N') or (agent['auxiliary'][0] == 'N'):
        candidates = get_prediction(timestep, candidates, agent, ax)  # returns top NUM_BEST_CANDIDATES
    elif (agent['dominant'][0] == 'S') or (
            agent['auxiliary'][0] == 'S'):  # for sensing: sort candidates and take the top NUM_BEST_CANDIDATES
        candidates = sorted(candidates, key=lambda k: k['fitness'], reverse=True)[:NUM_BEST_CANDIDATES]

    return candidates


def get_velocity_based_on_neighbours_velocities(run, agent):
    velocity_x = 0.0

    for neighbour in agent['neighbours']:
        velocity_x += neighbour['velocity']['x']

    # get the average velocity
    _len = float(len(agent['neighbours']))
    velocity_x = velocity_x / _len

    return velocity_x


def get_velocity_based_on_neighbours_personal_best(run, agent):
    if len(agent['neighbours']) > 0:
        others_best = {}
        for count, neighbour in enumerate(agent['neighbours']):
            if count == 0:  # first neighbour
                others_best = {'x': neighbour['personal_best']['x'],
                               'fitness': neighbour['personal_best']['fitness']}
            else:
                if neighbour['personal_best']['fitness'] > others_best['fitness']:
                    others_best = {'x': neighbour['personal_best']['x'],
                                   'fitness': neighbour['personal_best']['fitness']}

        team = agent['my_team']
        velocity_x = (others_best['x'] - agent['coordinate'][team])
    else:
        velocity_x = 0.0
    return velocity_x


def get_centroid(points):
    """ calculate the centroid of a list of points """
    x_coords = [p for p in points]
    _len = float(len(points))
    centroid_x = sum(x_coords) / _len
    return {'x': centroid_x}


def get_judging_acceleration(run, agent):
    """ get acceleration for judging function """
    major_weight = 0.8
    minor_weight = 0.2
    team = agent['my_team']

    if (agent['dominant'] == 'Fi') or (agent['auxiliary'] == 'Fi'):
        # get affected by neighbours centroid, a little by their personal best
        points = []
        for neighbour in agent['neighbours']:
            points.append((neighbour['x']))
        neighbours_centroid = get_centroid(points)
        acceleration_x = (major_weight * (neighbours_centroid['x'] - agent['coordinate'][team])) + (
                minor_weight * (agent['personal_best']['x'] - agent['coordinate'][team]))
    elif (agent['dominant'] == 'Fe') or (agent['auxiliary'] == 'Fe'):
        # get affected mainly by neighbour's velocity, a little by their neighbour's personal best
        velocity_acceleration_x = get_velocity_based_on_neighbours_velocities(run, agent)
        personal_best_acceleration_x = get_velocity_based_on_neighbours_personal_best(run, agent)
        acceleration_x = (major_weight * velocity_acceleration_x) + (minor_weight * personal_best_acceleration_x)

    elif (agent['dominant'] == 'Ti') or (agent['auxiliary'] == 'Ti'):  # introverted
        # get pulled towards own personal best

        mu = 0
        sigma = 2

        acceleration_x = (agent['personal_best']['x'] - agent['coordinate'][team]) + random.uniform(-2.0,
                                                                                     2.0)  # random.gauss(mu, sigma) # add random to simulate exploration to understand
    elif (agent['dominant'] == 'Te') or (agent['auxiliary'] == 'Te'):  # extraverted
        # get pulled towards neighbour's best personal best
        acceleration_x = get_velocity_based_on_neighbours_personal_best(run, agent)

    else:  # no judging function
        acceleration_x = 0.0

    return acceleration_x


def get_perception_acceleration(run, agent, ax, timestep):
    """ get acceleration for perception function """
    agent['old_candidates'] = agent[
        'new_candidates']  # new_candidates from previous run is the old_candidates for this run

    agent['new_candidates'] = get_candidates(run, agent, ax, timestep)

    team = agent['my_team']

    all_candidates = agent['old_candidates'] + agent['new_candidates']
    sorted_candidates = sorted(list(all_candidates), key=lambda k: k['fitness'],
                               reverse=True)  # sort all candidates by decreasing fitness

    best_candidate_weight = 0.5
    second_best_candidate_weight = 0.3
    third_best_candidate_weight = 0.2

    acceleration_perception_x = 0.0

    try:
        best = sorted_candidates[0]
        acceleration_perception_x += (best_candidate_weight * (best['x'] - agent['coordinate'][team]))
    except:
        pass
    try:
        second_best = sorted_candidates[1]
        acceleration_perception_x += (second_best_candidate_weight * (second_best['x'] - agent['coordinate'][team]))
    except:
        pass
    try:
        third_best = sorted_candidates[2]
        acceleration_perception_x += (third_best_candidate_weight * (third_best['x'] - agent['coordinate'][team]))
    except:
        pass

    return acceleration_perception_x


def clamp_velocity(velocity):
    """ clamp velocity """
    if velocity > VELOCITY_CAP:
        velocity = VELOCITY_CAP
    if velocity < -VELOCITY_CAP:
        velocity = -VELOCITY_CAP
    return velocity


def new_clamp_velocity(velocity_x):
    velocity_length = math.pow(velocity_x, 2)

    if (velocity_length > VELOCITY_CAP_SQUARE):
        k = velocity_x
        new_velocity_x = math.sqrt(VELOCITY_CAP_SQUARE / (k * k + 1.0))
        if (velocity_x * new_velocity_x < 0):
            new_velocity_x = new_velocity_x * -1.0

        velocity_x = new_velocity_x

    return velocity_x


def new_clamp_acceleration(agent, acceleration_perception_x, acceleration_judging_x):
    # dampen the effect of auxiliary function
    # print acceleration_perception_x, acceleration_perception_y, acceleration_judging_x, acceleration_judging_y
    if (agent['dominant'][0] == 'N') or (agent['dominant'][0] == 'S'):
        # this agent has dominant perception
        dominant_x = acceleration_perception_x
        auxiliary_x = acceleration_judging_x
    elif (agent['dominant'][0] == 'F') or (agent['dominant'][0] == 'T'):
        # this agent has dominant judging
        dominant_x = acceleration_judging_x
        auxiliary_x = acceleration_perception_x

    dominant_length = math.pow(dominant_x, 2)
    auxiliary_length = math.pow(auxiliary_x, 2)
    max_auxiliary_length = dominant_length / 2.0
    if (auxiliary_length > max_auxiliary_length):  # to avoid divide by zero problem
        k = auxiliary_x
        new_auxiliary_x = math.sqrt(max_auxiliary_length / (k * k + 1.0))

        # fix the signs
        if (auxiliary_x * new_auxiliary_x < 0):
            new_auxiliary_x = new_auxiliary_x * -1.0

        auxiliary_x = new_auxiliary_x

    return dominant_x, auxiliary_x

def get_guess(y):
    """ update the guess of other team's sub task location"""
    return random.gauss(y, 0.01)

def update_agent(run, timestep, agent, ax):
    """ update agent position """
    # if timestep == meeting_point:

    team = agent['my_team']

    acceleration_perception_x = get_perception_acceleration(run, agent, ax, timestep)
    acceleration_judging_x = get_judging_acceleration(run, agent)

    dominant_x, auxiliary_x = new_clamp_acceleration(agent, acceleration_perception_x, acceleration_judging_x)

    # update velocity (clamp if it is too high)
    agent['velocity']['x'] = new_clamp_velocity(
        agent['velocity']['x'] + dominant_x + auxiliary_x)

    #update x
    agent['x'] = agent['x'] + agent['velocity']['x']

    # update coordinate
    for ind in range(NUM_TEAMS):
        if ind == team:
            agent['coordinate'][ind] = agent['coordinate'][ind] + agent['velocity']['x']
        else:
            agent['coordinate'][ind] = get_guess(agent['coordinate'][ind])


    # update fitness
    agent['real_fitness'] = function(agent['coordinate'][team])
    agent['fitness'] = distort(agent['real_fitness'])

    # if the fitness in this timestep is better than personal best
    if agent['fitness'] > agent['personal_best']['fitness']:
        agent['personal_best'] = {'x': agent['coordinate'][team], 'fitness': agent['fitness'], 'timestep': timestep}

    if agent['real_fitness'] > agent['real_personal_best']['fitness']:
        agent['real_personal_best'] = {'x': agent['coordinate'][team], 'fitness': agent['real_fitness'],
                                       'timestep': timestep}

    # add to path
    agent['path'].append({
        'x': agent['coordinate'][team],
        'coordinate': copy.copy(agent['coordinate']),
        'fitness': agent['fitness'],
    })

    # if agent['id'] == 0 :
    #     print('in update',agent['path'])

    agent['neighbours'] = []  # clear neighbours for next run

    if PLOT and (timestep in PLOT_TIMESTEPS):
        plot_num = PLOT_TIMESTEPS.index(timestep)
        ax[plot_num, agent['id']].plot(agent['coordinate'][team],agent['fitness'], color='black', marker='o', markersize=1.2)
        ax[plot_num, agent['id']].plot(agent['coordinate'][team],agent['fitness'], color='white', marker='o', markersize=0.4)


def initialise_path_subplot(subplot):
    """ specify how the subplots should look """

    # subplot.axis([MIN, MAX, MIN, MAX])
    subplot.axis([MIN - AXIS_EXTRA, MAX + AXIS_EXTRA, MIN - AXIS_EXTRA, MAX + AXIS_EXTRA])
    subplot.set_xticks([], [])
    subplot.set_yticks([], [])
    subplot.axhline(0, linewidth=0.3, color='grey')
    subplot.axvline(0, linewidth=0.3, color='grey')

    # subplot.axis('off') # no axis
    subplot.spines['bottom'].set_color('grey')
    subplot.spines['bottom'].set_linewidth(0.5)
    subplot.spines['top'].set_color('grey')
    subplot.spines['top'].set_linewidth(0.5)
    subplot.spines['right'].set_color('grey')
    subplot.spines['right'].set_linewidth(0.5)
    subplot.spines['left'].set_color('grey')
    subplot.spines['left'].set_linewidth(0.5)

def initialise_team_subplot(subplot):
    subplot.axis([MIN - AXIS_EXTRA, MAX + AXIS_EXTRA, MIN - AXIS_EXTRA, MAX + AXIS_EXTRA])
    subplot.set_xticks([], [])
    subplot.set_yticks([], [])
    subplot.axhline(0, linewidth=0.3, color='grey')
    subplot.axvline(0, linewidth=0.3, color='grey')

    # subplot.axis('off') # no axis
    subplot.spines['bottom'].set_color('grey')
    subplot.spines['bottom'].set_linewidth(0.5)
    subplot.spines['top'].set_color('grey')
    subplot.spines['top'].set_linewidth(0.5)
    subplot.spines['right'].set_color('grey')
    subplot.spines['right'].set_linewidth(0.5)
    subplot.spines['left'].set_color('grey')
    subplot.spines['left'].set_linewidth(0.5)


def draw_paths():
    """ draw the movement of each agent from start to end """
    for run in range(NUM_RUNS):
        for i, agent in enumerate(agents[run]):
            # if agent['my_team'] == team:
            # if agent['dominant'][0] == 'F' or agent['dominant'] == 'Te':
            xpath = [d['coordinate'][0] for d in agent['path']]
            ypath = [d['coordinate'][1] for d in agent['path']]
            plt.plot(xpath, ypath, linewidth=0.4)  # draw path
            plt.plot(xpath, ypath, marker='x', markersize=1, linewidth=0.4)
            plt.axis([MIN - AXIS_EXTRA, MAX + AXIS_EXTRA, MIN - AXIS_EXTRA, MAX + AXIS_EXTRA])
            plt.annotate(agent['mbti'] + str(agent['id']) + str(agent['my_team']) + 's', xy=(xpath[0], ypath[0]), size=5,
                         xytext=(xpath[0], ypath[0]))
            plt.annotate(agent['mbti'] + str(agent['id']) + str(agent['my_team']) + 'e', xy=(agent['coordinate'][0], agent['coordinate'][1]), size=5,
                         xytext=(agent['coordinate'][0], agent['coordinate'][1]))
        # plt.hlines(0, -100, 100)
        # plt.vlines(0, -100, 100)
        plt.show()

def get_distance(p1, p2):
    """ get the euclidean distance between two points """
    return math.sqrt( ((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2) )

def check_coherence(p1, p2):
    """ get the euclidean distance between two points """
    return abs(np.array(p1) - np.array(p2))


def calculate_neighbours(run):
    """ get neighbours and their details for each agent """
    for i in range(NUM_AGENTS):

        for j in range(NUM_AGENTS):
            if j > i:  # only calculate the ones that are not already calculated (half of the matrix)
                if agents[run][i]['my_team'] == agents[run][j]['my_team']:
                    team = agents[run][i]['my_team']
                    distance = get_distance(agents[run][i]['coordinate'], agents[run][i]['coordinate'])

                    agents[run][i]['neighbours'].append(
                        {'id': j, 'distance': distance, 'velocity': agents[run][j]['velocity'],
                         'x': agents[run][j]['coordinate'][team], 'coordinate': agents[run][j]['coordinate'],
                         'fitness': agents[run][j]['fitness'], 'personal_best': agents[run][j]['personal_best']})
                    agents[run][j]['neighbours'].append(
                        {'id': i, 'distance': distance, 'velocity': agents[run][i]['velocity'],
                         'x': agents[run][i]['coordinate'][team], 'coordinate': agents[run][i]['coordinate'],
                         'fitness': agents[run][i]['fitness'], 'personal_best': agents[run][i]['personal_best']})

    # sort neighbours in increasing order of distance
    for i in range(NUM_AGENTS):
        sorted_neighbour = sorted(agents[run][i]['neighbours'], key=lambda k: k['distance'])
        agents[run][i]['neighbours'] = sorted_neighbour[
                                       :MAX_NEIGHBOURS]  # only consider the nearest MAX_NEIGHBOURS agents as neighbours


def mean_stddev(data):
    """ calculate the sample standard deviation. specify ddof=0 to compute the population standard deviation. """
    ddof = 1.0
    n = len(data)
    mean = sum(data) / float(n)
    if n < 2.0:  # make it still work for n < 2
        std = 0.0
    else:
        ss = sum((x - mean) ** 2 for x in data)
        pvar = ss / (n - ddof)
        std = pvar ** 0.5
    return {'mean': mean, 'std': std}


def analyse_results(seed):
    print("\n\nSeed: " + str(seed))
    print(sys.argv)
    for key, value in MBTI_COMPOSITION.items():
        if value > 0:
            print(key + ': ' + str(value))
    """ analyse results for all runs, mbti code, and agents """
    results = []

    fitness_agent_average = {}
    best_fitness_agent_average = {}
    best_fitness_timestep_agent_average = {}
    for i in range(NUM_AGENTS):
        fitness_agent_average.update({i: []})
        best_fitness_agent_average.update({i: []})
        best_fitness_timestep_agent_average.update({i: []})

    fitness_mbti_average = {}
    best_fitness_mbti_average = {}
    best_fitness_timestep_mbti_average = {}

    for mbti_code in MBTI_CODES:
        fitness_mbti_average.update({mbti_code: []})
        best_fitness_mbti_average.update({mbti_code: []})
        best_fitness_timestep_mbti_average.update({mbti_code: []})

    for run in range(NUM_RUNS):
        results.append([])

        results[run] = {}
        for mbti_code in MBTI_CODES:
            results[run].update(
                {mbti_code: {'fitness': 0.0, 'best_fitness': 0.0, 'best_fitness_timestep': 0.0, 'count': 0}})

        for agent in agents[run]:
            results[run][agent['mbti']]['fitness'] += agent['real_fitness']
            results[run][agent['mbti']]['best_fitness'] += agent['real_personal_best']['fitness']
            results[run][agent['mbti']]['best_fitness_timestep'] += agent['real_personal_best']['timestep']
            results[run][agent['mbti']]['count'] += 1

            fitness_agent_average[agent['id']].append(agent['real_fitness'])
            best_fitness_agent_average[agent['id']].append(agent['real_personal_best']['fitness'])
            best_fitness_timestep_agent_average[agent['id']].append(agent['real_personal_best']['timestep'])

        for mbti_code in MBTI_CODES:
            fitness_mbti_average[mbti_code].append(
                results[run][mbti_code]['fitness'] / results[run][mbti_code]['count'])
            best_fitness_mbti_average[mbti_code].append(
                results[run][mbti_code]['best_fitness'] / results[run][mbti_code]['count'])
            best_fitness_timestep_mbti_average[mbti_code].append(
                results[run][mbti_code]['best_fitness_timestep'] / results[run][mbti_code]['count'])

    print('\n\n' + str(
        seed) + '\t' + 'Final fitness avg' + '\t' + 'Final fitness std' + '\t' + 'Best fitness avg' + '\t' + 'Best fitness std' + '\t' + 'Best fitness timestep avg' + '\t' + 'Best fitness timestep std')
    for mbti_code in MBTI_CODES:
        final_fitness_result = mean_stddev(fitness_mbti_average[mbti_code])
        best_fitness_result = mean_stddev(best_fitness_mbti_average[mbti_code])
        best_fitness_timestamp_result = mean_stddev(best_fitness_timestep_mbti_average[mbti_code])
        print(
            mbti_code + '\t' + str(final_fitness_result['mean']) + '\t' + str(final_fitness_result['std']) + '\t' + str(
                best_fitness_result['mean']) + '\t' + str(best_fitness_result['std']) + '\t' + str(
                best_fitness_timestamp_result['mean']) + '\t' + str(best_fitness_timestamp_result['std']))

    group_fitness_results = [0] * NUM_TEAMS

    if len(MBTI_CODES) == 1:
        title = MBTI_CODES[0]
    else:
        title = "Group"

    run_group_best_fitness_result = []
    run_group_best_fitness_timestep_result = []
    temp_fitness = []
    temp_timestep = []
    for team in range(NUM_TEAMS):
        for run in range(NUM_RUNS):
            temp_fitness.append(run_group_best_fitness[run][team])
            temp_timestep.append(run_group_best_fitness_timestep[run][team])

        run_group_best_fitness_result.append(mean_stddev(temp_fitness)) # result over all runs of each team
        run_group_best_fitness_timestep_result.append(mean_stddev(temp_timestep))

    all_project_fitness = []
    for key in run_project_fitness:
        all_project_fitness.append(run_project_fitness[key]['project_fitness'])
    if len(all_project_fitness) > 0:
        analyse_project = mean_stddev(all_project_fitness)
        print('Average project fitness'+'\t'+str(analyse_project['mean']) + 'STD' + '\t' + str(analyse_project['std']))

    print('\n\n' + 'Number of failed run' + '\t' + str(len(run_project_fail)))

    print('Failed run' + '\t' + 'Fail criteria' + '\t'
          + 'Team 1 coordinate' + '\t' + 'Team 2 coordinate')
    for key in run_project_fail:
        print(str(key) + '\t' + str(FAIL_CRITERIA)+ '\t' + str(run_project_fail[key][0])+ '\t' + str(run_project_fail[key][1]))

    print('\n\n' + 'MBTI' + '\t' + 'Num of failed meetings' + '\t' + 'Num of meetings')
    for key,value in fail_attendees.items():
        print(str(key) + '\t' + str(value) + '\t' + str(count_attendees[key]))

    print('just for plot',fail_attendees, count_attendees)


    print("\n\nSeed: " + str(seed))

def record_results(seed):
    # CSV file header ["Meeting type", "Number of runs", "Number of timesetps","Fail Criteria", "Number of failed runs", "Average project fitness"]
    if ALL_ATTEND_MEETING:
        meeting_type = 'All'
    else:
        meeting_type = 'Best'
    num_failed_runs = len(run_project_fail)
    average_project_fitness = 0
    for key in run_project_fitness:
        average_project_fitness += run_project_fitness[key]['project_fitness']
    row = [meeting_type,NUM_RUNS, NUM_TIMESTEPS, FAIL_CRITERIA, num_failed_runs, average_project_fitness]
    with open('parallel_result.csv','a') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def initialise_agents(run):
    global current_group_best_agent
    global current_group_best_fitness
    # initialise agents in timestep 0
    group_best_fitness = [-50000.00] * NUM_TEAMS  # set it to something very low
    group_best_x = [0.0] * NUM_TEAMS
    group_best_agent = [0] * NUM_TEAMS
    # current_group_best_fitness = [0] * NUM_TEAMS
    # current_group_best_agent = [0] * NUM_TEAMS
    # team_average = [0]*NUM_TEAMS
    # create agents
    for n in range(NUM_AGENTS):
        # random initial position
        coordinate = [0] * NUM_TEAMS
        coordinate[0] = random.uniform(AGENT_COMPOSITION[n]['start_x_min'], AGENT_COMPOSITION[n]['start_x_max'])
        coordinate[1] = random.uniform(AGENT_COMPOSITION[n]['start_y_min'], AGENT_COMPOSITION[n]['start_y_max'])

        mbti = AGENT_COMPOSITION[n]['mbti']

        team = AGENT_COMPOSITION[n]['my_team']

        agent = {
            'run': run,
            'id': n,
            'mbti': mbti,
            'dominant': DOMINANT_FUNCTION[mbti],
            'auxiliary': AUXILIARY_FUNCTION[mbti],
            'x': coordinate[team],
            'my_team': team,
            'coordinate': coordinate,
        }

        agents[run].append(agent)

        # team_average[team] += x/TEAMS_SIZE_LIST[team]

    for n in range(NUM_AGENTS):
        team = AGENT_COMPOSITION[n]['my_team']
        coordinate = agents[run][n]['coordinate']
        # other_teams = [a for a in TEAMS_SIZE_LIST if a != team]
        # my_team_average = team_average[my_team]
        real_fitness = function(coordinate[team])
        fitness = distort(real_fitness)
        agents[run][n].update({
            # 'team_average': team_average[my_team],
            'fitness': fitness,
            'real_fitness': real_fitness,
            'velocity': {'x':random.uniform(-1.0, 1.0)},  # start with random velocity
            'personal_best': {'x': coordinate[team], 'fitness': fitness, 'timestep': 0},
            'real_personal_best': {'x': coordinate[team], 'fitness': real_fitness, 'timestep': 0},
            'old_candidates': [],
            'new_candidates': [],
            'path': [{'x': coordinate[team],'coordinate':copy.copy(coordinate), 'fitness': fitness}],
            'neighbours': [],
        })


        if real_fitness > group_best_fitness[team]:
            group_best_fitness[team] = real_fitness
            group_best_x[team] = coordinate[team]
            group_best_agent[team] = n
        if real_fitness > current_group_best_fitness[team]:
            current_group_best_fitness[team] = real_fitness
            current_group_best_agent[team] = n
    run_group_best_fitness[run] = group_best_fitness
    run_group_best_fitness_agent[run] = group_best_agent

    for team in range(NUM_TEAMS):
        group_best[run][team].append({'x': group_best_x[team], 'fitness': group_best_fitness[team]})
        run_group_best_fitness_timestep[run].append(0) # timestep = 0

    return group_best_x, group_best_fitness, group_best_agent


# def check_team_difference(meeting_document):
#     for i in range(NUM_TEAMS):
#         for j in range(NUM_TEAMS):
#             if j > i:
#                 difference = abs(meeting_document[i] - meeting_document[j])
#                 if difference >= FAIL_CRITERIA:
#                     print('Team1 coordinate:',meeting_document[i],'Team2 coordinate:',meeting_document[j])
#                     print('FAIL_CRITERIA:', FAIL_CRITERIA,'Fitness difference:',difference)
#                     sys.exit('Task failed: Outcomes of teams are too different')
#                 else:
#                     pass

def meeting(run):
    global current_group_best_agent
    global record_attendees
    global record_attendees
    update_coordinate = []
    if ALL_ATTEND_MEETING:
        for team in range(NUM_TEAMS):
            temp = 0
            num = 0
            for n in range(NUM_AGENTS):
                if agents[run][n]['my_team'] == team:
                    temp += agents[run][n]['x']
                    num += 1
            update_coordinate.append(temp/num)
    else:
        for team in range(NUM_TEAMS):
            team_agent_attending = current_group_best_agent[team]
            agent_type = agents[run][team_agent_attending]['mbti']
            dominant_type = agents[run][team_agent_attending]['dominant']
            count_attendees[agent_type] += 1
            count_attendees[dominant_type] += 1
            record_attendees[run][team] = {'mbti': agent_type, 'dominant': dominant_type}
            update_coordinate.append(agents[run][team_agent_attending]['coordinate'][team])

    for agent in agents[run]:
        team = agent['my_team']
        for i in range(NUM_TEAMS):
            if i != team:
                agent['coordinate'][i] = copy.copy(update_coordinate[i])
        agent['path'].append({'x': copy.copy(agent['x']), 'coordinate': copy.copy(agent['coordinate']),
                              'fitness': copy.copy(agent['fitness'])})

def final_meeting(run):
    global current_group_best_agent
    global current_group_best_fitness
    global fail_attendees
    meeting_document = []
    # final_best_subtask_fitness = []
    for team in range(NUM_TEAMS):
        team_agent_attending = current_group_best_agent[team]
        meeting_document.append(agents[run][team_agent_attending]['coordinate'])
        # final_best_subtask_fitness.append(agents[run][team_agent_attending]['fitness'])
    for i in range(NUM_TEAMS):
        for j in range(NUM_TEAMS):
            if j > i:
                if any(check_coherence(meeting_document[i], meeting_document[j]) > FAIL_CRITERIA):
                    # if there are any two teams fail to be coherent, the project fails
                    run_project_fail.update({run:[meeting_document[i],meeting_document[j],record_attendees[run][0],record_attendees[run][1]]})
                    for team in range(NUM_TEAMS):
                        agent_type = record_attendees[run][team]['mbti']
                        agent_dom = record_attendees[run][team]['dominant']
                        fail_attendees[agent_type] += 1
                        fail_attendees[agent_dom] += 1
                    continue
                else:
                    project_fitness = project_function(meeting_document[0][0], meeting_document[1][1])
                    run_project_fitness.update({run: {'project_fitness': project_fitness, 'coordinate':(meeting_document[0][0], meeting_document[1][1]),
                                                      'Subtask_1_fitness': copy.copy(current_group_best_fitness[i]),
                                                      'Subtask_2_fitness': copy.copy(current_group_best_fitness[j])}})

# @timing

def main():
    """ main function """
    if RANDOM_SEED < 0:
        seed = int(time.time())  # use current time as random seed
    else:
        seed = RANDOM_SEED

    random.seed(seed)

    for run in range(NUM_RUNS):
        if run % 10 == 0:
            print("Run: " + str(run))

        agents.append([])  # add new run
        group_best.append([[] for y in range(NUM_TEAMS)])  # add new run

        if PLOT:
            fig, ax = plt.subplots(len(PLOT_TIMESTEPS), max(NUM_AGENTS, 2), sharex=True,
                                   sharey=True)  # subplot doesn't work when there is one agent
        else:
            ax = None

        # initialise agents for timestep 0
        group_best_x, group_best_fitness, group_best_agent = initialise_agents(run)

        for t in range(START_TIMESTEP, MAX_TIMESTEP):  # update agents for timestep = 1 to timestep < MAX_TIMESTEP
            global current_group_best_fitness
            global current_group_best_agent
            if t in MEETING_POINT[:-1]:
                meeting(run)
            elif t == MEETING_POINT[-1]:
                # print(agents[0][0]['path'])
                final_meeting(run)
            else:
                calculate_neighbours(run)
                current_group_best_fitness = [0] * NUM_TEAMS
                current_group_best_agent = [0] * NUM_TEAMS
                for agent in agents[run]:
                    update_agent(run, t, agent, ax)
                    team = agent['my_team']
                    if agent['real_fitness'] > group_best_fitness[team]:
                        group_best_fitness[team] = agent['real_fitness']
                        group_best_x[team] = agent['coordinate'][team]
                        group_best_agent[team] = agent['id']

                        # update run group best
                        run_group_best_fitness[run][team] = group_best_fitness[team]
                        run_group_best_fitness_timestep[run][team] = t
                        run_group_best_fitness_agent[run][team] = agent['id']

                    if agent['real_fitness'] > current_group_best_fitness[team]:
                        current_group_best_fitness[team] = agent['real_fitness']
                        current_group_best_agent[team] = agent['id']

            for team in range(NUM_TEAMS):
                group_best[run][team].append({'x': copy.copy(group_best_x[team]), 'fitness': copy.copy(group_best_fitness[team])})

        if PLOT:
            plt.show()

    if ANALYSE:
        analyse_results(seed)
    if RECORD:
        record_results(seed)
    if PATH_PLOT:
        # for team in range(NUM_TEAMS):
        draw_paths()


if __name__ == "__main__":
    # check_parameters()
    main()
