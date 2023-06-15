# ASSIGNMENT 4
import numpy as np
import random as rd
import math
from matplotlib import pyplot as plt

rd.seed(13)


def initpop(pop_size, x_max, x_min, v_max):

    positions = np.random.uniform(low=x_min, high=x_max, size=(pop_size, 2))
    velocities = np.random.uniform(
        low=v_max[0], high=v_max[1], size=(pop_size, 2))
    return positions, velocities


def fitness(positions, pop_size):
    fit = []
    for i in range(pop_size):  # calculate the fitness for each particle using fitness function
        fit.append(math.sin(2*positions[i][0]-0.5 * math.pi) +
                   3 * math.cos(positions[i][1]) + 0.5*positions[i][0])

    return fit


def update_gbest(positions, fit, gbest_fit, gbest_position):
    max_fit = max(fit)
    max_fit_indx = np.argmax(fit, 0, None)
    if max_fit > gbest_fit:
        gbest_position = positions[max_fit_indx]
        gbest_fit = max_fit
    return gbest_position, gbest_fit


def update_pbest(new_positions, previous_pbest, new_fitness, particles_bestFit, pop_size):
    new_pbest = previous_pbest.copy()
    for i in range(pop_size):
        if new_fitness[i] > particles_bestFit[i]:
            particles_bestFit[i] = new_fitness[i]
            new_pbest[i] = new_positions[i]
    return new_pbest, particles_bestFit


def update_velocity(pop, pop_size, velocity, pbest, gbest,  c1=2, c2=2):

    new_velocity = np.zeros_like(velocity)
    r1 = rd.uniform(0, 1)
    r2 = rd.uniform(0, 1)
    for i in range(pop_size):
        for j in range(2):
            new_velocity[i][j] = velocity[i][j] + c1*r1 * \
                (pbest[i][j]-pop[i][j]) + c2*r2*(gbest[j]-pop[i][j])

    return new_velocity


def update_position(position, paticles_num, velocity, x_min, x_max):
    new_position = np.zeros_like(position)
    for i in range(paticles_num):
        for j in range(2):  # when j = 0  -> update X1  ,  when j = 1  -> update X12
            new_position[i][j] = position[i][j] + velocity[i][j]
            # to make sure that the new values of X1,X2 does not break the limits
            if new_position[i][j] > x_max[j]:
                new_position[i][j] = x_max[j]
            if new_position[i][j] < x_min[j]:
                new_position[i][j] = x_min[j]
    return new_position


def PSO(pop_size, position_min, position_max, v_max, generation):

    positions, velocities = initpop(
        pop_size, position_max, position_min, v_max)

    pbest_position = positions.copy()
    particles_fitness = fitness(positions, pop_size)
    particles_bestFit = particles_fitness.copy()
    gbest_position, gbest_fit = update_gbest(
        positions, particles_fitness, -1000000, [0, 0])

    for j in range(generation):
        velocities = update_velocity(
            positions, pop_size, velocities, pbest_position, gbest_position, 1.7, 1.7)
        positions = update_position(
            positions, pop_size, velocities, position_min, position_max)
        particles_fitness = fitness(
            positions, pop_size)
        gbest_position, gbest_fit = update_gbest(
            positions, particles_fitness, gbest_fit, gbest_position)
        pbest_position, particles_bestFit = update_pbest(
            positions, pbest_position, particles_fitness, particles_bestFit, pop_size)

    # Print the results
    print('Global Best Position: ', gbest_position, "\n")
    print('Best Fitness Value: ', max(particles_bestFit), "\n")
    print('Average Particle Best Fitness Value: ',
          np.average(particles_bestFit), "\n")
    # to plot the movement of the particles
    xpoints, ypoints = [], []
    for i in range(pop_size):
        xpoints.append(positions[i][0])
        ypoints.append(positions[i][1])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.plot(xpoints, ypoints)
    plt.title("the movement of the particles")
    plt.show()


PSO(50, [-2, -2], [3, 1], [0.1, 0.1], 200)
