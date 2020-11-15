import numpy as np
from time import time
import random
import matplotlib.pyplot as plt
from ga import GeneticAlgorithm, fitness_function


def simulate(params, target_sentence="Hello World"):
    np.random.seed(123)
    random.seed(123)

    MAX_GEN = params['max_gen']  # termination
    MAX_SUCCESS = params['max_success']  # termination
    POPULATION_SIZE = params['population_size']
    NUM_ATTRIBUTES = len(target_sentence)
    MUTATION_PROB = params['mutation_prob']
    CROSSOVER_PROB = params['crossover_prob']

    GA = GeneticAlgorithm(fitness_function,
                          num_attributes=NUM_ATTRIBUTES,
                          population_size=POPULATION_SIZE,
                          mutation_prob=MUTATION_PROB,
                          crossover_prob=CROSSOVER_PROB)
    GA.initialize_population()
    scores = []
    generation_counter = 0
    success_counter = 0

    while generation_counter < MAX_GEN and success_counter < MAX_SUCCESS:
        GA.compute_fitness_score()
        scores.append(np.mean(GA.fitness_scores))
        print('Generation', generation_counter, ", avg score:",
              scores[generation_counter], ", best:", GA.get_best())

        if GA.get_best() == target_sentence:
            success_counter += 1

        GA.run()
        generation_counter += 1

    return scores, generation_counter


def main():
    params = {"mutation_prob": .1, "crossover_prob": .75, "population_size": 100, "max_success": 22200, "max_gen": 1000}
    population_sizes = [10, 100, 500]
    mutation_probs = [0, .01, 1]
    scores = []
    times = []

    # influence of mutation_prob and pop_size
    param_control = "POPULATION_SIZE"  # which param to control

    if param_control == "POPULATION_SIZE":
        for ps in population_sizes:
            params['mutation_prob'] = .01
            params['population_size'] = ps
            tic = time()
            scores.append(simulate(params))
            toc = time()
            times.append(toc - tic)
            scores, gen_counter = simulate(params)
            plt.plot(scores, label="population_size={}, runtime={}s".format(ps, np.round(toc - tic, 3)))
            plt.xlabel('Generation')
            plt.ylabel('Score')
            plt.title('Average score per generation')
            plt.legend()
        plt.show()
    else:
        for mp in mutation_probs:
            params['mutation_prob'] = mp
            params['population_size'] = 100
            tic = time()
            scores.append(simulate(params))
            toc = time()
            times.append(toc - tic)
            scores, gen_counter = simulate(params)
            plt.plot(scores, label="mutation_prob={}".format(mp))
            plt.xlabel('Generation')
            plt.ylabel('Score')
            plt.title('Average score per generation')
            plt.legend()
        plt.show()


main()
