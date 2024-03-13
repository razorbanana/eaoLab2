import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
NUM_SPECIES = 3
TARGET_SIZE = 100

def ackley_function(x, a=20, b=0.2, c=2*np.pi ):
    d = len(x)
    term1 = -a * np.exp(-b * np.sqrt(1/d * np.sum(x**2)))
    term2 = -np.exp(1/d * np.sum(np.cos(c * x)))
    term3 = a + np.e
    return term1 + term2 + term3

def replace_worst_individuals(subPopulations, subFitnesses, offspring, fo, i):
    subFitnesses_np = np.array(subFitnesses[i])
    worst_indices = np.argsort(-subFitnesses_np)[:len(offspring)]
    #print(subFitnesses[i], worst_indices)
    for j, index in enumerate(worst_indices):
        subPopulations[i][index] = offspring[j]
        subFitnesses[i][index] = fo[j]

def selection(population, fitness, num_parents):
    # Select parents based on their fitness
    num_parents = int(num_parents)
    sorted_indices = np.argsort(fitness)
    selected_parents = np.array(population)[sorted_indices[:num_parents]]
    return selected_parents

def crossover(parents, offspring_size):
    offspring_size = int(offspring_size)
    offspring = []
    crossover_point = np.random.randint(1, len(parents[0])+1)
    for i in range(offspring_size):
        # Index of the first parent to mate.
        parent1_idx = i % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (i+1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        k = parents[parent1_idx].copy()
        k[crossover_point:] = parents[parent2_idx][crossover_point:]
        offspring.append(k)
    return offspring

def mutation(offspring_crossover, mutation_rate):
    for idx in range(len(offspring_crossover)):
        for gene_idx in range(len(offspring_crossover[0])):
            if np.random.rand() < mutation_rate:
                # Generate a random value to add to the gene.
                offspring_crossover[idx][gene_idx] += np.random.uniform(-0.5*offspring_crossover[idx][gene_idx], 0.5*offspring_crossover[idx][gene_idx])
    return offspring_crossover

def geneticOperators(population, fitness, num_parents=0.5, mutation_rate=0.2):
    selected_parents = selection(population, fitness, len(population)*num_parents)
    offspring_crossover = crossover(selected_parents, len(population)*num_parents)
    offspring_mutation = mutation(offspring_crossover, mutation_rate)
    return offspring_mutation

def group1(n):
    return np.array_split(range(n), NUM_SPECIES)

def group2(n):
    numbers = list(range(n))
    np.random.shuffle(numbers)
    return np.array_split(numbers, NUM_SPECIES)

def coevolve_best(n, max_iter=1000, bounds=[-32.768, 32.768]):
    b = np.random.uniform(bounds[0], bounds[1], n)
    fb = ackley_function(b)
    subIndexes = group1(n)
    #print(subIndexes)
    subFitnesses = []
    subPopulations = []
    fb_history = [fb]
    for i in range(NUM_SPECIES):
        subPop = []
        subFitness = []
        for j in range(TARGET_SIZE):
            subSol = np.random.uniform(bounds[0], bounds[1], len(subIndexes[i]))
            subPop.append(subSol)
            temp = b.copy()
            temp[subIndexes[i]] = subSol
            fitness = ackley_function(temp)
            subFitness.append(fitness)
        subPopulations.append(subPop)           
        subFitnesses.append(subFitness)
    iter = 0
    while iter < max_iter and fb > 0.01:
        iter += 1
        for i in range(NUM_SPECIES):
            offspring = geneticOperators(subPopulations[i], subFitnesses[i])
            fo = []
            for j in offspring:
                temp = b.copy()
                temp[subIndexes[i]] = j
                fo.append(ackley_function(temp))
            for j in range(len(offspring)):
                if fo[j] < fb:
                    fb = fo[j]
                    b[subIndexes[i]] = offspring[j]
            replace_worst_individuals(subPopulations, subFitnesses, offspring, fo, i)
        fb_history.append(fb)
    return fb_history

def coevolve_worst(n, max_iter=1000, bounds=[-32.768, 32.768]):
    b = np.random.uniform(bounds[0], bounds[1], n) 
    fb = ackley_function(b)
    subIndexes = group1(n)
    #print(subIndexes)
    subFitnesses = []
    subPopulations = []
    for i in range(NUM_SPECIES):
        subPop = []
        subFitness = []
        for j in range(TARGET_SIZE):
            subSol = np.random.uniform(bounds[0], bounds[1], len(subIndexes[i]))
            subPop.append(subSol)
            temp = b.copy()
            temp[subIndexes[i]] = subSol
            fitness = ackley_function(temp)
            subFitness.append(fitness)
        subPopulations.append(subPop)           
        subFitnesses.append(subFitness)    
    for i in range(NUM_SPECIES):
        for j in range(TARGET_SIZE):
            temp = b.copy()
            temp[subIndexes[i]] = subPopulations[i][j] 
            if ackley_function(temp) > fb:
                    fb = ackley_function(temp)
                    b[subIndexes[i]] = subPopulations[i][j] 
    #print(b)
    result = b.copy()
    fresult = fb.copy()
    fbresult = []
    for i in range(NUM_SPECIES):
        fbresult.append(fb)
    iter = 0
    while iter < max_iter and fresult > 0.01:
        iter += 1
        for i in range(NUM_SPECIES):
            offspring = geneticOperators(subPopulations[i], subFitnesses[i])
            fo = []
            for j in offspring:
                temp = b.copy()
                temp[subIndexes[i]] = j
                fo.append(ackley_function(temp))
            for j in range(len(offspring)):
                if fo[j] < fbresult[i]:
                    result[subIndexes[i]] = offspring[j]
                    fbresult[i] = fo[j] 
                    fresult = ackley_function(result)    
            replace_worst_individuals(subPopulations, subFitnesses, offspring, fo, i)

def coevolve_worst2(n, max_iter=1000, bounds=[-32.768, 32.768]):
    b = np.random.uniform(bounds[0], bounds[1], n) 
    fb = ackley_function(b)
    subIndexes = group1(n)
    subFitnesses = []
    subPopulations = []
    result = b.copy()
    fresult = fb.copy()
    fresult_history = [fresult]
    for i in range(NUM_SPECIES):
        subPop = []
        subFitness = []
        for j in range(TARGET_SIZE):
            subSol = np.random.uniform(bounds[0], bounds[1], len(subIndexes[i]))
            subPop.append(subSol)
            temp = b.copy()
            temp[subIndexes[i]] = subSol
            fitness = ackley_function(temp)
            subFitness.append(fitness)
        subPopulations.append(subPop)           
        subFitnesses.append(subFitness)    
    iter = 0
    while iter < max_iter and fresult > 0.01:
        iter += 1
        for i in range(NUM_SPECIES):
            offspring = geneticOperators(subPopulations[i], subFitnesses[i])
            fo = []
            for j in offspring:
                temp = b.copy()
                temp[subIndexes[i]] = j
                fo.append(ackley_function(temp))
            for j in range(len(offspring)):
                temp = b.copy()
                temp[subIndexes[i]] = result[subIndexes[i]]
                if fo[j] < ackley_function(temp):
                    result[subIndexes[i]] = offspring[j]
                    fresult = ackley_function(result)
                if fo[j] > fb:
                    fb = fo[j]
                    b[subIndexes[i]] = offspring[j]     
            replace_worst_individuals(subPopulations, subFitnesses, offspring, fo, i)
        fresult_history.append(fresult)
    return fresult_history

def coevolve_elitism(n, max_iter=1000, bounds=[-32.768, 32.768], elite_rate=0.2):
    b = np.random.uniform(bounds[0], bounds[1], n)
    fb = ackley_function(b)
    subIndexes = group1(n)
    #print(subIndexes)
    subFitnesses = []
    subPopulations = []
    fb_history = [fb]
    for i in range(NUM_SPECIES):
        subPop = []
        subFitness = []
        for j in range(TARGET_SIZE):
            subSol = np.random.uniform(bounds[0], bounds[1], len(subIndexes[i]))
            subPop.append(subSol)
            temp = b.copy()
            temp[subIndexes[i]] = subSol
            fitness = ackley_function(temp)
            subFitness.append(fitness)
        subPopulations.append(subPop)           
        subFitnesses.append(subFitness)
    iter = 0
    while iter < max_iter and fb > 0.01:
        iter += 1
        for i in range(NUM_SPECIES):
            offspring = geneticOperators(subPopulations[i], subFitnesses[i])
            fo = []
            for j in offspring:
                temp = b.copy()
                temp[subIndexes[i]] = j
                fo.append(ackley_function(temp))
            for j in range(len(offspring)):
                if fo[j] < fb:
                    fb = fo[j]
                    b[subIndexes[i]] = offspring[j]
            replace_worst_individuals(subPopulations, subFitnesses, offspring, fo, i)
            
        # Elitism: Preserve the best individuals in each species
        for i in range(NUM_SPECIES):
            sorted_indices = np.argsort(subFitnesses[i])
            num_elites = int(elite_rate * TARGET_SIZE)
            elite_indices = sorted_indices[:num_elites]
            chosen_index = np.random.choice(elite_indices)
            b[subIndexes[i]] = subPopulations[i][chosen_index]
            fb = subFitnesses[i][chosen_index]
        fb_history.append(fb)
    return np.array(fb_history)

def plot_experiment(n=10, bounds=[-5,5]):
    fb_history_1 = coevolve_best(n, bounds=bounds)
    fb_history_2 = coevolve_worst2(n, bounds=bounds)
    fb_history_3 = coevolve_elitism(n, bounds=bounds)
    print("Best num of iterations - " + str(len(fb_history_1)))
    print("Worst num of iterations - " + str(len(fb_history_2)))
    print("Elitism num of iterations - " + str(len(fb_history_3)))
    max_length = max(len(fb_history_1), len(fb_history_2), len(fb_history_3))
    fb_history_1_interp = interp1d(np.linspace(0, 1, len(fb_history_1)), fb_history_1)(np.linspace(0, 1, max_length))
    fb_history_2_interp = interp1d(np.linspace(0, 1, len(fb_history_2)), fb_history_2)(np.linspace(0, 1, max_length))
    fb_history_3_interp = interp1d(np.linspace(0, 1, len(fb_history_3)), fb_history_3)(np.linspace(0, 1, max_length))

    # Plot all fb_history arrays together
    iterations = np.arange(max_length)
    plt.plot(iterations, fb_history_1_interp, label='Best')
    plt.plot(iterations, fb_history_2_interp, label='Worst')
    plt.plot(iterations, fb_history_3_interp, label='Elitism')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Ackley Function')
    plt.title('Coevolution Progress')
    plt.legend()

    plt.show()

def main():
    #x = np.random.uniform(-5, 5, dimensions)
    #x = np.array([1,1,1,1,1,1,1,1,1,1])
    plot_experiment()
    

if __name__ == "__main__":
   main()  