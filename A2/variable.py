import numpy as np 
import matplotlib.pyplot as plt
from time import time
import mlrose_hiive as hive
import mlrose
 
def showLegend(fig, ax):
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
 
def runComplexity(pType, fitness, lengths):
 
    # basePath = 'C:\\Users\\mwest\\Desktop\\ML\\source\\Machine-Learning-Local - Copy\\Graphs\\randomized\\Complexity'
    basePath = None
    hill = np.zeros((len(lengths), 3))
    annealing = np.zeros((len(lengths), 3))
    genetic = np.zeros((len(lengths), 3))
    mimic = np.zeros((len(lengths), 3))
    optimal = np.zeros((len(lengths), 3))

    i = 0
    for arrLen in lengths: 
        if pType == 'TSP':  
            coords_list = set()  
            for j in range(arrLen):
                coords_list.add((np.random.randint(1, arrLen), np.random.randint(1, arrLen))) 
            coords_list = list(coords_list)
            arrLen = len(coords_list)
            fitness = mlrose.TravellingSales(coords = coords_list) 
            problem = mlrose.TSPOpt(length = arrLen, fitness_fn = fitness, maximize=True) 
        else:
            problem = mlrose.DiscreteOpt(length=arrLen, fitness_fn=fitness)    
        
        y, time = runComplexityHill(pType, problem)
        hill[i, 0] = arrLen
        hill[i, 1] = y
        hill[i, 2] = time
        print('{0} \t Hill Length: {1} - Score:{2} - Time:{3}'.format(pType, arrLen, y, time)) 
        
        y, time = runComplexityAnnealing(pType, problem)
        annealing[i, 0] = arrLen
        annealing[i, 1] = y
        annealing[i, 2] = time
        print('{0} \t Annealing Length: {1} - Score:{2} - Time:{3}'.format(pType, arrLen, y, time)) 
        
        y, time = runComplexityGenetic(pType, problem)
        genetic[i, 0] = arrLen
        genetic[i, 1] = y
        genetic[i, 2] = time
        print('{0} \t Genetic Length: {1} - Score:{2} - Time:{3}'.format(pType, arrLen, y, time)) 
        
        y, time = runComplexityMIMIC(pType, problem)
        mimic[i, 0] = arrLen
        mimic[i, 1] = y
        mimic[i, 2] = time
        print('{0} \t MIMIC Length: {1} - Score:{2} - Time:{3}'.format(pType, arrLen, y, time)) 

        y, time = runComplexityOptimal(pType, problem)
        optimal[i, 0] = arrLen
        optimal[i, 1] = y
        optimal[i, 2] = time
        print('{0} \t Optimal Length: {1} - Score:{2} - Time:{3}'.format(pType, arrLen, y, time)) 
        
        i += 1

    fig, ax = plt.subplots()
    plt.title('{0} Complexity vs Fitness Evaluation'.format(pType)) 
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Complexity')
    ax.plot(hill[:, 0], hill[:, 1],  label='Randomized Hill Fitness') 
    ax.plot(annealing[:, 0], annealing[:, 1],  label='Annealing Fitness') 
    ax.plot(genetic[:, 0], genetic[:, 1],  label='Genetic Fitness') 
    ax.plot(mimic[:, 0], mimic[:, 1],  label='MIMIC Fitness') 
    ax.plot(optimal[:, 0], optimal[:, 1],  label='Optimal Fitness + 1') 
    showLegend(fig, ax)
    plt.savefig('{0}\\{1} Fitness.png'.format(basePath, pType))
 
    fig, ax = plt.subplots()
    plt.title('{0} Complexity vs Time Evaluation'.format(pType)) 
    ax.set_ylabel('Time')
    ax.set_xlabel('Complexity')
    ax.plot(hill[:, 0], hill[:, 2],  label='Randomized Hill Time') 
    ax.plot(annealing[:, 0], annealing[:, 2],  label='Annealing Time') 
    ax.plot(genetic[:, 0], genetic[:, 2],  label='Genetic Time') 
    ax.plot(mimic[:, 0], mimic[:, 2],  label='MIMIC Time')  
    showLegend(fig, ax)
    plt.savefig('{0}\\{1} Time.png'.format(basePath,  pType))

 
    # plt.show()

def runComplexityOptimal(pType, problem):
    state = np.zeros((problem.length))

    fitness = problem.fitness_fn

    if pType == 'One Max':
        for i in range(1, problem.length):
            state[i] = 1

    elif pType == 'Flip Flop':
        for i in range(1, problem.length):
            if i % 2 == 0:
                state[i] = 1
            else:
                state[i] = 0 
    else: 
        for i in range(1, problem.length):
            state[i] = 0

        head = problem.length * .1
        head = int(np.ceil(head)) + 1
        for i in range(head):
            state[i] = 1
    
        # b = []
        # for i in range(100):
        #     b.append(0)
        
        # for i in range(11):
        #     b[i] = 1
        # state = np.array(b)
        # print(fitness.evaluate(state))


    best = fitness.evaluate(state)

    return best + 1, 0
  
def runComplexityHill(pType, problem):    

    if pType == 'One Max':
        neighbor = 20 
        iterations = 400
    elif pType == 'Flip Flop':
        neighbor = 36
        iterations = 200
    else:
        neighbor = 75 
        iterations = 400
        
    s = time() 
    best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                        max_attempts=neighbor,    
                                                        max_iters=iterations,
                                                        curve=False,  
                                                        random_state=1) 
    # best_state, best_fitness, c = mlrose.random_hill_climb(problem,
    #                                                     max_attempts=neighbor,    
    #                                                     max_iters=iterations,
    #                                                     curve=False,  
    #                                                     random_state=1) 
    timeTaken = time() - s 
    return best_fitness, timeTaken
    
def runComplexityAnnealing(pType, problem):    
    if pType == 'One Max':
        neighbor = 10 
        iterations = 400
    elif pType == 'Flip Flop':
        neighbor = 36
        iterations = 700
    else:
        neighbor = 4 
        iterations = 1000
 
    s = time() 
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(),
                                                        max_attempts=neighbor,
                                                        max_iters=iterations,
                                                        random_state=1)
    # best_state, best_fitness, c = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(),
    #                                                     max_attempts=neighbor,
    #                                                     max_iters=iterations,
    #                                                     random_state=1)
    timeTaken = time() - s
    
    return best_fitness, timeTaken
    
def runComplexityGenetic(pType, problem):    
    if pType == 'One Max':
        neighbor = 50
        populationSize = 800
        iterations = 1000 
        mutation = 0.001
    elif pType == 'Flip Flop':
        neighbor = 36
        populationSize = 1000
        iterations = 1000 
        mutation = 0.001
    else:
        neighbor = 50
        populationSize = 250
        iterations = 10000 
        mutation = 0.15
    s = time() 
    # best_state, best_fitness = mlrose.genetic_alg(problem, 
    #                                                 pop_size=populationSize, 
    #                                                 mutation_prob=mutation, 
    #                                                 max_attempts=neighbor,  
    #                                                 max_iters= iterations,
    #                                                 random_state=1)
    best_state, best_fitness, c = hive.genetic_alg(problem, 
                                                    pop_size=populationSize, 
                                                    mutation_prob=mutation, 
                                                    max_attempts=neighbor,  
                                                    max_iters= iterations,
                                                    random_state=1)
    timeTaken = time() - s 
    return best_fitness, timeTaken
  
def runComplexityMIMIC(pType, problem):   
    # return 0, 0
    if pType == 'One Max':
        neighbor = 2
        populationSize = 200
        iterations = 100
    elif pType == 'Flip Flop':
        neighbor = 36
        populationSize = 800
        iterations = 1000 
    else:
        # neighbor = 36
        # populationSize = 800
        # iterations = 100 
        neighbor = 36
        populationSize = 800
        iterations = 100
 
    s = time() 
    best_state, best_fitness = mlrose.mimic(problem, pop_size= populationSize
                                            , max_attempts=neighbor
                                            , max_iters= iterations
                                            , random_state=1
                                            , fast_mimic=True) 
    # best_state, best_fitness, c = hive.mimic(problem, pop_size= populationSize
    #                                         , max_attempts=neighbor
    #                                         , max_iters= iterations
    #                                         , random_state=1) 
    timeTaken = time() - s
    
    return best_fitness, timeTaken



def run(): 
    # basePath = 'C:\\Users\\mwest\\Desktop\\ML\\source\\Machine-Learning-Local - Copy\\Graphs\\randomized\\Complexity\\One Max\\'
    basePath = None
 
 

    # lengths = range(1, 501, 25) 
    # lengths = range(1, 101, 50) 
    lengths = [10, 100, 200, 300, 400, 500]
    lengths = [10, 50, 100, 150, 200]
    lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # lengths = [5, 10]
    fitness = mlrose.OneMax()
    runComplexity('One Max', fitness, lengths)

    fitness = mlrose.FourPeaks(t_pct=0.10)  
    runComplexity('Four Peaks', fitness, lengths)

    fitness = mlrose.FlipFlop()  
    runComplexity('Flip Flop', fitness, lengths)
 
    # runComplexity('TSP', None, lengths)

     # fitness = mlrose.Queens()
    # runComplexity('Queens', fitness, lengths)

    return
