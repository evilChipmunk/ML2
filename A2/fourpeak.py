import numpy as np 
import mlrose
import mlrose_hiive as hive
import matplotlib.pyplot as plt
from time import time

 

def plotLine(x, y, ax, label, xLabel='Iterations'): 
    dev = np.std(y, axis=0)  
    ax.plot(x, y,  label=label)
    ax.fill_between(x, y - dev,  y + dev, alpha=0.1,)  
    color = 'tab:blue'
    ax.set_ylabel('Fitness', color=color)
    ax.set_xlabel(xLabel)

def plotTime(x, times, ax): 
    timeData = np.mean(times, axis=0)
    ax2 = ax.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Time (s)', color=color)
    ax2.plot(x, timeData, color=color, label='Time')
    ax2.tick_params(axis='y', labelcolor=color) 

def showLegend(fig, ax):
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)


def runHill(problem, basePath):
    iterations = 1000
    restart = 0 
    neighborhood = [2, 4, 12, 36]
    neighborhood = [20]
    neighborhood = [2, 4, 12, 36, 50, 75]
    neighborhood = [2, 75]
    fig, ax = plt.subplots()
    plt.title('Random Hill')
    # fig.tight_layout()
 
    times = np.zeros((len(neighborhood), iterations))
    nIndex = 0
    for neighbor in neighborhood:
        s = time()
        x = []  
        y = []
        for i in range(1, iterations + 1):
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=neighbor,  
                                                                restarts=restart, 
                                                                max_iters=i,
                                                                curve=False,  
                                                                random_state=1) 
            x.append(i)
            y.append(best_fitness) 
            e = time()
            timeTaken = e - s
            times[nIndex, i - 1] = timeTaken 
            print('Itt: {0} - Time:{1}'.format(i, timeTaken))
        nIndex += 1 
        plotLine(x, y, ax, 'Neighbors: {0}'.format(neighbor))      
    plotTime(x, times, ax)  
    showLegend(fig, ax)
 
    if basePath:
        plt.savefig('{0}\\{1}.png'.format(basePath, 'Hill'))
    else:
        plt.show()
    return
 
def runAnnealing(problem, basePath): 
    iterations = 500
    iterations = 1000 
    neighborhood = [10]
    neighborhood = [2, 4, 12, 36, 50, 75]
    neighborhood = [4]
    schedules = [('Exp', mlrose.ExpDecay()), ('Arith', mlrose.ArithDecay()), ('Geom', mlrose.GeomDecay())] 
    schedules = [('Exp', mlrose.ExpDecay())] 
  
    times = np.zeros((len(neighborhood), iterations))
    nIndex = 0
    schedule = mlrose.ExpDecay() 
    fig, ax = plt.subplots()
    plt.title('Annealing')
    for neighbor in neighborhood:
        s = time()
        x = []  
        y = []
        for i in range(1, iterations + 1):
            best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule,
                                                                max_attempts=neighbor,
                                                                max_iters=i,
                                                                random_state=1)
            x.append(i)
            y.append(best_fitness)  
            e = time() 
            timeTaken = e - s
            times[nIndex, i - 1] = timeTaken 
            print('Itt: {0} - Time:{1}'.format(i, timeTaken))
        nIndex += 1
        plotLine(x, y, ax, 'Neighbors: {0} - {1}'.format(neighbor, 'Exponential Decay'))   
    plotTime(x, times, ax)   
    showLegend(fig, ax)
    
    if basePath:
        plt.savefig('{0}\\{1}.png'.format(basePath, 'Annealing'))
    else:
        plt.show()
    return

def runGenetic(problem, basePath): 
    populationSizes = [20, 30, 50, 100, 200, 250]
    neighborhood = [2, 4, 18, 19] 
    neighborhood = [5, 100] 
    neighborhood = [2, 4, 12, 36]
    neighborhood = [2, 4, 12, 36, 50, 75] 
    neighborhood = [36, 50]
    iterations = 1000
 
    populationSizes = [10, 20, 30, 50, 100, 250, 500, 750, 1000]
    times = np.zeros((len(neighborhood), len(populationSizes)))
    nIndex = 0
    fig, ax = plt.subplots()
    plt.title('Genetic')
    for neighbor in neighborhood: 
        s = time()
        pIndex = 0
        x = []
        y = []
        for i in populationSizes: 
            best_state, best_fitness = mlrose.genetic_alg(problem, 
                                                            pop_size=i, 
                                                            mutation_prob=0.001, 
                                                            max_attempts=neighbor,  
                                                            max_iters= iterations,
                                                            random_state=1)
            x.append(i)
            y.append(best_fitness)  
            e = time()
            timeTaken = e - s
            times[nIndex, pIndex] = timeTaken
            pIndex += 1
            print('Pop: {0} - Time:{1}'.format(i, timeTaken))
        nIndex += 1  
        plotLine(x, y, ax, 'Neighbors - {0}'.format(neighbor), 'Population')  
    plotTime(x, times, ax)   
    showLegend(fig, ax)
    if basePath:
        plt.savefig('{0}\\{1}.png'.format(basePath, 'Genetic'))
    else:
        plt.show()

def runMimic(problem, basePath): 
    populationSizes = [20, 30, 50, 100, 200, 250]
    neighborhood = [2, 4, 18, 19] 
    neighborhood = [2, 4, 12, 36]
    neighborhood = [2]
    neighborhood = [2, 4, 12, 36, 50, 75]
    neighborhood = [36]
    fig, ax = plt.subplots()
    plt.title('MIMIC')
    iterations = 100

    populationSizes = [10, 20, 30, 50, 100, 250, 500, 750, 1000]
    times = np.zeros((len(neighborhood), len(populationSizes))) 
    nIndex = 0

    for neighbor in neighborhood: 
        s = time()
        pIndex = 0
        x = []
        y = []
        for i in populationSizes:   
            best_state, best_fitness = mlrose.mimic(problem, pop_size= i
                                                    , max_attempts=neighbor
                                                    , max_iters= iterations
                                                    , random_state=1
                                                    , fast_mimic=True)
            x.append(i)
            y.append(best_fitness)  
            e = time()
            timeTaken = e - s
            times[nIndex, pIndex] = timeTaken
            pIndex += 1
            print('Pop: {0} - Time:{1}'.format(i, timeTaken))
        nIndex += 1  
        plotLine(x, y, ax, 'Neighbors - {0}'.format(neighbor), 'Population')   
    plotTime(x, times, ax)   
    showLegend(fig, ax)
    if basePath:
        plt.savefig('{0}\\{1}.png'.format(basePath, 'MIMIC'))
    else:
        plt.show()


def run(): 
    fitness = mlrose.FourPeaks(t_pct=0.1)  
    arrLen = 100 
    problem = mlrose.DiscreteOpt(length=arrLen, fitness_fn=fitness)  

    # basePath = 'C:\\Users\\mwest\\Desktop\\ML\\source\\Machine-Learning-Local - Copy\\Graphs\\randomized\\Four Peaks\\'
    basePath = None
    runHill(problem, basePath) 
    runAnnealing(problem, basePath) 
    runGenetic(problem, basePath)
    runMimic(problem, basePath)

    return
