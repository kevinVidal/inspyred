# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:03:24 2020

@author: vidalk
"""


import inspyred
from random import Random
import _DoE.DoE as DOE  # utilisation de pyDOE
import time , os
import numpy as np
from _Surrogate import KRG, KPLSK, KPLS # utilisation de la libarie surrogate modeling toolbox (SMT)
from _Surrogate.utility_modules import cross_validation, load_data # utilisation de la libarie surrogate modeling toolbox (SMT)
from pprint import pprint

###########################################################################

def generate_random_population(random, args):
    ''' template for generating a set of random individual inside a population '''
    population = []
    bounder = args["_ec"].bounder
    for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
        population.append(random.uniform(lo, hi))
    return population
    
###########################################################################

def evaluate_hyperparameters(candidates, args):
    ''' objectif function for hyper parameter mono-objective optimization '''
    nb_obj = args.get('num_obj', 3)
    p = args.get('poly', None)
    c = args.get('corr', None)
    cv = args.get('cross_validation', None)
    fitness = []
    theta_mat = []
    for cs in candidates:
        theta_mat = cs
        model = KRG(theta0=theta_mat,print_global= False, poly= p, corr=c)
        r = cv(model,K=1, nproc=1) 
        if nb_obj>1:
            value_fitness = 0
            for j in range(nb_obj):
                value_fitness += r[j]
        else:
            value_fitness = r
        fitness.append(value_fitness)
    return fitness

###########################################################################

def surrogate_evaluator_mo(candidates, args):
    #### NOT WORKING #####
    ''' objectif function for surrogate assisted multi-objectif optimization '''
    model = args.get('metamodel', None)
    nb_obj = args.get('num_obj', None)
    fitness = []
    for c in candidates:
        results = model.predict_values(np.array([c]))
        f = []
        for j in range(nb_obj):
            f.append(results[0][j])
        fitness.append(inspyred.ec.emo.Pareto(f))
    return fitness

###########################################################################

def surrogate_evaluator_so(candidates, args):
    ''' objectif function for surrogate assisted single-objectif optimization '''
    model = args.get('metamodel', None)
    nb_obj = args.get('num_obj', None)
    weight = 1/float(nb_obj) # a voir si il y a une ponderation
    weight = 1
    fitness = []
    for c in candidates:
        results = model.predict_values(np.array([c]))
        value_fitness = 0
        for j in range(nb_obj):
            value_fitness += weight*results[0][j]
        fitness.append(value_fitness)
    return fitness

###########################################################################

def opt_hyperParameters(directory, X, Y, nb_obj, nb_dim, p = 'linear', c = 'squar_exp', init_parameters=[1e-2]):
    ''' function for optimizing the hyperparameters of a given kriging function
    inputs:
    ----------------
    directory: str path
        location where the optimization files are created
    X,Y : data array
        results data to test with the metamodel
    nb_obj : int
        define the number of objectif can be len(output_col) 
    
    optional:
    ---------------
    p : str
        Regression function type (can be 'constant'[ordinary kriging], 'linear' or 'quadratic' [universal kriging])  default= 'linear'
    c : str
        Correlation function type (can be 'abs_exp' or 'squar_exp') default = 'squar_exp'
    init_parameters: array
        intial hyperparameters for the metamodel, default = 0.01
        
    returns:
    ---------------
    final_hyperparameters : list
        list of the hyperparameter values directly usable for the metamodel fitting
    '''
    time_stamp = time.strftime("%d-%m-%Y_%I-%M-%S_%p")
    t_i = KRG(theta0=init_parameters,print_global= False, poly= p, corr=c)
    cv = cross_validation(X,Y)
    r_i = cv(t_i,K=1, debug=True, nproc=1) 
    print('-> hyperparameters optimization >> Initial correlation coefficient : ')
    pprint(r_i)
    
    try:
        minr_i =  min(r_i)
    except:
        minr_i = r_i
        
    print('-> hyperparameters optimization >> minimum correlation coefficient = {}'.format(minr_i))
    if minr_i < 0.7:
        # optimize if 1 correlation coef < 0.7
        if minr_i < 0.3:
            # fine search
            # number of pop
            pop = 30
            # number of iteration
            nb_ite = 20
        elif minr_i < 0.5:
            # medium search
            # number of pop
            pop = 30
            # number of iteration
            nb_ite = 10  
        else:
            # coarse search
            # number of pop
            pop = 15
            # number of iteration
            nb_ite = 10           
        
        # seed the optimizer with a lhs sampling for better volumetric repartitions
        init_canditates = DOE.lhs(len(X.T), pop-1, criterion='maximin', iterations=10)
        for i in range(nb_dim):
            # scaled each column to min max of hyperparameters
            DOE.scaling_doe(init_canditates,i,1e-4,1e-1)
            
        # add default hyperparameters 1e-02
        init_canditates = DOE.union(init_canditates,[[1e-02 for _ in range(len(X.T))]])
    
        # optimizer setup
        ea = inspyred.ec.GA(Random())
        ea.terminator = [inspyred.ec.terminators.evaluation_termination, 
                         inspyred.ec.terminators.diversity_termination]
        #ea.observer = [inspyred.ec.observers.file_observer,inspyred.ec.observers.plot_observer,inspyred.ec.observers.stats_observer]  
        ea.observer = [inspyred.ec.observers.file_observer,inspyred.ec.observers.stats_observer]
        # optimize
        start_time = time.time()
        stat_f = directory + '\\' + time_stamp + '_statistics.csv'
        indiv_f = directory+ '\\' + time_stamp + '_individuals.csv'
        
        # create dir if needed
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        print('\n-> hyperparameters optimization >> Start optimization of hyperparameters')     
        with open(stat_f, 'w+') as stat, open(indiv_f, 'w+') as ind:
            final_pop = ea.evolve(generator=generate_random_population,
                                  evaluator=evaluate_hyperparameters,
                                  pop_size=pop,
                                  maximize=True,
                                  bounder=inspyred.ec.Bounder(1e-4, 1e-1),
                                  seeds = list(init_canditates),
                                  max_evaluations=nb_ite*pop,
                                  num_obj=nb_obj,
                                  poly = p,
                                  corr = c,
                                  cross_validation = cv,
                                  statistics_file = stat,
                                  individuals_file = ind)
        
        print('-> hyperparameters optimization >> finished in {:1.2f} seconds '.format(time.time() - start_time))   
        inspyred.ec.analysis.generation_plot(stat_f, errorbars=True, 
                                             save_file=directory + '\\' + time_stamp + '_HyperparametersOpti_GenerationPlot.png')
        inspyred.ec.analysis.allele_plot(indiv_f, normalize=False, alleles=None, generations=None, 
                                         save_file=directory + '\\' + time_stamp + '_HyperparametersOpti_AllelePlot.png')
            
        # sort and extract optimal hyperparameters
        final_pop.sort(reverse=True)
        final_hyperparameters = final_pop[0].candidate
    else:
        final_hyperparameters = [1e-2]
        
    # cross validation with final hyperparameters
    t_f = KRG(theta0=final_hyperparameters,print_global= False, poly= p, corr=c)  
    r, y_p, y_t  = cv(t_f,K=1, nproc=1, output_array=True) 
    print('-> hyperparameters optimization >> Correlation coefficient after optimization: ')
    pprint(r)
    gain = []
    try:
        for i in range(len(r)):
            gain.append('{:1.2%}'.format((r[i]-r_i[i])/abs(r_i[i])))
    except:
        gain.append('{:1.2%}'.format((r-r_i)/abs(r_i)))
    print('-> hyperparameters optimization >> new hyperparameters gain : {} '.format(gain))  
    print('-> hyperparameters optimization >> optimal hyperparameters : ')
    print(final_hyperparameters)
    print('\n')
    
    return final_hyperparameters


def surrogate_assisted_optimization(optimizer,min_max,nb_obj,surrogate):
    ''' function for optimizing the surrogate output 
    inputs:
    ----------------
    optimizer: str
        defines if mono or multi objectives
    min_max : list
        defines bounds of thge input values
    nb_obj : int
        defines number of objectives, in case of mono objective with nb_obj>1, the output is the weighted summ of all objectives value with equal weight (=1/nb_obj) 
    surrogate : obj
        surrogate object, the surrogate intiialization and training must be done before calling this function
        
    returns:
    -------------
    best.candidate, best.fitness : list
    list of the input values with the best fitness, and the fitness of the best candidate for comparison with actual computation
    '''    
    prng = Random()
    prng.seed(time.time())
    if optimizer == 'multi':
        # not developped NOT WORKING
        pop = 50
        ite = 10
#        # optimizer setup using NSGA2
#        ea = inspyred.ec.emo.NSGA2(prng)
#        ea.variator = [inspyred.ec.variators.blend_crossover, 
#                       inspyred.ec.variators.gaussian_mutation]
#        
#        ea.terminator = inspyred.ec.terminators.generation_termination
#        final_pop = ea.evolve(generator=generate_random_population, 
#                              evaluator=surrogate_evaluator_mo, 
#                              pop_size=pop,
#                              maximize=False,
#                              bounder=inspyred.ec.Bounder(min_max[0], min_max[1]),
#                              max_generations=ite*pop,
#                              metamodel= surrogate,
#                              num_obj= nb_obj)
#        
#        final_arc = ea.archive
#        print('-> MO-SAO optimization >> Search best Solutions: \n')
#        best_fitness = 0
#        best_candidate = []
#        weight = 1/nb_obj
#        # select a canditate based on equal objectives ponderation
#        for f in final_arc:
#            fitness_w = 0
#            for value in range(len(output_col)):
#                fitness_w += weight*f.fitness[value]
#            if fitness_w > best_fitness:
#                best_candidate = f
#        return best_candidate.candidate, best_candidate.fitness
    elif optimizer == 'mono':
        pop = 200
        ite = 50
        # optimizer setup using GA
        ea = inspyred.ec.GA(prng)
        ea.terminator = [inspyred.ec.terminators.evaluation_termination, 
                         inspyred.ec.terminators.diversity_termination]
        ea.observer = [inspyred.ec.observers.stats_observer]
        final_pop = ea.evolve(generator=generate_random_population,
                                  evaluator=surrogate_evaluator_so,
                                  pop_size=pop,
                                  maximize=False,
                                  bounder=inspyred.ec.Bounder(min_max[0], min_max[1]),
                                  max_evaluations=ite*pop,
                                  metamodel= surrogate,
                                  num_obj=nb_obj)
        
        return final_pop[0].candidate, final_pop[0].fitness
        
        

if __name__ == '__main__':
    #### HARDCODED
    baseDir = r'D:\_Scripts_Python\Projet_dynamho\_Test'
    file_in = 'pilote_1.plan'
    file_out = 'results_1.plan'
    
    opti_dir = baseDir+'\opti_dir'
    
    output_col = (3,4,5,6)
    
    path_in = os.path.join(baseDir,file_in)
    path_out = os.path.join(baseDir,file_out)
    
    X, Y, line_plan, line_result = load_data(path_in,path_out,delimiter='|', usecols=output_col)
    
    # optimizer hyperparameters
    hp = opt_hyperParameters(opti_dir, X, Y, len(output_col))
    # create the surrogate object
    surrogate = KRG(theta0=hp,print_global= False, poly= 'linear', corr='squar_exp')  
    # train model with entire data set
    surrogate.set_training_values(X,Y[:,:])
    surrogate.train()
    
    min_max=((-18, -5, -10, -800, -150, 0.3),( 12,  2,  30,  650,   80, 0.9))
    
    best = surrogate_assisted_optimization('mono',min_max,len(output_col),surrogate)
    print(best)
    import _FileDataManagement.DyNaMHo_IOFile as io
    with open(opti_dir+'\sao_results.csv','a+') as res:
        header = ['parameter_{}'.format(i+1) for i in range(len(best[0]))]
        header.append('; predicted fitness')
        res.write(io.listToString(header,';')+'\n')
        for items in best:
            if isinstance(items, list):
                for val in items:
                    res.write(str(val)+';')
            if isinstance(items, float):
                res.write(';'+str(items)+';')
        res.write('\n')
        
    # append a new colum
    
                    
        
        
    