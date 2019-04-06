#----------
# PARSING
#----------

def parsing_dimacs(m):
    """
    A function to parse a provided DIMACS-file.
    
    Args:
        m (str): The DIMACS-file's file path 
        
    Returns:
        A list of lists containing all of the DIMACS-file's constrains. Each constrain is represented by a seperate sub-list.
    
    """
    
    dimacs = list()
    dimacs.append(list())
    
    with open(m) as mfile:
        for line in mfile:
            tokens = line.split()
            if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                for tok in tokens:
                    lit = int(tok)
                    if lit == 0:
                        dimacs.append(list())
                    else:
                        dimacs[-1].append(lit)
    assert len(dimacs[-1]) == 0
    dimacs.pop()

    return dimacs
    
def parsing_text(m):
    """
    A function to parse a provided text-file containing a model's features or interactions.
    
    Args:
        m (str): The text-file's file path 
        
    Returns:
        When parsing a feature file: a dictionary with the features' names as keys and the features' values as values. 
        When parsing an interactions file: a dictionary with tuples of features concatenated by # as keys and the tuples' values as values.
    
    """
    features = dict()
    
    with open(m) as ffile:
        for line in ffile:
            line = line.replace("\n", "")
            tokens = line.split(": ")
            if len(tokens[-1]) > 0 and len(tokens) > 1:
                features[tokens[0]] = float(tokens[-1])
            else:
                features[tokens[0]] = ""
                
    return features

def parsing_variants(m):
    """
    A function to parse a provided text-file containing a model's variants.
    
    Args:
        m (str): The text-file's file path
        
    Returns:
        A list of lists containing all variants. Each variant is represented by a seperate sub-list.
        
    """
    cnf = list()

    with open(m) as ffile:
        for line in ffile:
            line = line.replace("\n", "")
            line = [int(i) for i in list(line)]
            if len(line) > 0:
                cnf.append(line)
    return cnf
#----------
# WRITING TO FILE
#----------

def writing_text(directory, old_data, new_data, filename):
    """
    A function to write the new features and interactions to their respective txt-files.
    
    Args:
        directory (str): The path to the new txt-file
        old_data (dict): The dict with feature/interaction names
        new_data (dict): The dict with new/estimated feature/interaction values
        filename (str): Name for the new file
        
    """
    from shutil import copyfile

    new_data_copy = list(new_data)
    old_data_keys = list(old_data.keys())
    new_file = directory + "/" + filename + ".txt"
    
    with open(new_file, 'w') as ffile:
        ffile.write("\n".join(str(i) for i in old_data))
        ffile.close()
    
    with open(new_file, 'r') as ffile:
        data = ffile.readlines()
        for line in range(0, len(data)):
            tokens = data[line].split()
            if len(tokens[-1]) > 0:# and len(tokens) > 1:
                #data[line] = ('#'.join(map(str, tokens[:-1]))) + " " + str(new_data_copy[0]) +"\n"
                data[line] = tokens[0] + ": " + str(new_data_copy[0])+"\n"
                new_data_copy.pop(0)

    with open(new_file, 'w') as ffile:
        ffile.writelines(data)


#----------
# GENERATING VARIANTS
#----------

def get_valid_variants(c, size):
    """
    A function to compute the valid variants of a model.
    
    Args:
        c (list): All constrains provided for the model.
        size (int): The desired number of variants for the model.
        
    Returns:
        A numpy matrix with variants, which satisfy the provided constrains. Each row represents one variant.
    """
    import itertools
    import pycosat
    from random import shuffle
    
    new_c = c.copy()
    
    sampling_method = str(config['Variants']['Sampling_Method'])
    Shuffle_Constraints = str(config['Variants']['Permutation_Method'])
    
    assert (Shuffle_Constraints in ["complete", "clauses", "no_permutation"] ), ("Options for Permutation_Method are: complete, clauses, no_permutation")
    assert (sampling_method in ["random", "feature-wise", "pair-wise", "neg-feature-wise","neg-pair-wise"]), ("Options for Sampling_Method are: random, feature-wise, neg-feature-wise, pair-wise, neg-pair-wise")
    
    sampling = {
            'feature-wise': lambda x, i, j: x.append([i]),
            'pair-wise': lambda x, i, j: x.extend([[i], [j]]),
            'neg-feature-wise': lambda x, i, j: x.append([-(i)]),
            'neg-pair-wise': lambda x, i, j: x.extend([[-i], [-j]])
        }.get(sampling_method)
    
    sol_collection = list()
    
    
    if Shuffle_Constraints == "no_permutation" and sampling_method == "random":
        solutions = list(itertools.islice(pycosat.itersolve(new_c), size))
        for elem in solutions:
            solution = transform2binary(elem)
            sol_collection.append(solution)
    else:
        for i in range(0,size):
            if Shuffle_Constraints == "clauses" or Shuffle_Constraints == "complete":
                shuffle(new_c) #shuffle the constraints
                if Shuffle_Constraints == "complete":
                    for constraint in new_c: #shuffle feature assignment in constraints
                        shuffle(constraint)
            c_copy = list(new_c)
            if sampling_method != "random":
                sampling(c_copy, i+1, (i+1)%(size)+1)

            solution = pycosat.solve(c_copy)
            if solution != "UNSAT":
                new_c.append([j * -1 for j in solution])
                solution = transform2binary(solution)
                sol_collection.append(solution)

    m_sol_list = np.asmatrix(sol_collection)
    
    return m_sol_list

#----------
# APPENDING INTERACTIONS TO VARIANTS
#----------

def append_interactions(v, f, i):
    """
    A function which check for each variant, if they satisfy the previously provided (or estimated) interactions.
    It does so by looking up the involved features for each interaction and checking if those features are set to 1 for
    the respective variant. If so, the program appends a 1 (interaction satisfied) to the variant,
    else it append a 0 (interaction not satisfied).
    
    Args:
        v (numpy matrix): All previously computed variants, which satisfy the provided constrains
        f (dict): All features with their names as keys and their values as values
        i (dict): All interactions with feature tuples as keys and their values as values
        
    Returns:
        A numpy matrix with variants and information about which interactions they satisfy.
        Each row represents one variants and its interactions information. 
    
    """

    valid_interaction = np.array([[1]])
                                                    
    def check_for_interaction(row):
        for elem in i.keys():
            valid_interaction[0,0] = 1
            tokens = elem.split("#")
            for feature in tokens:
                index = list(f.keys()).index(feature) - 1
                if row[0, index] == 0:
                    valid_interaction[0,0] = 0
                    break
            row = np.concatenate((row, valid_interaction), axis=1)#np.insert(row, -1, valid_interaction)
        return row

    v = np.apply_along_axis( check_for_interaction, axis=1, arr=v )                                             
    
    return v

#----------
# PERFORMANCE CALCULATION
#----------


def performance(v, f_and_i, len_f_and_i):
    """
    A function to calculate the fitness/cost (depending on the model's application area) of all previously computed variants.

    Args:
        v (numpy matrix): All previously computed variants with information about which interaction they satisfy
        f_and_i (numpy matrix): The provided or estimated values for all features and interactions

    Returns:
        An array of all variant fitnesses/costs. 
    """
    root = f_and_i[0,0]
    v = np.transpose(v)
    len_ratio = len_f_and_i / f_and_i.shape[1]
    f_and_i = np.delete(f_and_i, 0, 1)
    m_fitness = np.dot(f_and_i,v)
    if len_ratio != 1:
        m_fitness = m_fitness * len_ratio
    m_fitness = np.add(m_fitness,root)
    m_fitness = np.asarray(m_fitness)
    m_fitness = m_fitness.ravel()
               
    return m_fitness                                                 

#----------
# KERNEL DENSITY ESTIMATION
#----------

def kde(data, size):
    """
    A function which to perform a kernel density estimation.
    
    Args:
        data (list): All values, which are used to perform the kernel density estimation. The intial distribution.
        
    Returns:
        A tuple of lists which contain x-axis coordinates for plotting the results (index [0])
        and the estimated values for the estimated distribution (index [1])
    
    """
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV

    if str(config['Miscellaneous']['KDE_bandwidth']) != "auto":
        auto_bandwith = False
        try:
            kde_bandwidth = float(config['Miscellaneous']['KDE_bandwidth'])
        except:
            sys.exit("KDE_bandwidth must be float. Please check your configuration file!")
    else:
        auto_bandwith = True
    
    data = np.reshape(data, (-1, 1))
    
    #use grid search cross-validation to optimize the bandwidth
    if auto_bandwith:
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(data)
        bandwidth = (grid.best_estimator_.bandwidth)
    else:
        bandwidth = kde_bandwidth
    data_plot = np.linspace(data[data != 0].min(), data[data != 0].max(), size)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    log_dens = kde.score_samples(data_plot)
    
    return(data_plot, np.exp(log_dens))   
       
#----------
# ESTIMATE FEATURES
#----------

def estimation(old_data, new_data):
    """
    A function to estimate new values for the non-attributed variablitiy model.
    
    Args:
        old_data (dict): Features or Interactions, with their names as keys and their values as values.
        new_data (dict): Features or Interactions, with their names as keys, but without corresponding values.
        
    Returns: The new_data with estimated values for every key.  
    """
    import itertools
    
    values = list(old_data.values())
    
    #performance relevance for features. Top 20% of values
    value_range = max(values) - min(values)
    perf_rel_threshhold = min(values) + 0.8 * value_range
    amount_perf_rel = len([elem for elem in values if elem > perf_rel_threshhold])
    
    estimated_values = kde(values, len(new_data))    
   
    for key in new_data.keys():
        p = random.uniform(min(estimated_values[1]),max(estimated_values[1]))#Wert zw kleinster und größter Density
        selectors = [x >=p for x in estimated_values[1]]#alle Densities, die größer als p sind
        valid_densities = list(itertools.compress(estimated_values[0], selectors))#alle Werte, für die validen Densities
        index = random.randrange(len(valid_densities))
        new_data[key] = valid_densities[index][0]

    return new_data

#----------
# GENERATE INTERACTIONS
#----------

#check if interaction is true and false for at least one variant, 
#as well as if the interaction members are true and false in at least one variant
def check_interaction(c, f, random_features):
    """
    A function, which checks if an interaction (1) can occure in at least one variant but 
    (2) won't occure in every variant of a model given the provided list of constrains.
    
    Args:
        c (list): The model's constrains
        f (dict): The model's features
        random_features (list): The features, which were generated by the new_interactions function
        
    Returns:
        True, if the interaction can occure in at least one but not all variants
        False, if the interaction can't occure or if the features are dependent on eachother 
    """
    import pycosat
    
    constrains = list(c)
    
    for elem in random_features:
        index = list(f.keys()).index(elem)
        constrains.append([index])
    
    if pycosat.solve(constrains) == "UNSAT":
        return False
    
    constrains = list(c)
    index = list(-list(f.keys()).index(elem) for elem in random_features)
    constrains.append(index)
    
    if pycosat.solve(constrains) == "UNSAT":
        return False

    return True

def new_interactions(constraint_list, f, specs):
    """
    A function to generate new interactions between features. The generation is handled by threads. The amount of threads is dependend on the number of different interaction degrees.
    
    Args:
        constraint_list (list): The previously aquired list (of lists) with all constraints of a model.
        f (dict): The models features.
        specs (list): The amount of new interactions, followed by value pairs for ratio in percent and interaction degree, e.g. [100, 50, 2, 50, 3].
    
    Returns:
        a dictionary with the new interactions as keys.
    """
    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool
    import random
    
    legit_int = bool 
    total_amount = specs[0]
    interaction_ratio = list(specs[1::2])
    interaction_degree = list(specs[2::2])
    
    all_new_interactions = dict()
    splitted_new_interactions = dict()
    for elem in interaction_degree:
        splitted_new_interactions["dict"+str(elem)] = {}
        
    if str(config['Miscellaneous']['NumberOfThreads']) != "auto":
        number_of_threads = int(config['Miscellaneous']['NumberOfThreads'])
    else:
        number_of_threads = os.cpu_count()
    
     #some sweet, sweet error handling:
    assert (sum(interaction_ratio) == 100), ("The interaction ratios must sum up to 100. Currently they sum up to: ", sum(interaction_dist))
    
    def worker(amount):
        new_interactions = dict()
        for elem in range(len(interaction_degree)):
            if elem == len(interaction_degree)-1:
                amount_new_int = amount
            else:
                amount_new_int = round(amount / 100.0 * interaction_ratio[elem])
                amount = amount - amount_new_int
            
            while ( amount_new_int > len(splitted_new_interactions["dict"+str(interaction_degree[elem])])):
                legit_int = False
                while legit_int == False:
                    random_feature = list(np.random.choice(list(f.keys())[1:], interaction_degree[elem]))
                    if check_interaction(constraint_list, f, random_feature):
                        legit_int = True
                        random_feature = sorted(random_feature)
                        interaction = ""
                        for i in random_feature:
                            interaction = interaction + str(i) + "#" 
                        interaction = interaction[:-1]
                        splitted_new_interactions["dict"+str(interaction_degree[elem])][interaction] = ""
            #new_interactions.update(new_int_degree_subdict)
        #return new_interactions
                    
    pool = ThreadPool()
    l = [total_amount] * (number_of_threads)
    
    pool.map(worker, l)
    
    for elem in range(len(interaction_degree)):
        desired_amount = total_amount *  interaction_ratio[elem] / 100
        while ( desired_amount < len(splitted_new_interactions["dict"+str(interaction_degree[elem])])):
            rchoice= random.choice(list(splitted_new_interactions["dict"+str(interaction_degree[elem])].keys()))
            del splitted_new_interactions["dict"+str(interaction_degree[elem])][rchoice]  
        all_new_interactions.update(splitted_new_interactions["dict"+str(interaction_degree[elem])])

    pool.close() 
    pool.join()
    
    print("Finished with creating interactions")
    
    return all_new_interactions
    
#----------
# CONCATENATE FEATURES AND INTERACTIONS
#----------

def concatenate(f, i):
    """
    A function to convert two lists into arrays and concatenate them.
    
    Args:
        f (list): Values for all features.
        i (list): Values for all interactions.
        
    Returns:
        An array with the concatenated feature and interaction values.
    
    """
    m_f = np.asarray(f)
    m_i = np.asarray(i)    
    f_and_i = np.append(m_f, m_i)
    f_and_i = np.asmatrix(f_and_i)
    
    return f_and_i


#----------
# COMPUTE SIMILARITIES
#----------    

def compute_similarities(data, e_data):                                                   
    """
    A function to compute the similarity between two given datasets.
    Currenty available similarity measures are: Anderson-Darling-Test, Pearson correlation coefficient, euclidean distance
    
    Args:
        data (list): data from the attributed model.
        e_data (list): estimated data from the non-attributed model.
        
    Returns:
        A float value as similarity measure between 0 and 1. 
        0 means no similarity at all; 1 means a strong similary between the two datasets. 
    """
    import itertools    
    import scipy.stats as sps
    import scipy.spatial as spsp
    from itertools import groupby

    np.warnings.filterwarnings('ignore')
    sim_results = list() 
    
    sim_measures = list(config['NSGAII']['Similarity_Measures'].split(", "))
    
    if "AD" in sim_measures:
        try:
            AD_result = sps.anderson_ksamp([data, e_data])
            if AD_result[-1] > 1:
                pass
            else:
                sim_results.append(AD_result[-1])
        except Exception:
            pass
        
    if "PCC" in sim_measures:
        PCC_result = sps.pearsonr(data, e_data)
        sim_results.append(abs(PCC_result[0]))
                
    if "ED" in sim_measures:
        ED_result = spsp.distance.euclidean(data, e_data)
        ED_result = 1 / (ED_result + 1)
        sim_results.append(ED_result)
        
    if "KS" in sim_measures:
        KS_result = sps.ks_2samp(data, e_data)
        sim_results.append(1- min(KS_result[0], 0))                                                   
 
    mean_similarity = sum(sim_results)/len(sim_results)

    return mean_similarity

#----------
# PLOTTING
#----------

def plotting(avm, vm, filepath):
    """
    A function which takes the given and estimated feature, interaction and fitness values and compares them with them help of plot diagrams
    
    Args:
        avm (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the attributed model's variants
        vm (lsit): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the non-attributed model's variants
        filepath (str): path to the results folder
              
    """
    import matplotlib.pyplot as plt
    
    #instantiating stuff
    try:
        amount_bins = int(config['Miscellaneous']['NumberOfBins'])
    except:
        sys.exit("NumberOfBins must be an integer. Please check your configuration file!")
    
    #PREPARE THE DATA   
    #real feature values    
    values_rF = list(avm[0].values())
    values_kde_F = kde(values_rF, len(avm[0]))
    values_eF = vm[0]
    bin_F = np.linspace(min(values_rF), max(values_rF), amount_bins)
    bin_eF = np.linspace(min(values_eF), max(values_eF), amount_bins)
    
    #real variant fitness values    
    values_rV = avm[-1]
    values_kde_V = kde(values_rV, vm[-1].size)
    values_eV = vm[-1]
    bin_V = np.linspace(values_rV[values_rV != 0].min(), values_rV[values_rV != 0].max(), amount_bins)
    bin_eV = np.linspace(values_eV[values_eV != 0].min(), values_eV[values_eV != 0].max(), amount_bins)    
    
    if str(config['AttributedModel']['With_Interactions']) == "True":
        #real interaction values    
        values_rI =list(avm[1].values())
        values_kde_I = kde(values_rI, len(vm[1]))
        values_eI = vm[1]
        bin_I = np.linspace(min(values_rI), max(values_rI), amount_bins)
        bin_eI = np.linspace(min(values_eI), max(values_eI), amount_bins)        
        
    #INITIALIZE PLOT    
    if str(config['AttributedModel']['With_Interactions']) == "True":
        fig = plt.figure(figsize=(30,30))
        rF = fig.add_subplot(331)
        kdeF = fig.add_subplot(332)
        eF = fig.add_subplot(333)

        rI = fig.add_subplot(334)
        kdeI = fig.add_subplot(335)
        eI = fig.add_subplot(336)

        rV = fig.add_subplot(337)
        kdeV = fig.add_subplot(338)
        eV = fig.add_subplot(339)
        
    if str(config['AttributedModel']['With_Interactions']) == "False":
        fig = plt.figure(figsize=(30,20))
        rF = fig.add_subplot(231)
        kdeF = fig.add_subplot(232)
        eF = fig.add_subplot(233)

        rV = fig.add_subplot(234)
        kdeV = fig.add_subplot(235)
        eV = fig.add_subplot(236) 
    
    #PLOT THE DATA
    rF.set_title("real Features")
    rF.hist(values_rF, bins=bin_F,fc="grey", density=True)
    rF.set_xlabel('value')
    rF.set_ylabel('density')
    
    kdeF.set_title("kde Features")
    kdeF.plot(values_kde_F[0][:, 0], values_kde_F[1], linewidth=2, color="grey",alpha=1)
    kdeF.hist(values_rF, bins=bin_F, density=True, fc="black", alpha=0.1)
    kdeF.set_xlabel('value')
    kdeF.set_ylabel('density')
    
    eF.set_title("estimated Features")
    eF.hist(values_eF, bins=bin_eF, density=True,fc="grey")
    eF.hist(values_rF, bins=bin_F, density=True,fc="black", alpha=0.1)
    eF.set_xlabel('value')
    eF.set_ylabel('density')
    
    #######
    if str(config['AttributedModel']['With_Interactions']) == "True":
        rI.set_title("real Interactions")
        rI.hist(values_rI, bins=bin_I, density=False,fc="grey", weights=np.ones(len(values_rI)) / len(values_rI))
        rI.set_xlabel('value')
        rI.set_ylabel('density')

        kdeI.set_title("kde Interactions")
        kdeI.plot(values_kde_I[0][:, 0], values_kde_I[1], linewidth=2,color="grey", alpha=1)
        kdeI.hist(values_rI, bins=bin_I, fc='black', alpha=0.1, density=True)
        kdeI.set_xlabel('value')
        kdeI.set_ylabel('density')

        eI.set_title("estimated Interactions")
        eI.hist(values_eI, bins=bin_eI, density=False,fc="grey", weights=np.ones(len(values_eI)) / len(values_eI))
        eI.hist(values_rI, bins=bin_I, density=False,fc="black", weights=np.ones(len(values_rI)) / len(values_rI), alpha=0.1)
        eI.set_xlabel('value')
        eI.set_ylabel('density')
    
    ######
    
    rV.set_title("real Variants")
    rV.hist(values_rV, bins=bin_V, density=False, fc="grey",weights=np.divide(1, values_rV))
    rV.set_xlabel('value')
    rV.set_ylabel('density')
    
    kdeV.set_title("kde Variants")
    kdeV.plot(values_kde_V[0][:, 0], values_kde_V[1], linewidth=2,color="grey", alpha=1)
    kdeV.hist(values_rV, bins=bin_V, fc='black', alpha=0.1, density=True)
    kdeV.set_xlabel('value')
    kdeV.set_ylabel('density')
    
    eV.set_title("estimated Variants")
    eV.hist(values_eV, bins=bin_eV, density=False,fc="grey", weights=np.divide(1, values_eV))
    eV.hist(values_rV, bins=bin_V, density=False,fc="black", weights=np.divide(1, values_rV), alpha=0.1)
    eV.set_xlabel('value')
    eV.set_ylabel('density')
    
    #save the plot
    plt.savefig(filepath + 'plots.png', bbox_inches='tight')
    plt.savefig(filepath +'plots.pdf', bbox_inches='tight')
    
    #show the plot
    #plt.show()
    plt.clf()
    plt.close()

#----------
# HELPER FUNCTIONS
#----------
def transform2binary(sol):
    """
    A function which takes a valid variant, consisting of positive and negative integers and transforming it into binary values
    Args:
        sol (list): A list that contains one valid variant, represented by positve and negative integers
        
    Returns:
        A list that contains the valid variants transformed into binary, where negative integers are now represented as 0 and positive integers as 1
    """
    sol = sorted(sol, key=abs)
    for index, elem in enumerate(sol):
        if float(elem) < 0:
            sol[index] = 0
        else:
            sol[index] = 1
    return sol


#=======================
#
# NSGA-II STARTS HERE
#
#=======================

#----------
# DETERMINE FRONT RANKS
#----------
#returns all data with assigned Front Ranks as list: [[Rank1], [Rank2], ... , [RankN]]
def front_rank_assignment(P):# works :D
    """
    A function to compute the ranks of a population given a list of objectives.
    
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        
    Returns:
        R: A list of lists with the population ordered by rank, with the following schema: [[Rank1], [Rank2], ... , [RankN]]
     """
    P_copy = P.copy()
    R = list()
    i = 0
    
    while (len(P_copy) >=1):
        R.append([])
        R[i].extend(non_dominated_search(P_copy))
        for elem in R[i]:
            del P_copy[elem]
        R[i] = [list(elem) for elem in R[i]]
        i = i+1

    return R

#returns Pareto Front of Input
def non_dominated_search(P): #def non_dominated_search(P, O):# works :D
    """
    A function to compute the front rank of a population given a list of objectives.
    
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        
    Returns:
        Front: A list of all candidate solutions, which are in the front rank i.e. which aren't dominated by other candidate solutions.
    """
    Keys = P.keys()
    Front = [x for x in Keys]
    TrueFront = []
    FrontBool = [0]*len(Front)

    for elem in Front:
        Front_copy = [x for x in Front if x != elem and FrontBool[Front.index(x)] !=1]
        for individual in Front_copy:
            if pareto_dominates(P, individual, elem):
                FrontBool[Front.index(elem)] = 1
                #try:
                    #del Front[Front.index(elem)]
                #except:
                    #pass
            elif pareto_dominates(P, elem, individual):
                FrontBool[Front.index(individual)] = 1
                #del Front[Front.index(individual)]
                
    for i in range(0,len(Front)):
        if FrontBool[i] == 0:
            TrueFront.append(Front[i])

    return TrueFront

def pareto_dominates(P, A, B): #works :D
    """
    A function which checks, if candidate solution A dominates candidate solution B.
    
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        A (list): The candidate solution A
        B (list): The candidate solution B
         
    Returns:
        dominate: True, if candidate solution A dominates candidate solution B. False, otherwise. 
    """
    dominate = False
    for obj in range(len(list(P.values())[0])):
        if  P[A][obj] > P[B][obj] :
            dominate = True
        elif P[B][obj] > P[A][obj] :
            return False
    return dominate

#Checks if Element A dominates Element B

#----------
# DETERMINE SPARSITY
#----------
def Sparsity(F, O):
    """
    A function which calculates the sparsity of a rank of candidate solutions.
    
    Args:
        F (list): A rank of candidate solutions.
        O (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        
    Returns:
        F: The list of candindate solutions ordered by their sparsity. 
    """
    import operator
    
    Sparsity = list()
    Objectives = list()
    
    for elem in range(0, len(F)):
        Sparsity.append(0)
        Objectives.append(O[tuple(F[elem])])

    for obj in range(0, len(Objectives[0])):
        O = [x[obj] for x in Objectives]
        O.sort(reverse=True)
        F_sorted_by_obj = [x for _,x in sorted(zip(O,F), reverse=True)]
        O_sorted_by_obj = sorted(Objectives, key=operator.itemgetter(obj), reverse=True)  
        O_range = O[-1] - O[0]
        if O_range == 0:
            O_range = 1

        Sparsity[0] = np.inf #element with lowest obj values gets infinite sparsity
        Sparsity[-1] = np.inf # same with highest obj values
        
        for j in range(1, len(F_sorted_by_obj)-1):
            Sparsity[j] = Sparsity[j] + ((O_sorted_by_obj[j+1][obj] - O_sorted_by_obj[j-1][obj])/O_range)
    
    F = [x for _,x in sorted(zip(Sparsity,F_sorted_by_obj), reverse=True)]
    return F

#----------
# GENERATE NEW POPULATION
#----------
def breed(P, pop_size, f_i_range):
    """
    A function which takes the population of candidate solutions and uses it to generate a new set of candidate solutions.
    
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        pop_size (int): The number of candidate solutions in the population.
        f_i_range (float): The value range of the features and - if applicable - interactions
        
    Returns:
        Offspring: A numpy matrix with the generated, new candidate solutions. The number of new candidate solutions is defined by the pop_size
    """
    Offspring = list()
    choice = str(config['NSGAII']['Selection_Algorithm'])
    choice_CO = str(config['NSGAII']['Recombination_Algorithm'])
    
    for i in range(0, pop_size, 2):
        Parent_A, Parent_B = {
            'tournament_selection':  tournament_selection,
            'fitness_proportionate_selection': fitness_proportionate_selection,
            'stochastic_universal_sampling': stochastic_universal_sampling
        }.get(choice)(P)
        #Child_A, Child_B = Crossover(Parent_A, Parent_B, f_i_range)
        
        Child_A, Child_B = {
            'Line_Recombination': line_recombination,
            'simulated_binary_CO': sim_binary_CO,      
        }.get(choice_CO)(Parent_A, Parent_B, f_i_range)
        
        
        Offspring.append(Child_A)
        Offspring.append(Child_B)
    
    while len(Offspring) != pop_size:
        Offspring = Offspring[:-1]
    
    return np.asmatrix(Offspring)

#choose element depending on Front Rank and Sparsity
def tournament_selection(P):
    """
    A function which picks a number of random candidate solutions from the popultion and computes, which is the best one regarding rank and sparsity.
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        
    Returns:
        Best: A candidate solution with a good rank and sparsity.
    """       
    R = front_rank_assignment(P)
    Pop = [list(x) for x in list(P.keys())]
    S = Sparsity(Pop,P)
    t = 2

    BestA = random.choice(Pop)
    BestB = random.choice(Pop)
    Bests = [BestA, BestB]
    for Best in Bests:
        for i in range(1, t):
            Next = random.choice(Pop)
            Next_FR = findItem(R, Next)[0]
            Best_FR = findItem(R, Best)[0]
            if Next_FR < Best_FR:
                Best = Next
            elif Next_FR == Best_FR:
                if S.index(Next) < S.index(Best):
                    Best = Next
        
    return BestA, BestB

def fitness_proportionate_selection(P):
    """
    A function which picks a candidate solution with a propability proportionate to their fitness value.
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
           
    Returns:
        Best: A candidate solution
    """       
    Total_O= list()
    Parents = list()
    
    for obj in list(P.values()):
        Total_O.append(sum(obj))
    
    if sum(Total_O) == 0:
        Total_O = [elem+1 for elem in Total_O]
        
    for i in range(1, len(Total_O)):
        Total_O[i] = Total_O[i]+Total_O[i-1]
        
    for j in range(2):
        n = random.uniform(0, Total_O[len(Total_O)-1])
        for i in range(1, len(Total_O)):
            if Total_O[i-1] < n <= Total_O[i]:
                Parents.append(list(list(P.keys())[i])) 
        if len(Parents) == j:
            Parents.append(list(list(P.keys())[0]))
    
    return Parents[0], Parents[1]

def stochastic_universal_sampling(P):
    """
    A function which picks a candidate solution with a propability proportionate to their fitness value. If a solutio
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
           
    Returns:
        Best: A candidat solution
    """ 
    Total_O= list()
    Parents = list()
    
    for obj in list(P.values()):
        Total_O.append(sum(obj))
    
    if sum(Total_O) == 0:
        Total_O = [elem+1 for elem in Total_O]
        
    for i in range(1, len(Total_O)):
        Total_O[i] = Total_O[i]+Total_O[i-1]
    
    n = random.uniform(0, (Total_O[-1] / 2))        
    for j in range(2):
        i = 0
        while Total_O[i] < n:
            i = i+1
        n = n + (Total_O[-1] / 2)
        Parents.append(list(list(P.keys())[i])) 
    
    return Parents[0], Parents[1]

def line_recombination(A, B, f_i_range):
    """
    A function, which takes two candidate solutions and crosses them over to generate two new candidate solutions.
    
    Args:
        A (list): A candidate solution.
        B (list): A candidate solution.
        f_i_range (float): The value range of the features and - if applicable - interactions
        
    Returns:
        Mutation(A_new), Mutation(B_new): Two lists, with the new candidate solutions.
    """     
    alpha = np.random.rand(len(A),1).T
    
    A_new = np.add(np.multiply(alpha, np.asmatrix(A)), np.multiply(np.subtract(1,alpha), np.asmatrix(B)))
    B_new = np.add(np.multiply(alpha, np.asmatrix(B)), np.multiply(np.subtract(1,alpha), np.asmatrix(A)))
    A_new = A_new.tolist()[0]
    B_new = B_new.tolist()[0]
    
    return Mutation(A_new, 0, 0.01*f_i_range), Mutation(B_new, 0, 0.01*f_i_range)

def sim_binary_CO(A, B, f_i_range):
    """
    A function, which takes two candidate solutions and crosses them over to generate two new candidate solutions.
    
    Args:
        A (list): A candidate solution.
        B (list): A candidate solution.
        f_i_range (float): The value range of the features and - if applicable - interactions
        
    Returns:
        Mutation(A_new), Mutation(B_new): Two lists, with the new candidate solutions.
    """ 
    eta = 2
    rand = random.random()
    A_new = [0] * len(A)
    B_new = [0] * len(B)
    if rand <= 0.5:
        beta = math.pow((2.0 * rand),(1/(eta+1)))
    else:
        beta = math.pow((1/(2.0 * (1-rand))),(1/(eta+1)))
        
    for i in range(0, len(A)):
        A_new[i] = 0.5 * (((1 + beta) * A[i]) + ((1 - beta) * B[i]))
        B_new[i] = 0.5 * (((1 - beta) * A[i]) + ((1 + beta) * B[i]))
        
    return Mutation(A_new, 0, 0.01*f_i_range), Mutation(B_new, 0, 0.01*f_i_range)

def Mutation(A, mean, f_i_range):#intermediate recombination
    """
    A function, which takes a candidate solution and mutates them to generate a new candidate solution.
    
    Args:
        A (list): A candidate solution.
        f_i_range (float): The value range of the features and - if applicable - interactions
        
    Returns:
        A: A list with the new, mutated candidate solution.
    """
    A_mutated = list(A)
    p = 1
    mu = mean
    sigma = f_i_range#standard deviation
    
    for i in range(0, len(A_mutated)):
        if p >= np.random.random_sample():
            while True:
                s = np.random.normal(mu, sigma, 1)[0]
                break
            A_mutated[i] = A_mutated[i] + s

    return A_mutated

def ComparableFrontFitness(P):
    """
    A function, which calculates the mean total fitness of a list of candidate solutions 
    
    Args:
        P (dict): A dictioray of candidate solutions.
        
    Returns:
        A float value, which represent the mean total similarity of a list of candidate solutions.
    """
    Total_O= list()
    
    for obj in list(P.values()):
        Total_O.append(sum(obj))
    
    Max_Fitness = max(Total_O)
    Mean_Fitness = sum(Total_O)/len(Total_O)
  
    return Max_Fitness, Mean_Fitness

def define_results(BestFront_dict):
    assert (str(config['Miscellaneous']['ResultsToBeSaved']) in ["all" , "overall-best" , "custom"] ), ("Options for ResultsToBeSaved are: all, overall-best, custom")
    if str(config['Miscellaneous']['ResultsToBeSaved']) == "all":
        BestFront = [list(x) for x in list(BestFront_dict.keys())]

    if str(config['Miscellaneous']['ResultsToBeSaved']) == "overall-best":
        obj_values = list(BestFront_dict.values())
        maximum = obj_values[0]
        for elem in obj_values:
            if sum(elem) > sum(maximum):
                maximum = elem   
        for solution, values in BestFront_dict.items():
            if values == maximum:
                BestFront = [solution]

    if str(config['Miscellaneous']['ResultsToBeSaved']) == "custom":
        Custom_Specs = config['Miscellaneous']['ResultsCustomSpecs'].split(", ")
        Custom_Specs = list(map(float, Custom_Specs))

        def weighted_sum (obj, specs):
            weighted_sum = 0
            for i in range(0, len(obj)):
                weighted_sum = weighted_sum + (obj(i)*specs(i))    
            return weighted_sum

        obj_values = list(BestFront_dict.values())
        maximum = obj_values[0]
        for elem in obj_values:
            if weighted_sum(elem) > weighted_sum(maximum):
                maximum = elem
        for solution, values in BestFront_dict.items():
            if values == maximum:
                BestFront = [solution]
                
    return BestFront

#----------
# HELP FUNCTIONS
#----------       

def findItem(theList, item):
    return [(ind, theList[ind].index(item)) for ind in range(len(theList)) if item in theList[ind]]

#----------
# MAIN FUNCTION
#---------- 
def nsga2(feature_and_optional_interaction, v, e_v):
    """
    A function, which takes the given and estimated feature (and optional interaction) values, 
    as well as the lists of variants of the attributed and non-attributed model and optimizes 
    
    Args:
        feature_and_optional_interaction (list of lists): The first and second sublist contain the feature values of the attributed and non-attributed model respectively. 
        The third and fourth sublis contain the interaction values of the attributed and non-attributed model respectively, if provided.
        v (numpy matrix): The previously computed variants of the attributed model
        e_v (numpy matrix): The previously computed variants of the non-attributed model
    Returns:
        BestFront: A list with feature (and optional interaction) values e.i candidate solutions, which have an optimal similarity to the value distribution of the atributed model.
    """
    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool
    from functools import partial
    import itertools
    
    try:
        pop_size = int(config['NSGAII']['Population_Size'])
    except:
        sys.exit("Population_Size must be an integer. Please check your configuration file!")
    
    archive_size = pop_size
    max_gen = np.inf
    gen_no = 0
    interaction_option = False
    converge_counter = 0
    Old_Best = 0.0
    Old_Mean = 0.0
    
    if str(config['NSGAII']['Maximum_Generations']) != "auto":
        try:
            max_gen = int(config['NSGAII']['Maximum_Generations'])
        except:
            sys.exit("Maximum_Generations must be an integer. Please check your configuration file!")
      
    if str(config['AttributedModel']['With_Interactions']) == "True":
        interaction_option = True 
    
    if str(config['Miscellaneous']['NumberOfThreads']) != "auto":
        number_of_threads = int(config['Miscellaneous']['NumberOfThreads'])
    else:
        number_of_threads = os.cpu_count()
     
    #get list of features for attributed and non-attributed model
    feature_list = feature_and_optional_interaction[0]
    feature_list_for_estimation = feature_and_optional_interaction[1]
    feature_list_pure = list(feature_list.values())
    f_min_max = max(feature_list_pure) - min(feature_list_pure)
    
    #if provided: get interactions for attributed and non-attributed model
    if interaction_option:
        interactions_list = feature_and_optional_interaction[2]
        interactions_list_for_estimation = feature_and_optional_interaction[3]
        interactions_list_pure = list(interactions_list.values())
        i_min_max = max(interactions_list_pure) - min(interactions_list_pure)
        f_i_range = (f_min_max + i_min_max)/2
        f_and_i = concatenate(feature_list_pure, interactions_list_pure)
    else:
        f_i_range = f_min_max
        f_and_i = np.asmatrix(feature_list_pure)
    
    #build initial population
    for i in range(pop_size):
        #get initial values for estimated features
        e_feature_list = estimation(feature_list, feature_list_for_estimation)
        e_feature_list_pure = list(e_feature_list.values())
        
        #if provided: get initial values for estimated interactions
        if interaction_option:
            e_interaction_list = estimation(interactions_list, interactions_list_for_estimation)
            e_interactions_list_pure = list(e_interaction_list.values())
            e_f_and_i = concatenate(e_feature_list_pure, e_interactions_list_pure)
        else:
            e_f_and_i = np.asmatrix(e_feature_list_pure)
            
        if 'P' in locals():
            P = np.vstack([P, e_f_and_i])
        else:
            P = np.matrix(e_f_and_i)
            
    upsize_f_kde = kde(feature_list_pure, len(e_feature_list_pure))
    upsize_f = list()
    for elem in e_feature_list_pure:
        p = random.uniform(min(upsize_f_kde[1]),max(upsize_f_kde[1]))#Wert zw kleinster und größter Density
        selectors = [x >=p for x in upsize_f_kde[1]]#alle Densities, die größer als p sind
        valid_densities = list(itertools.compress(upsize_f_kde[0], selectors))#alle Werte, für die validen Densities
        index = random.randrange(len(valid_densities))
        upsize_f.append(valid_densities[index][0])
    feature_list_pure = list(upsize_f)
    
    if interaction_option:
        upsize_i_kde = kde(interactions_list_pure, len(e_interactions_list_pure))
        upsize_i = list()
        for elem in e_interactions_list_pure:
            p = random.uniform(min(upsize_i_kde[1]),max(upsize_i_kde[1]))#Wert zw kleinster und größter Density
            selectors = [x >=p for x in upsize_i_kde[1]]#alle Densities, die größer als p sind
            valid_densities = list(itertools.compress(upsize_i_kde[0], selectors))#alle Werte, für die validen Densities
            index = random.randrange(len(valid_densities))
            upsize_i.append(valid_densities[index][0])
        interactions_list_pure = list(upsize_i)
            
    #initialize archive
    A = list()
   
    while(gen_no<max_gen and converge_counter<3):
        print("Generation:", gen_no+1)
        
        if len(A)!=0:
            P = np.append(P, np.asarray(A), axis=0)
                
        #compute objectives (similarities)
        
        def p_similarity_worker(s_f):
            Pop_Fitness = dict()
            for i in range(s_f[0], s_f[1]):
                F_Fitness = compute_similarities(feature_list_pure, P[i, 0:len(e_feature_list_pure)].flatten().tolist()[0])
                V_Fitness = compute_similarities(performance(v, f_and_i, f_and_i.shape[1]),performance(e_v, P[i], f_and_i.shape[1]))
                if interaction_option:
                    I_Fitness = compute_similarities(interactions_list_pure, P[i, len(e_feature_list_pure):].flatten().tolist()[0])
                    All_Fitnesses = [F_Fitness, I_Fitness, V_Fitness]
                else:
                    All_Fitnesses = [F_Fitness, V_Fitness]
                Pop_Fitness[tuple(P[i,0:].flatten().tolist()[0])] = All_Fitnesses
            return Pop_Fitness
        
        pool = ThreadPool()
        intervals = [int(round(x*(P.shape[0]/number_of_threads))) for x in range(number_of_threads+1)]
        s_f = list()
        for s, f in zip(intervals[:-1], intervals[1:]):
            s_f.append([s,f])

        pool_fitness = pool.map(p_similarity_worker, s_f)
        pool.close() 
        pool.join()
        
        Pop_Fitness = dict()
        for elem in pool_fitness:
            Pop_Fitness.update(elem)
        
        R = front_rank_assignment(Pop_Fitness)
        BestFront = dict((tuple(elem), Pop_Fitness[tuple(elem)]) for elem in R[0])
        
        Best, Mean = ComparableFrontFitness(BestFront)
        if Best - Old_Best < 0.0 or Mean - Old_Mean == 0:
            converge_counter = converge_counter +1
        else:
            converge_counter = 0
        Old_Best = Best
        Old_Mean = Mean
        
        del A[:]
        
        for i in range(0,len(R)):
            if len(A) + len(R[i]) >= archive_size:
                S = Sparsity(R[i], Pop_Fitness)
                A.extend(S[:(archive_size - len(A))])
                break
            else:
                A.extend(R[i])
        
        A_Fitness = dict()
        for elem in A:
            A_Fitness[tuple(elem[0:])] = Pop_Fitness[tuple(elem[0:])]  

        P = breed(A_Fitness, pop_size, f_i_range)
        gen_no = gen_no +1
    return BestFront

#=======================
#
# NSGA-II FOR AVM MODIFICATION
#
#=======================

#----------
# FITNESS FUNCTIONS
#---------- 
def compute_fulfilled_objectives(new_model, model):
    #Get selected change operations and their probability
    Change_Operations = config['Scope_for_Changes']['Change_Operations'].split(", ")
    Change_Probs = list()
    for elem in Change_Operations:
        Change_Probs.append(float(config[str(elem)]['Probability']))   
    Relevance_Treshhold = float(config['Scope_for_Changes']['Relevance_Treshhold'])    
    
    #get elements of dictionary, which where objected to modification
    if config['Scope_for_Changes']['Change_Feature'] == "most_influential":
        feature_dict = most_influential(model[0,0], Relevance_Treshhold)
        new_feature_dict = most_influential(new_model[0,0], Relevance_Treshhold)
    else:
        feature_dict = model[0,0]
        new_feature_dict = new_model[0,0]
    model_dict = dict(feature_dict)
    new_model_dict = dict(new_feature_dict)
        
    if config['Model'].getboolean('With_Interactions') == True:
        if config['Scope_for_Changes']['Change_Interaction'] == "most_influential":
            interactions_dict= most_influential(model[0,1], Relevance_Treshhold)
            new_interactions_dict = most_influential(new_model[0,1], Relevance_Treshhold)
            model_dict.update(most_influential(model[0,1], Relevance_Treshhold))
            new_model_dict.update(most_influential(new_model[0,1], Relevance_Treshhold))
        else:
            interactions_dict = model[0,1]
            new_interactions_dict = new_model[0,1]
        model_dict.update(interactions_dict)
        new_model_dict.update(new_interactions_dict) 
    
    #calculate fitness for all operations
    all_fitness = list()
    for elem in range(len(Change_Operations)): 
            fitness = {
            'Noise_small':  noise_small_objective,
            'Noise_big': noise_big_objective,
            'Linear_Transformation': linear_transformation_objective,
            'Negation': negation_objective
            }.get(Change_Operations[elem])(new_model_dict, model_dict, Change_Probs[elem])
            
            all_fitness.append(fitness)
            
    if config['Scope_for_Changes']['Change_Feature'] != "none":
        try:
            f_perc = float(config['Scope_for_Changes']['Change_Feature_percentage'])
        except:
            sys.exit("Change_Feature_percentage must be a float. Please check your configuration file!")
        fitness = feature_objective(new_feature_dict, feature_dict, f_perc)
        all_fitness.append(fitness)
    if config['Model'].getboolean('With_Interactions') == True and \
    config['Scope_for_Changes']['Change_Interaction'] != "none":
        try:
            i_perc = float(config['Scope_for_Changes']['Change_Interaction_percentage'])
        except:
            sys.exit("Change_Interaction_percentage must be a float. Please check your configuration file!")
        fitness = interactions_objective(new_interactions_dict, interactions_dict, i_perc)
        all_fitness.append(fitness)
    return all_fitness
    
def feature_objective(new_model_dict, model_dict, p):
    differences = 0
    
    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] != model_dict[elem]:
            differences = differences +1    
    share_of_change = differences / len(list(model_dict.keys()))
    feature_fitness = 1 - abs(p - share_of_change)
    return feature_fitness

def interactions_objective(new_model_dict, model_dict, p):
    differences = 0
    
    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] != model_dict[elem]:
            differences = differences +1      
    share_of_change = differences / len(list(model_dict.keys()))
    interactions_fitness = 1 - abs(p - share_of_change)
    return interactions_fitness
    
    
def noise_small_objective(new_model_dict, model_dict, p):
    sigma = float(config['Noise_small']['Standard_deviation'])
    noise_fitness = noise_objective(new_model_dict, model_dict, p, sigma)   
    return noise_fitness

def noise_big_objective(new_model_dict, model_dict, p):
    sigma = float(config['Noise_big']['Standard_deviation'])
    noise_fitness = noise_objective(new_model_dict, model_dict, p, sigma)
    return noise_fitness

def noise_objective(new_model_dict, model_dict, p, sigma):
    differences = 0
    
    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] != model_dict[elem] and \
        (new_model_dict[elem] < model_dict[elem] + (model_dict[elem] * sigma) or \
         new_model_dict[elem] > model_dict[elem] + (model_dict[elem] * sigma)):
            differences = differences +1
    
    share_of_change = differences / len(list(model_dict.keys()))
    noise_fitness = 1 - abs(p - share_of_change)

    return noise_fitness

def linear_transformation_objective(new_model_dict, model_dict, p):
    differences = 0
    operation = str(config['Linear_Transformation']['Operation'])
    operand = float(config['Linear_Transformation']['Operand'])
    
    transform = {
            'addition': lambda x: x + x *operand,
            'substraction': lambda x: x - x *operand,
            'multiplication': lambda x: x *( x *operand),
            'division' :lambda x: x /( x *operand)
        }.get(operation)
       
    for elem in list(new_model_dict.keys()):
        new_elem = transform(model_dict[elem])
        if new_model_dict[elem] != model_dict[elem] and \
        (new_elem == new_model_dict[elem]):
            differences = differences +1      
    share_of_change = differences / len(list(model_dict.keys()))
    lin_trans_fitness = 1 - abs(p - share_of_change)

    return lin_trans_fitness

def negation_objective(new_model_dict, model_dict, p):
    differences = 0
    
    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] == model_dict[elem]*-1: 
            differences = differences +1       
    share_of_change = differences / len(list(model_dict.keys()))
    negation_fitness = 1 - abs(p - share_of_change)

    return negation_fitness

 #----------
# GENERATE NEW POPULATION
#----------
def breed_KT(P, P_dict, pop_size):
    """
    A function which takes the population of candidate solutions and uses it to generate a new set of candidate solutions.
    
    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        P_dict(dict): A dictionary which containts the current population as keys and the current population split into features and (if applicable) interactions as dictionaries, where the keys are the names and the values are the values
        pop_size (int): The number of candidate solutions in the population.
        
    Returns:
        Offspring: A numpy matrix with the generated, new candidate solutions. The number of new candidate solutions is defined by the pop_size
    """
    Offspring = list()
    choice_S = str(config['NSGAII']['Selection_Algorithm'])
    choice_CO = str(config['NSGAII']['Recombination_Algorithm'])
    interaction_option = False
    
    if str(config['Model']['With_Interactions']) == "True":
        interaction_option = True 
    
    for i in range(0, pop_size, 4):
        Parent_A, Parent_B = {
            'tournament_selection':  tournament_selection,
            'fitness_proportionate_selection': fitness_proportionate_selection,
            'stochastic_universal_sampling': stochastic_universal_sampling
        }.get(choice_S)(P)
        Child_A, Child_B = {
            'one_point_CO': one_point_CO,
            'two_point_CO': two_point_CO,
            'universal_CO': universal_CO            
        }.get(choice_CO)(P_dict[tuple(Parent_A)], P_dict[tuple(Parent_B)], interaction_option)

        Offspring.append(Child_A)
        Offspring.append(Child_B)
    
    while len(Offspring) > pop_size/2:
        Offspring = Offspring[:-1]
    
    return np.asmatrix(Offspring) 

def one_point_CO(A, B, interaction_option):
    from copy import deepcopy
    if interaction_option:
        f_list = list(A[0].keys())
        i_list = list(A[1].keys())
    else:
        f_list = list(A[0].keys())
    new_A = deepcopy(A)
    new_B = deepcopy(B)
    
    c =  np.random.randint(len(A[0]))
    for elem in range(0, c):
        new_A[0].update({f_list[elem]: B[0][f_list[elem]]})   
        new_B[0].update({f_list[elem]: A[0][f_list[elem]]})    
    
    if interaction_option:
        c = np.random.randint(len(A[1]))
        for elem in range(0, c):
            new_A[1].update({i_list[elem]: B[1][i_list[elem]]})   
            new_B[1].update({i_list[elem]: A[1][i_list[elem]]})   

        return (new_A[0], new_A[1]), (new_B[0], new_B[1])
    return [new_A[0]], [new_B[0]]

def two_point_CO(A, B, interaction_option):
    from copy import deepcopy
    if interaction_option:
        f_list = list(A[0].keys())
        i_list = list(A[1].keys())
    else:
        f_list = list(A[0].keys())
    new_A = deepcopy(A)
    new_B = deepcopy(B)
    
    c =  np.random.randint(len(A[0]))
    d =  np.random.randint(len(A[0]))
        
    if c > d:
        c, d = d, c
    for elem in range(c, d):
        new_A[0].update({f_list[elem]: B[0][f_list[elem]]})   
        new_B[0].update({f_list[elem]: A[0][f_list[elem]]})    
    
    if interaction_option:
        c = np.random.randint(len(A[1]))
        d = np.random.randint(len(A[1]))
        if c > d:
            c, d = d, c
        for elem in range(c, d):
            new_A[1].update({i_list[elem]: B[1][i_list[elem]]})   
            new_B[1].update({i_list[elem]: A[1][i_list[elem]]})   

        return (new_A[0], new_A[1]), (new_B[0], new_B[1])
    return [new_A[0]], [new_B[0]]

def universal_CO(A, B, interaction_option):
    from copy import deepcopy

    if interaction_option:
        f_list = list(A[0].keys())
        i_list = list(A[1].keys())
    else:
        f_list = list(A[0].keys())
    new_A = deepcopy(A)
    new_B = deepcopy(B)
    
    for elem in range(0, len(f_list)):
        t = np.random.random_sample()
        if t > 0.5:
            new_A[0].update({f_list[elem]: B[0][f_list[elem]]})   
            new_B[0].update({f_list[elem]: A[0][f_list[elem]]})    
    
    if interaction_option:
        for elem in range(0, len(i_list)):
            t = np.random.random_sample()
            if t > 0.5:
                new_A[1].update({i_list[elem]: B[1][i_list[elem]]})   
                new_B[1].update({i_list[elem]: A[1][i_list[elem]]})   

        return (new_A[0], new_A[1]), (new_B[0], new_B[1])
    return [new_A[0]], [new_B[0]]
    
    
    
    
    
#----------
# MAIN FUNCTION
#---------- 
def nsga2_KT(feature_and_optional_interaction, v):
    """
    A function, which takes the given and estimated feature (and optional interaction) values, 
    as well as the lists of variants of the attributed and non-attributed model and optimizes 
    
    Args:
        feature_and_optional_interaction (list of lists): The first and second sublist contain the feature values of the attributed and non-attributed model respectively. 
        The third and fourth sublis contain the interaction values of the attributed and non-attributed model respectively, if provided.
        v (numpy matrix): The previously computed variants of the attributed model
        e_v (numpy matrix): The previously computed variants of the non-attributed model
    Returns:
        BestFront: A list with feature (and optional interaction) values e.i candidate solutions, which have an optimal similarity to the value distribution of the atributed model.
    """
    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool
    from functools import partial
    import itertools
    
    try:
        pop_size = int(config['NSGAII']['Population_Size'])
    except:
        sys.exit("Population_Size must be an integer. Please check your configuration file!")

    archive_size = pop_size
    max_gen = np.inf
    gen_no = 0
    interaction_option = False
    converge_counter = 0
    Old_Best = 0.0
    Old_Mean = 0.0
    
    if str(config['NSGAII']['Maximum_Generations']) != "auto":
        try:
            max_gen = int(config['NSGAII']['Maximum_Generations'])
        except:
            sys.exit("Maximum_Generations must be an integer. Please check your configuration file!")
      
    if str(config['Model']['With_Interactions']) == "True":
        interaction_option = True 
    
    if str(config['Miscellaneous']['NumberOfThreads']) != "auto":
        number_of_threads = int(config['Miscellaneous']['NumberOfThreads'])
    else:
        number_of_threads = os.cpu_count()
     
    #get list of features for attributed and non-attributed model
    feature_list = feature_and_optional_interaction[0]
    #feature_list_pure = list(feature_list.values())
    
    #if provided: get interactions for attributed and non-attributed model
    if interaction_option:
        interactions_list = feature_and_optional_interaction[1]
        f_and_i = np.asmatrix((feature_list, interactions_list))
        
    else:
        f_and_i = np.matrix(feature_list)
    
    #build initial population
    for i in range(pop_size):
        if interaction_option:
            feature_list_new, interactions_list_new =  modify_model([feature_list,interactions_list])
            interactions_list_pure = list(interactions_list_new.values()) 
            new_f_and_i = np.matrix((feature_list_new, interactions_list_new))
            #new_f_and_i = (feature_list_new, interactions_list_new)
        else:
            feature_list_new = modify_model([feature_list])
            new_f_and_i = np.matrix(feature_list_new)
            
        if 'P' in locals():
            #P = np.vstack([P, np.asmatrix(({"Test": "test"},{"Test": "test"}))])
            P = np.vstack([P, new_f_and_i])
        else:
            P = new_f_and_i
            
    #initialize archive
    A = list()
   
    while(gen_no<max_gen and converge_counter <3):
        print("Generation:", gen_no+1)
        
        if len(A)!=0:
            P = np.vstack([P, m_A])
                
        #compute objectives (similarities)
        
        def p_similarity_worker(s_f):
            Pop_Fitness = dict()
            Fitness_to_candidate = dict()
            for i in range(s_f[0], s_f[1]):
                All_Fitnesses = compute_fulfilled_objectives(P[i], f_and_i)
                candidate = list(P[i,0].values())
                if interaction_option:
                    candidate.extend(list(P[i,1].values()))
                    Fitness_to_candidate[tuple(candidate)] = (P[i,0], P[i,1])
                else:
                    Fitness_to_candidate[tuple(candidate)] = ([P[i,0]])
                Pop_Fitness[tuple(candidate)] = All_Fitnesses
            return [Pop_Fitness, Fitness_to_candidate]
        
        pool = ThreadPool()
        intervals = [int(round(x*(P.shape[0]/number_of_threads))) for x in range(number_of_threads+1)]
        s_f = list()
        for s, f in zip(intervals[:-1], intervals[1:]):
            s_f.append([s,f])

        pool_fitness = pool.map(p_similarity_worker, s_f)
        pool.close() 
        pool.join()
        
        Pop_Fitness = pool_fitness[0][0]
        candidate_list2dict = pool_fitness[0][1]
        
        R = front_rank_assignment(Pop_Fitness)
        BestFront = dict((tuple(elem), Pop_Fitness[tuple(elem)]) for elem in R[0])
        
        Best, Mean = ComparableFrontFitness(BestFront)
        if Best - Old_Best < 0.0 or Mean - Old_Mean == 0:
            converge_counter = converge_counter +1
        else:
            converge_counter = 0
        Old_Best = Best
        Old_Mean = Mean
        
        del A[:]
      
        try:
            del m_A
        except:
            pass
        
        for i in range(0,len(R)):
            S = Sparsity(R[i], Pop_Fitness)
            if len(A) + len(R[i]) >= archive_size:
                A.extend(S[:(archive_size - len(A))])
                break
            else:
                A.extend(R[i])
                
        
        for i in range(0,len(A)):
            candidate= candidate_list2dict[tuple(A[i])]
            if 'm_A' in locals():  
                m_A = np.vstack([m_A, np.matrix(candidate)])
            else:
                m_A = np.matrix(candidate)
            
                
        A_Fitness = dict()
        
        for elem in A:
            A_Fitness[tuple(elem[0:])] = Pop_Fitness[tuple(elem[0:])]
        
        P = breed_KT(A_Fitness, candidate_list2dict, pop_size)
        
        for i in range(0, pop_size, 2):
            if interaction_option:
                feature_list_new, interactions_list_new =  modify_model([feature_list,interactions_list])     
                new_f_and_i = np.matrix((feature_list_new, interactions_list_new))
            else:
                feature_list_new = modify_model([feature_list_new])         
                new_f_and_i = np.matrix(feature_list_new)
            P = np.vstack([P, new_f_and_i])   
        
        gen_no = gen_no +1
    return BestFront
	
#========================
# ADDITIONAL FUNCTIONS FOR TRANSFER LEARNING
#========================

def most_influential(dict_values, threshold):
    """
    A function which takes a list of items and returns the values that are larger than 75% of the dataset
    
    Args:
        list_values (list): A list with float values
        
    Returns:
        A list with values that are larger than 75% of the dataset
    """    
    list_values = list(dict_values.values())
    list_values.sort(reverse=True)
    culled_list = list_values[0:round(len(list_values)*threshold)]
    influential = {key:dict_values[key] for key in dict_values if dict_values[key] in culled_list}
    return influential

#---------
#NOISE
#---------
def noise(value_change, p, mu, sigma):
    """
    A function, which takes a candidate solution and adds noise to some of their elements.

    Args:
        value_changes (dict): A dictionary with feature or interaction names and values.
        p (float): probability of changing a value
        mu (float): the mean for the normal distribution
        sigma (float): the standard deviation for the normal distribution

    Returns:
        A dict with with feature or interaction names and values with added noise.
    """
    for i, j in value_change.items():
        if p >= np.random.random_sample():
            while True:
                s = np.random.normal(mu, sigma, 1)[0]
                break
            new_j = j + (j*s)
            value_change[i] = new_j
    return value_change
    
def noise_small(value_change, p):
    try:
        mu = float(config['Noise_small']['Mean'])
        sigma = float(config['Noise_small']['Standard_deviation'])
    except:
        sys.exit("Mean and Standard_deviation must be float. Please check your configuration file!")
        
    value_change = noise(value_change, p, mu, sigma)
    return value_change

def noise_big(value_change, p):
    try:
        mu = float(config['Noise_big']['Mean'])
        sigma = float(config['Noise_big']['Standard_deviation'])
    except:
        sys.exit("Mean and Standard_deviation must be float. Please check your configuration file!")
    
    value_change = noise(value_change, p, mu, sigma)
    return value_change

#---------
#LINEAR TRANSFORMATION
#---------
def linear_transformation(value_change, p):
    operation = str(config['Linear_Transformation']['Operation'])
    assert (operation  in ["addition" , "subtraction" , "multiplication" , "division"] ), ("Options for Operations with Linear_Transformation are: addition, subtraction, multiplication, division") 
    try:
        operand = float(config['Linear_Transformation']['Operand'])
    except:
        sys.exit("Operand for Linear_Transformation must be float. Please check your configuration file!")
    
    
    transform = {
            'addition': lambda x: x + x *operand,
            'substraction': lambda x: x - x *operand,
            'multiplication': lambda x: x *( x *operand),
            'division' :lambda x: x /( x *operand)
        }.get(operation)
    
    for i, j in value_change.items():
        if p >= np.random.random_sample(): 
            j_new = transform(j)
            value_change[i] = j_new
    return value_change

#---------
#NEGATION
#---------
def negation(value_change, p):
    for i, j in value_change.items():
        if p >= np.random.random_sample():
            new_j = j *-1
            value_change[i] = new_j
    return value_change
    
def bad_region(c, f_amount):
    """
    A function which tries to find bad regions in a SAT problems search space.
    
    Args: 
        c (list of lists): The list of constrains
        f_amount (int): Amount of Features that aee involved in the SAT
        
    Returns:
        List of Features and their setting, which result in unsatisfied assignments for the SAT
    """
    import pycosat

    bad_regions = list()
    for i in range(1,f_amount+1):
        c_copy = list(c)
        c_copy.append([i])
        if pycosat.solve(c_copy) == "UNSAT":
            bad_regions.append(-i)

        c_copy = list(c)
        c_copy.append([-i])
        if pycosat.solve(c_copy) == "UNSAT":
            bad_regions.append(i)
    
    return bad_regions

def modify_model(data):
    Change_Operations = config['Scope_for_Changes']['Change_Operations'].split(", ")
    Change_Probs = list()
    for elem in Change_Operations:
        assert (str(elem)  in ["Noise_small" , "Noise_big" , "Linear_Transformation" , "Negation"] ), ("Options for Change_Operations are: Noise_small, Noise_big, Linear_Transformation, Negation")
        try:
            Change_Probs.append(float(config[str(elem)]['Probability']))
        except:
            sys.exit("Probability for Noise_small, Noise_big, Linear_Transformation and Negation must be float. Please check your configuration file!")
    try:
        Relevance_Treshhold = float(config['Scope_for_Changes']['Relevance_Treshhold'])
    except:
        sys.exit("Relevance_Treshhold must be a float. Please check your configuration file!")
    
    feature_list_new = dict(data[0])
    if config['Model'].getboolean('With_Interactions') == True:
        interactions_list_new = dict(data[-1])
        
    
    #========================
    #CHOSE DATA
    #========================     
    
    #Features
    assert (str(config['Scope_for_Changes']['Change_Feature'])  == "all" or "most-influential" or "none"), ("Options for Change_Feature are: all, most-influential, none")
    
    if config['Scope_for_Changes']['Change_Feature'] == "most_influential":
        feature_changes = most_influential(feature_list_new, Relevance_Treshhold)
    else:
        feature_changes = feature_list_new
        
    #Interactions
    if config['Model'].getboolean('With_Interactions') == True:
        assert (str(config['Scope_for_Changes']['Change_Interaction']) in ["all" , "most-influential" , "none"] ), ("Options for Change_Interaction are: all, most-influential, none")
        if config['Scope_for_Changes']['Change_Interaction'] == "most_influential":
            interactions_changes = most_influential(interactions_list_new, Relevance_Treshhold)
        else:
            interactions_changes = interactions_list_new
            
    #========================
    #MODIFY DATA
    #========================

    #if user wants to perform more than one data modification, the program will perform them in succession:
    for elem in range(len(Change_Operations)): 
        if config['Scope_for_Changes']['Change_Feature'] != "none":
            feature_changes = {
            'Noise_small':  noise_small,
            'Noise_big': noise_big,
            'Linear_Transformation': linear_transformation,
            'Negation': negation
            }.get(Change_Operations[elem])(feature_changes, Change_Probs[elem])
            
            feature_list_new.update(feature_changes)
        
        if config['Model'].getboolean('With_Interactions') == True and config['Scope_for_Changes']['Change_Interaction'] != "none":         
            interactions_changes = {
            'Noise_small':  noise_small,
            'Noise_big': noise_big,
            'Linear_Transformation': linear_transformation,
            'Negation': negation
            }.get(Change_Operations[elem])(interactions_changes, Change_Probs[elem])
            
            interactions_list_new.update(interactions_changes)
    
    if config['Model'].getboolean('With_Interactions') == True:
        return feature_list_new, interactions_list_new
    else:
        return feature_list_new

def plotting_KT(old_data, new_data, filepath):
    """
    A function which takes the original and modified features, interactions and fitness values and compares them with them help of plot diagrams
    
    Args:
        old_data (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the original data
        new_data (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the modified data
              
    """
    import matplotlib.pyplot as plt
    
    #instantiating stuff
    try:
        amount_bins = int(config['Miscellaneous']['NumberOfBins'])
    except:
        sys.exit("NumberOfBins must be an integer. Please check your configuration file!")
    
    #PREPARE THE DATA   
    #feature values    
    old_F = old_data[0]
    new_F = new_data[0]
    kde_old_F = kde(old_F, len(old_data[0]))
    kde_new_F = kde(new_F, len(new_data[0]))
    bin_old_F = np.linspace(min(old_F), max(old_F), amount_bins)
    bin_new_F = bin_old_F
    
    #variant fitness values    
    old_V= old_data[-1]
    new_V = new_data[-1]
    kde_old_V = kde(old_V, old_data[-1].size)
    kde_new_V = kde(new_V, new_data[-1].size) 
    bin_old_V = np.linspace(old_V[old_V != 0].min(), old_V[old_V != 0].max(), amount_bins)
    bin_new_V = bin_old_V
	
    if str(config['Model']['With_Interactions']) == "True":
        #real interaction values    
        old_I =list(old_data[1])
        new_I =list(new_data[1])
        kde_old_I = kde(old_I,len(old_data[1]))
        kde_new_I = kde(new_I, len(new_data[1])) 
        bin_old_I = np.linspace(min(old_I), max(old_I), amount_bins)
        bin_new_I = np.linspace(min(new_I), max(new_I), amount_bins)
        
    #INITIALIZE PLOT    
    if str(config['Model']['With_Interactions']) == "True":
        fig = plt.figure(figsize=(30,30))
        oF = fig.add_subplot(331)
        nF = fig.add_subplot(332)
        F = fig.add_subplot(333)

        oI = fig.add_subplot(334)
        nI = fig.add_subplot(335)
        I = fig.add_subplot(336)

        oV = fig.add_subplot(337)
        nV = fig.add_subplot(338)
        V = fig.add_subplot(339)
        
    if str(config['Model']['With_Interactions']) == "False":
        fig = plt.figure(figsize=(30,20))
        oF = fig.add_subplot(231)
        nF = fig.add_subplot(232)
        F = fig.add_subplot(233)

        oV = fig.add_subplot(234)
        nV = fig.add_subplot(235)
        V = fig.add_subplot(236) 
    
    #PLOT THE DATA
    oF.set_title("old feature values")
    oF.hist(old_F, bins=bin_old_F,fc="#a58a66", density=True,alpha=0.5)
    oF.plot(kde_old_F[0][:, 0], kde_old_F[1], linewidth=2, color="#ba824c",alpha=1)
    oF.set_xlabel('value')
    oF.set_ylabel('density')
    
    nF.set_title("new feature values")
    nF.hist(new_F, bins=bin_new_F,fc="#669ba5", density=True, alpha=0.5)
    nF.plot(kde_new_F[0][:, 0], kde_new_F[1], linewidth=2, color="#43676d",alpha=1)
    nF.set_xlabel('value')
    nF.set_ylabel('density')
    
    F.set_title("old and new feature values")
    F.hist(new_F, bins=bin_new_F,fc="#a58a66", density=True, alpha=0.5)
    F.hist(old_F, bins=bin_old_F,fc="#669ba5", density=True, alpha=0.5)
    F.plot(kde_new_F[0][:, 0], kde_new_F[1], linewidth=2, color="#43676d",alpha=1)
    F.plot(kde_old_F[0][:, 0], kde_old_F[1], linewidth=2, color="#ba824c",alpha=1)
    F.set_xlabel('value')
    F.set_ylabel('density')
    
    #######
    if str(config['Model']['With_Interactions']) == "True":
        oI.set_title("old interaction values")
        oI.hist(old_I, bins=bin_old_I,fc="#a58a66", density=True, alpha=0.5)
        oI.plot(kde_old_I[0][:, 0], kde_old_I[1], linewidth=2, color="#ba824c",alpha=1)
        oI.set_xlabel('value')
        oI.set_ylabel('density')

        nI.set_title("new interaction values")
        nI.hist(new_I, bins=bin_new_I,fc="#669ba5", density=True, alpha=0.5)
        nI.plot(kde_new_I[0][:, 0], kde_new_I[1], linewidth=2, color="#43676d",alpha=1)
        nI.set_xlabel('value')
        nI.set_ylabel('density')

        I.set_title("old and new interaction values")
        I.hist(old_I, bins=bin_old_I,fc="#a58a66", density=True, alpha=0.5)
        I.hist(new_I, bins=bin_new_I,fc="#669ba5", density=True, alpha=0.5)
        I.plot(kde_old_I[0][:, 0], kde_old_I[1], linewidth=2, color="#ba824c",alpha=1)        
        I.plot(kde_new_I[0][:, 0], kde_new_I[1], linewidth=2, color="#43676d",alpha=1)
        I.set_xlabel('value')
        I.set_ylabel('density')
    
    ######
    
    oV.set_title("old variant values")
    oV.hist(old_V, bins=bin_old_V,fc="#a58a66", density=True, alpha=0.5)
    oV.plot(kde_old_V[0][:, 0], kde_old_V[1], linewidth=2, color="#ba824c",alpha=1)
    oV.set_xlabel('value')
    oV.set_ylabel('density')
    
    nV.set_title("new variant values")
    nV.hist(new_V, bins=bin_new_V,fc="#669ba5", density=True, alpha=0.5)
    nV.plot(kde_new_V[0][:, 0], kde_new_V[1], linewidth=2, color="#43676d",alpha=1)
    nV.set_xlabel('value')
    nV.set_ylabel('density')
    
    V.set_title("old and new variant values")
    V.hist(old_V, bins=bin_old_V,fc="#a58a66", density=True, alpha=0.5)
    V.hist(new_V, bins=bin_new_V,fc="#669ba5", density=True, alpha=0.5)
    V.plot(kde_old_V[0][:, 0], kde_old_V[1], linewidth=2, color="#ba824c",alpha=2)
    V.plot(kde_new_V[0][:, 0], kde_new_V[1], linewidth=2, color="#43676d",alpha=2)
    V.set_xlabel('value')
    V.set_ylabel('density')
    
    #save the plot
    plt.savefig(filepath + '/plots.png', bbox_inches='tight')
    plt.savefig(filepath +'/plots.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
#========================
#
# MAIN FUNCTION FOR AVM-GENERATION
#
#========================

def avm_generation():

    #========================
    #GET ATTRIBUTES FROM CONFIG-FILE
    #========================
    Attrb_DIMACS = config['AttributedModel']['DIMACS-file']
    Attrb_Feature = config['AttributedModel']['Feature-file']
    Non_Attrb_DIMACS = config['NonAttributedModel']['DIMACS-file']
    Non_Attrb_Feature = config['NonAttributedModel']['Feature-file']

    if config['AttributedModel'].getboolean('With_Interactions') == True:
        Attrb_Interactions = config['AttributedModel']['Interactions-file']
        Interactions_Specs = config['NonAttributedModel']['New_Interactions_Specs'].split(", ")
        try:
            Interactions_Specs = list(map(int, Interactions_Specs))
        except:
            sys.exit("Interaction_Specs must be a sequence of integers. Please check your configuration file!")
    
    try:
        valid_variants_size = int(config['Variants']['NumberOfVariants'])
    except:
        sys.exit("NumberOfVariants must be an integer. Please check your configuration file!")

    sampling_method = str(config['Variants']['Sampling_Method'])

    #========================
    #READ LISTS FROM FILES
    #========================
    constraint_list = parsing_dimacs(Attrb_DIMACS)# constrains which influence valid variants
    constraint_list_for_estimation = parsing_dimacs(Non_Attrb_DIMACS)
    feature_list = parsing_text(Attrb_Feature)#feature names and values 
    feature_list_for_estimation = parsing_text(Non_Attrb_Feature)
    
    if config['AttributedModel'].getboolean('With_Interactions') == True:
        interactions_list = parsing_text(Attrb_Interactions)#interactions of features and their values
    print("Finished with parsing the files")

    #========================
    #USE CONSTRAINTS TO GENERATE VALID VARIANTS
    #========================
    if sampling_method != "random":
        valid_variants = get_valid_variants(constraint_list, len(feature_list.keys())-1)
        valid_variants_for_estimation = get_valid_variants(constraint_list_for_estimation, len(feature_list_for_estimation.keys())-1)
    else:
        valid_variants = get_valid_variants(constraint_list, valid_variants_size)
        valid_variants_for_estimation = get_valid_variants(constraint_list_for_estimation, valid_variants_size)
    
        
    print("Finished with creating variants")

    #========================
    #PREPARE FEATURE AND INTERACTION LISTS
    #========================
    feature_list_pure = list(feature_list.values())

    if config['AttributedModel'].getboolean('With_Interactions') == True:
        interactions_list_pure = list(interactions_list.values())
        interaction_list_for_estimation = new_interactions(constraint_list_for_estimation, feature_list_for_estimation, Interactions_Specs)
        valid_complete_variants = append_interactions(valid_variants, feature_list, interactions_list)
        e_valid_complete_variants = append_interactions(valid_variants_for_estimation, feature_list_for_estimation, interaction_list_for_estimation)
        f_and_i = concatenate(feature_list_pure, interactions_list_pure)
        avm = [feature_list, interactions_list]
        nsga_data = [feature_list, feature_list_for_estimation, interactions_list, interaction_list_for_estimation]
    else:
        valid_complete_variants = valid_variants
        e_valid_complete_variants = valid_variants_for_estimation
        f_and_i = np.asmatrix(feature_list_pure)
        avm = [feature_list]
        nsga_data = [feature_list, feature_list_for_estimation]

    #========================
    #CALCULATE THE FITNESS OF THE AVM FOR EVERY VALID VARIANT
    #========================
    fitness_scores = performance(valid_complete_variants, f_and_i, f_and_i.shape[1])
    avm.append(fitness_scores)

    #========================
    #START OPTIMIZING
    #========================
    if config['AttributedModel'].getboolean('With_Variants') == True:
    #NSGA-II
        print("Starting NSGA-II")
        BestFront_dict = nsga2(nsga_data, valid_complete_variants, e_valid_complete_variants)
    else:
    #Just KDE
        e_feature_list = estimation(feature_list, feature_list_for_estimation)
        e_feature_list_pure = list(e_feature_list.values())
        BestFront =  [e_feature_list_pure]

        #if provided: get initial values for estimated interactions
        if config['AttributedModel'].getboolean('With_Interactions') == True:
            e_interaction_list = estimation(interactions_list, interaction_list_for_estimation)
            e_interactions_list_pure = list(e_interaction_list.values())
            BestFront = concatenate(e_feature_list_pure, e_interactions_list_pure).tolist()

    #========================
    #SAVE THE RESULTS
    #========================
    print("Finished with calculating results")
    print("Start saving results")

    #define name and path of the directory
    if str(config['Miscellaneous']['DirectoryToSaveResults']) != "auto":
        directory = str(config['Miscellaneous']['DirectoryToSaveResults'])
    else:
        directory = datetime.datetime.now().strftime("AVM-Generation_Results/Gen_results-%Y-%m-%d_%H%M%S")

    if not os.path.exists(directory):
        os.makedirs(directory)

    #define results to be saved
    BestFront = define_results(BestFront_dict)
    
    #save results
    for i in range(0,len(BestFront)):
        if not os.path.exists(directory +"/result"+str(i+1)):
            os.makedirs(directory +"/result"+str(i+1))
        #write results into a txt-file:
        filedirectory = directory +"/result"+str(i+1)

        e_feature_list = BestFront[i][0:len(list(feature_list_for_estimation.values()))]
        vm = list([e_feature_list])
        if str(config['AttributedModel']['With_Interactions']) == "True":
            e_interactions_list = BestFront[i][len(list(feature_list_for_estimation.values())):]
            vm.append(e_interactions_list)
        e_fitness_scores = performance(e_valid_complete_variants, np.asmatrix(BestFront[i]), f_and_i.shape[1])
        vm.append(e_fitness_scores)
        
        writing_text(filedirectory, feature_list_for_estimation, e_feature_list, "new_features")
        if config['AttributedModel'].getboolean('With_Interactions') == True:
            writing_text(filedirectory, interaction_list_for_estimation, e_interactions_list, "new_interactions")

        #PLOTTING, SO WE CAN LOOK AT SOMETHING
        plotting(avm, vm, filedirectory+"/")

    print("Finished with saving results")

#========================
#
# MAIN FUNCTION FOR AVM MODIFICATION
#
#========================

def avm_modificator():
    #========================
    #GET ATTRIBUTES FROM CONFIG-FILE
    #========================
    Attrb_DIMACS = config['Model']['DIMACS-file']
    Attrb_Feature = config['Model']['Feature-file']
    
    if config['Model'].getboolean('With_Interactions') == True:
        Attrb_Interactions = config['Model']['Interactions-file']
        
    try:
        valid_variants_size = int(config['Variants']['NumberOfVariants'])
    except:
        sys.exit("NumberOfVariants must be an integer. Please check your configuration file!")
        
    sampling_method = str(config['Variants']['Sampling_Method'])


    #========================
    #READ LISTS FROM FILES
    #========================
    constraint_list = parsing_dimacs(Attrb_DIMACS)# constrains which influence valid variants
    feature_list = parsing_text(Attrb_Feature)#feature names and values
    if config['Model'].getboolean('With_Interactions') == True:
        interactions_list = parsing_text(Attrb_Interactions)#interactions of features and their values
    
    print("Finished with parsing the files")

    #========================
    #USE CONSTRAINTS TO GENERATE VALID VARIANTS
    #========================
    if config['Model'].getboolean('With_Variants') == True:
        if sampling_method != "random":
            valid_variants = get_valid_variants(constraint_list, len(feature_list.keys())-1)
        else:
            valid_variants = get_valid_variants(constraint_list, valid_variants_size)
    else:
        valid_variants = np.asmatrix([1] * (len(feature_list)-1))
                                        
    print("Finished with creating variants")
    
    #========================
    #PREPARE FEATURE AND INTERACTION LISTS
    #========================
    feature_list_pure = list(feature_list.values()) 

    if config['Model'].getboolean('With_Interactions') == True:
        interactions_list_pure = list(interactions_list.values())
        valid_complete_variants = append_interactions(valid_variants, feature_list, interactions_list)
        f_and_i = concatenate(feature_list_pure, interactions_list_pure)
        nsga_data = [feature_list, interactions_list]
    else:
        valid_complete_variants = valid_variants
        f_and_i = np.asmatrix(feature_list_pure)
        nsga_data = [feature_list]
        
    #========================
    #START OPTIMIZING
    #========================  
    print("Starting NSGA-II")
    BestFront_dict = nsga2_KT(nsga_data, valid_complete_variants)
    
    #========================
    #COMMON AND DEAD FEATURES
    #========================
    if config['Search_Space'].getboolean('Find_common_and_dead_features') == True:
        bad_regions = bad_region(constraint_list,len(feature_list_pure))    
    
    #========================
    #CREATE COHERENT DATASET
    #========================
    fitness_scores = performance(valid_complete_variants, f_and_i, f_and_i.shape[1])
    if config['Model'].getboolean('With_Interactions') == True:
        old_data = [feature_list_pure, interactions_list_pure, fitness_scores]
    else:
        old_data = [feature_list_pure, fitness_scores]    
    
    #========================
    #SAVE RESULTS
    #======================== 
    print("Finished with calculating results")
    print("Start saving results")

    #define name and path of the directory
    if str(config['Miscellaneous']['DirectoryToSaveResults']) != "auto":
        directory = str(config['Miscellaneous']['DirectoryToSaveResults'])
    else:
        directory = datetime.datetime.now().strftime("AVM-Modification_results/Mod_results-%Y-%m-%d_%H%M%S")

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #define results to be saved
    BestFront = define_results(BestFront_dict)

    for i in range(0,len(BestFront)):
        if not os.path.exists(directory +"/result"+str(i+1)):
            os.makedirs(directory +"/result"+str(i+1))
        filedirectory = directory +"/result"+str(i+1)   

        #save results
        feature_list_pure_new = BestFront[i][:len(feature_list_pure)]
        writing_text(filedirectory+"/", feature_list, feature_list_pure_new, "new_features")
        if config['Model'].getboolean('With_Interactions') == True:
            interactions_list_pure_new = BestFront[i][len(feature_list_pure):]
            f_and_i_new = concatenate(feature_list_pure_new, interactions_list_pure_new)
            writing_text(filedirectory+"/", interactions_list, interactions_list_pure_new, "new_interactions")
        else:
            f_and_i_new = np.asmatrix(feature_list_pure_new)
            
        fitness_scores_new = performance(valid_complete_variants, f_and_i_new, f_and_i.shape[1])
        
        if config['Model'].getboolean('With_Interactions') == True:
            new_data = [feature_list_pure_new, interactions_list_pure_new, fitness_scores_new]
        else:
            new_data = [feature_list_pure_new, fitness_scores_new]

        if config['Search_Space'].getboolean('Find_bad_regions') == True:
            f = open(directory +"/bad_regions.txt","w+")
            f.write(', '.join(map(str, np.array(bad_regions))))
            f.close()

        #PLOTTING, SO WE CAN LOOK AT SOMETHING
        plotting_KT(old_data, new_data, filedirectory+"/")

    print("Finished with saving results")

import math
import random
import seaborn as sns; sns.set()
import numpy as np
from array import *
import time
import os
import datetime
import sys

random.seed()

#========================
#GET LOCATION OF CONFIG-FILE
#========================
import argparse

parser = argparse.ArgumentParser(description='Thor.')
parser.add_argument('path', metavar='config file path', type=str, nargs=1, help="the config.ini's file path")
    
args = parser.parse_args()
config_location = args.path[0]

import configparser
global config
config = configparser.ConfigParser()
config.read(config_location)
config.sections()

print("Performing ", config['UseCase']['UseCase'])

#========================
#RUN AVM-GENERATOR
#========================
if config['UseCase']['UseCase'] == "AVM-Generation":
    avm_generation()
    print("The program terminated as expected :)")

#========================
#RUN AVM-MODIFICATOR
#========================
elif config['UseCase']['UseCase'] == "AVM-Modification":
    avm_modificator()
    print("The program terminated as expected :)")
        
else:
    print("Usage of wrong use-case")
    print("The two possible use-cases are:")
    print("AVM-Generation")
    print("AVM-Modification")
    print("")
    print("Please check your configuration files for the right notation!")
	

   
   