import itertools
import math
import operator
import os
import random
import sys
from copy import deepcopy
import numpy as np
from scipy import stats as sps, spatial as spsp
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


class Nsga2:
    PERFORMANCE_RELEVANCE_REL_THRESH = 0.8

    def __init__(self, conf, n_jobs, seed=None):
        self.config = conf
        self.n_jobs = n_jobs
        self.seed = seed
        np.random.seed(self.seed)
        print("Seeding NSGA2 with seed", self.seed)

    def nsga2(self, avm_new, vm_new):
        """
        A function, which takes the given and estimated feature (and optional interaction) values,
        as well as the lists of variants of the attributed and non-attributed model and optimizes

        Args:
            #TODO
        Returns:
            BestFront: A list with feature (and optional interaction) values e.i candidate solutions, which have an optimal similarity to the value distribution of the atributed model.
        """

        try:
            pop_size = int(self.config['NSGAII']['Population_Size'])
        except:
            sys.exit("Population_Size must be an integer. Please check your configuration file!")

        archive_size = pop_size
        max_gen = np.inf
        gen_no = 0
        interaction_option = False
        converge_counter = 0
        old_best = 0.0
        old_mean = 0.0
        avm_performances = avm_new.calc_performance_for_validation_variants()

        if str(self.config['NSGAII']['Maximum_Generations']) != "auto":
            try:
                max_gen = int(self.config['NSGAII']['Maximum_Generations'])
            except:
                sys.exit("Maximum_Generations must be an integer. Please check your configuration file!")

        if vm_new.interactions_specs:
            interaction_option = True

        if self.n_jobs != "auto":
            number_of_threads = int(self.n_jobs)
        else:
            number_of_threads = os.cpu_count()

        # get list of features for attributed and non-attributed model
        feature_influences_avm = avm_new.get_feature_influences()
        feature_influences_vm = vm_new.get_feature_influences()
        feature_influence_vals_avm = list(feature_influences_avm.values())
        influence_range = max(feature_influence_vals_avm) - min(feature_influence_vals_avm)

        # if provided: get interactions for attributed and non-attributed model
        if interaction_option:
            interactions_influences_avm = avm_new.get_interaction_influences()
            interactions_influences_vm = vm_new.get_interaction_influences()
            interactions_values_avm = list(interactions_influences_avm.values())
            interaction_range_avm = max(interactions_values_avm) - min(interactions_values_avm)
            mean_ranges_features_influences = (influence_range + interaction_range_avm) / 2
            # feature_interaction_value_vector = concatenate(feature_influence_vals, interactions_list_pure)
        else:
            mean_ranges_features_influences = influence_range

        # build initial population
        population = []
        for i in range(pop_size):
            # get initial values for estimated features
            feature_influences_vm_estimated = self.estimate_influences_with_kde(feature_influences_avm,
                                                                                feature_influences_vm)
            vm_estimated = deepcopy(vm_new)
            vm_estimated.set_feature_influences(feature_influences_vm_estimated)
            # if provided: get initial values for estimated interactions
            if interaction_option:
                interaction_influences_vm_estimated = self.estimate_influences_with_kde(interactions_influences_avm,
                                                                                        interactions_influences_vm)
                vm_estimated.set_interaction_influences(interaction_influences_vm_estimated)

            population.append(vm_estimated)

        # TODO: not sure what this does
        upsize_kde_data_plot, upsize_kde_densities = kde(feature_influence_vals_avm, vm_new.get_feature_num())
        upsize_features = list()
        for _ in range(vm_new.get_feature_num()):
            p = random.uniform(min(upsize_kde_densities),
                               max(upsize_kde_densities))  # Wert zw kleinster und größter Density
            selectors = [x >= p for x in upsize_kde_densities]  # alle Densities, die größer als p sind
            valid_densities = list(
                itertools.compress(upsize_kde_data_plot, selectors))  # alle Werte, für die validen Densities
            index = random.randrange(len(valid_densities))
            upsize_features.append(valid_densities[index][0])
        feature_influence_vals_avm = list(upsize_features)

        if interaction_option:
            upsize_i_kde = kde(interactions_values_avm, vm_new.get_interaction_num())
            upsize_i = list()
            for _ in range(vm_new.get_interaction_num()):
                p = random.uniform(min(upsize_i_kde[1]), max(upsize_i_kde[1]))  # Wert zw kleinster und größter Density
                selectors = [x >= p for x in upsize_i_kde[1]]  # alle Densities, die größer als p sind
                valid_densities = list(
                    itertools.compress(upsize_i_kde[0], selectors))  # alle Werte, für die validen Densities
                index = random.randrange(len(valid_densities))
                upsize_i.append(valid_densities[index][0])
            interactions_values_avm = list(upsize_i)

        # initialize archive
        archive = list()
        while gen_no < max_gen and converge_counter < 3:
            print("Generation:", gen_no + 1)
            if len(archive) != 0:
                population = [*population, *archive]

            def p_similarity_worker(s_f, vm_estimations):
                pop_fitness = dict()
                for i in range(s_f[0], s_f[1]):
                    vm_estimated = vm_estimations[i]
                    vm_estimated_features = list(vm_estimated.get_feature_influences().values())
                    feature_fitness = compute_similarities(feature_influence_vals_avm, vm_estimated_features,
                                                           self.config)
                    performances_vm_estimated = vm_estimated.calc_performance_for_validation_variants()
                    variant_fitness = compute_similarities(avm_performances, performances_vm_estimated, self.config)
                    if interaction_option:
                        vm_estimated_interactions = list(vm_estimated.get_interaction_influences().values())
                        interaction_fitness = compute_similarities(interactions_values_avm, vm_estimated_interactions,
                                                                   self.config)
                        fitness_tuple = [feature_fitness, interaction_fitness, variant_fitness]
                    else:
                        fitness_tuple = [feature_fitness, variant_fitness]
                    pop_fitness[vm_estimated] = fitness_tuple
                return pop_fitness

            gloabl_interval = [0, len(population)]
            # TODO run in parallel
            pop_fitness = p_similarity_worker(gloabl_interval, population)
            ranks = front_rank_assignment(pop_fitness)
            best_front = {elem: pop_fitness[elem] for elem in ranks[0]}
            best, mean = compare_front_fitness(best_front)
            if best - old_best < 0.0 or mean - old_mean == 0:
                converge_counter = converge_counter + 1
            else:
                converge_counter = 0
            old_best = best
            old_mean = mean
            del archive[:]
            for i in range(0, len(ranks)):
                if len(archive) + len(ranks[i]) >= archive_size:
                    sparsity_ordered_rank = order_by_sparsity(ranks[i], pop_fitness)
                    archive.extend(sparsity_ordered_rank[:(archive_size - len(archive))])
                    break
                else:
                    archive.extend(ranks[i])
            archive_fitness = dict()
            for elem in archive:
                archive_fitness[elem] = pop_fitness[elem]
            population = breed(archive_fitness, pop_size, mean_ranges_features_influences, self.config)
            gen_no = gen_no + 1
        return best_front

    # ----------
    # MAIN FUNCTION
    # ----------
    def nsga2_KT(self, avm_old):
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
        try:
            pop_size = int(self.config['NSGAII']['Population_Size'])
        except:
            sys.exit("Population_Size must be an integer. Please check your configuration file!")

        archive_size = pop_size
        max_gen = np.inf
        gen_no = 0
        interaction_option = False
        converge_counter = 0
        old_best = 0.0
        old_mean = 0.0

        avm_performances = avm_old.calc_performance_for_validation_variants()
        if str(self.config['NSGAII']['Maximum_Generations']) != "auto":
            try:
                max_gen = int(self.config['NSGAII']['Maximum_Generations'])
            except:
                sys.exit("Maximum_Generations must be an integer. Please check your configuration file!")
        if avm_old.uses_interactions():
            interaction_option = True
        if self.n_jobs != "auto":
            number_of_threads = int(self.n_jobs)
        else:
            number_of_threads = os.cpu_count()

        # get list of features for attributed and non-attributed model
        feature_influences_avm = avm_old.get_feature_influences()
        feature_influence_vals_old_avm = list(feature_influences_avm.values())
        influence_range = max(feature_influence_vals_old_avm) - min(feature_influence_vals_old_avm)

        # if provided: get interactions for attributed and non-attributed model
        if interaction_option:
            interactions_influences_avm = avm_old.get_interaction_influences()
            interactions_values_avm = list(interactions_influences_avm.values())
            interaction_range_avm = max(interactions_values_avm) - min(interactions_values_avm)
            mean_ranges_features_influences = (influence_range + interaction_range_avm) / 2
            # feature_interaction_value_vector = concatenate(feature_influence_vals, interactions_list_pure)
        else:
            mean_ranges_features_influences = influence_range
            # feature_interaction_value_vector = np.asmatrix(feature_influence_vals)
        feature_interaction_value_vector = avm_old.get_feature_interaction_value_vector()

        # build initial population
        population = []
        for i in range(pop_size):
            avm_modified = self.modify_model(avm_old)
            population.append(avm_modified)

        # initialize archive
        archive = list()
        while gen_no < max_gen and converge_counter < 3:
            print("Generation:", gen_no + 1)
            if len(archive) != 0:
                population = [*population, *archive]

            def p_similarity_worker(s_f, avm_estimations):
                pop_fitness = {}
                for i in range(s_f[0], s_f[1]):
                    avm_estimated = avm_estimations[i]
                    fitnesses = compute_fulfilled_objectives(avm_old, avm_estimated, self.config, avm_performances)
                    pop_fitness[avm_estimated] = fitnesses
                return pop_fitness

            # TODO make run parallel
            gloabl_interval = [0, len(population)]
            pop_fitness = p_similarity_worker(gloabl_interval, population)
            ranks = front_rank_assignment(pop_fitness)
            best_front = {elem: pop_fitness[elem] for elem in ranks[0]}
            best, mean = compare_front_fitness(best_front)
            if best - old_best < 0.0 or mean - old_mean == 0:
                converge_counter = converge_counter + 1
            else:
                converge_counter = 0
            old_best = best
            old_mean = mean

            del archive[:]
            for i in range(0, len(ranks)):
                if len(archive) + len(ranks[i]) >= archive_size:
                    sparsity_ordered_rank = order_by_sparsity(ranks[i], pop_fitness)
                    archive.extend(sparsity_ordered_rank[:(archive_size - len(archive))])
                    break
                else:
                    archive.extend(ranks[i])

            archive_fitness = {}
            for elem in archive:
                archive_fitness[elem] = pop_fitness[elem]
            population = breed_KT(archive_fitness, pop_size, self.config)

            # modifies original avm with no respect to neither population nor archive
            freshly_modified_models = []
            for n in range(0, pop_size, 2):
                new_model = self.modify_model(avm_old)
                freshly_modified_models.append(new_model)

            population.extend(freshly_modified_models)
            gen_no = gen_no + 1
        return best_front

    def estimate_influences_with_kde(self, avm_feature_influences, vm_feature_influences):
        """
        A function to estimate new values for the non-attributed variablitiy model.

        Args:
            old_data (dict): Features or Interactions, with their names as keys and their values as values.
            new_data (dict): Features or Interactions, with their names as keys, but without corresponding values.

        Returns: The new_data with estimated values for every key.
        """

        avm_values = list(avm_feature_influences.values())

        # performance relevance for features. Top 20% of values
        value_range = max(avm_values) - min(avm_values)
        kde_data_plot, kde_densities = kde(avm_values, len(vm_feature_influences))
        for key in vm_feature_influences.keys():
            p = random.uniform(min(kde_densities),
                               max(kde_densities))  # Wert zw kleinster und größter Density
            selectors = [x >= p for x in kde_densities]  # alle Densities, die größer als p sind
            valid_densities = [a[0] for a in
                               itertools.compress(kde_data_plot, selectors)]  # alle Werte, für die validen Densities
            index = random.randrange(len(valid_densities))
            new_val = valid_densities[index]
            vm_feature_influences[key] = new_val

        return vm_feature_influences

    def noise(self, value_change, p, mu, sigma):
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
        for feature_id, feature_value in value_change.items():
            if p >= np.random.random_sample():
                while True:
                    s = np.random.normal(mu, sigma, 1)[0]
                    break
                new_j = feature_value + (feature_value * s)
                value_change[feature_id] = new_j
        return value_change

    def noise_small(self, value_change, p):
        try:
            mu = float(self.config['Noise_small']['Mean'])
            sigma = float(self.config['Noise_small']['Standard_deviation'])
        except:
            sys.exit("Mean and Standard_deviation must be float. Please check your configuration file!")

        value_change = self.noise(value_change, p, mu, sigma)
        return value_change

    def noise_big(self, value_change, p):
        try:
            mu = float(self.config['Noise_big']['Mean'])
            sigma = float(self.config['Noise_big']['Standard_deviation'])
        except:
            sys.exit("Mean and Standard_deviation must be float. Please check your configuration file!")

        value_change = self.noise(value_change, p, mu, sigma)
        return value_change

    def linear_transformation(self, value_change, p):
        operation = str(self.config['Linear_Transformation']['Operation'])
        assert (operation in ["addition", "subtraction", "multiplication", "division"]), (
            "Options for Operations with Linear_Transformation are: addition, subtraction, multiplication, division")
        try:
            operand = float(self.config['Linear_Transformation']['Operand'])
        except:
            sys.exit("Operand for Linear_Transformation must be float. Please check your configuration file!")

        transform = {
            'addition': lambda x: x + x * operand,
            'substraction': lambda x: x - x * operand,
            'multiplication': lambda x: x * x * operand,
            'division': lambda x: x / x * operand
        }.get(operation)

        for i, j in value_change.items():
            if p >= np.random.random_sample():
                j_new = transform(j)
                value_change[i] = j_new
        return value_change

    def negation(self, value_change, p):
        for i, j in value_change.items():
            if p >= np.random.random_sample():
                new_j = j * -1
                value_change[i] = new_j
        return value_change

    def modify_model(self, avm):
        change_operations = self.config['Scope_for_Changes']['Change_Operations']
        change_probs = {}
        avm_modified = deepcopy(avm)
        for operation_id in change_operations:
            assert (str(operation_id) in ["Noise_small", "Noise_big", "Linear_Transformation", "Negation"]), (
                "Options for Change_Operations are: Noise_small, Noise_big, Linear_Transformation, Negation")
            try:
                change_probs[operation_id] = float(self.config[operation_id]['Probability'])
            except:
                sys.exit("Probability for Noise_small, Noise_big, Linear_Transformation and Negation must be float. "
                         "Please check your configuration file!")

        feature_list_new = avm_modified.get_feature_influences()
        if avm_modified.uses_interactions():
            interactions_list_new = avm_modified.get_interaction_influences()

        # Features
        assert (str(
            self.config['Scope_for_Changes']['Change_Feature']) == "all" or "most-influential" or "none" or None), (
            "Options for Change_Feature are: all, most-influential, none")

        if self.config['Scope_for_Changes']['Change_Feature'] == "most-influential":
            try:
                relevance_treshhold = float(self.config['Scope_for_Changes']['Relevance_Treshhold'])
            except:
                sys.exit("Relevance_Treshhold must be a float. Please check your configuration file!")
            feature_changes = most_influential(feature_list_new, relevance_treshhold)
        else:
            feature_changes = feature_list_new

        # Interactions
        if avm_modified.uses_interactions():
            assert (str(self.config['Scope_for_Changes']['Change_Interaction']) in ["all", "most-influential",
                                                                                    "none"]), (
                "Options for Change_Interaction are: all, most-influential, none")
            if self.config['Scope_for_Changes']['Change_Interaction'] == "most-influential":
                try:
                    relevance_treshhold = float(self.config['Scope_for_Changes']['Relevance_Treshhold'])
                except:
                    sys.exit("Relevance_Treshhold must be a float. Please check your configuration file!")
                interactions_changes = most_influential(interactions_list_new, relevance_treshhold)
            else:
                interactions_changes = interactions_list_new

        # if user wants to perform more than one data modification, the program will perform them in succession:
        for operation_id in change_operations:
            if self.config['Scope_for_Changes']['Change_Feature'] != "none":
                feature_changes = {
                    'Noise_small': self.noise_small,
                    'Noise_big': self.noise_big,
                    'Linear_Transformation': self.linear_transformation,
                    'Negation': self.negation
                }[operation_id](feature_changes, change_probs[operation_id])

                feature_list_new.update(feature_changes)

            if avm_modified.uses_interactions() and self.config['Scope_for_Changes']['Change_Interaction'] != "none":
                interactions_changes = {
                    'Noise_small': self.noise_small,
                    'Noise_big': self.noise_big,
                    'Linear_Transformation': self.linear_transformation,
                    'Negation': self.negation
                }[operation_id](interactions_changes, change_probs[operation_id])

                interactions_list_new.update(interactions_changes)

        for feature_id, feature_val in feature_changes.items():
            avm_modified.set_feature_influence(feature_id, feature_val)

        if avm.uses_interactions():
            for interaction_id, interaction_val in interactions_changes.items():
                avm_modified.set_interaction_influence(interaction_id, interaction_val)

        return avm_modified


def breed(population_objectives, pop_size, features_interactions_influence_range, config):
    """
    A function which takes the population of candidate solutions and uses it to generate a new set of candidate solutions.

    Args:
        population_objectives (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        pop_size (int): The number of candidate solutions in the population.
        features_interactions_influence_range (float): The value range of the features and - if applicable - interactions

    Returns:
        offspring: A numpy matrix with the generated, new candidate solutions. The number of new candidate solutions is defined by the pop_size
    """
    offspring = list()
    choice = str(config['NSGAII']['Selection_Algorithm'])
    choice_CO = str(config['NSGAII']['Recombination_Algorithm'])

    for i in range(0, pop_size, 2):
        Parent_A, Parent_B = {
            'tournament_selection': tournament_selection,
            'fitness_proportionate_selection': fitness_proportionate_selection,
            'stochastic_universal_sampling': stochastic_universal_sampling
        }.get(choice)(population_objectives)
        # TODO move mapping to more central place
        Child_A, Child_B = {
            'Line_Recombination': line_recombination,
            'simulated_binary_CO': sim_binary_co,
        }.get(choice_CO)(Parent_A, Parent_B, features_interactions_influence_range)
        offspring.append(Child_A)
        offspring.append(Child_B)
    while len(offspring) != pop_size:
        offspring = offspring[:-1]
    return offspring


def compute_similarities(data, e_data, config):
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

    np.warnings.filterwarnings('ignore')
    sim_results = list()
    sim_measures = list(config['NSGAII']['Similarity_Measures'])
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
        # KS_result = sps.spearmanr(data, e_data)
        sim_results.append(1 - min(KS_result[0], 0))
    mean_similarity = sum(sim_results) / len(sim_results)
    return mean_similarity


def front_rank_assignment(population_fitness_dicts):  # works :D
    """
    A function to compute the ranks of a population given a list of objectives.

    Args:
        population_fitness_dicts (dict): A dictionary which containts the current population as keys and a list of their objective costs as values

    Returns:
        R: A list of lists with the population ordered by rank, with the following schema: [[Rank1], [Rank2], ... , [RankN]]
     """
    population_copy = population_fitness_dicts.copy()
    ranks = list()
    i = 0
    while len(population_copy) >= 1:
        ranks.append([])
        front_rank = non_dominated_search(population_copy)
        ranks[i].extend(front_rank)
        for elem in ranks[i]:
            del population_copy[elem]
        i = i + 1
    return ranks


def non_dominated_search(population_fitness_dicts):  # def non_dominated_search(P, O):# works :D
    """
    A function to compute the front rank of a population given a list of objectives.

    Args:
        population_fitness_dicts (dict): A dictionary which containts the current population as keys and a list of their objective costs as values

    Returns:
        Front: A list of all candidate solutions, which are in the front rank i.e. which aren't dominated by other candidate solutions.
    """
    keys = population_fitness_dicts.keys()
    front = [x for x in keys]
    true_front = []
    front_bool = [0] * len(front)

    for elem in front:
        front_copy = [x for x in front if x != elem and front_bool[front.index(x)] != 1]
        for individual in front_copy:
            if pareto_dominates(population_fitness_dicts, individual, elem):
                front_bool[front.index(elem)] = 1
            elif pareto_dominates(population_fitness_dicts, elem, individual):
                front_bool[front.index(individual)] = 1
    for i in range(0, len(front)):
        if front_bool[i] == 0:
            true_front.append(front[i])
    return true_front


def pareto_dominates(population_fitness_dicts, candidate_a, candidate_b):  # works :D
    """
    A function which checks if candidate solution A dominates candidate solution B.

    Args:
        population_fitness_dicts (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        candidate_a (list): The candidate solution A
        candidate_b (list): The candidate solution B

    Returns:
        dominate: True, if candidate solution A dominates candidate solution B. False, otherwise.
    """
    dominate = False
    for objective_id in range(len(list(population_fitness_dicts.values())[0])):
        if population_fitness_dicts[candidate_a][objective_id] > population_fitness_dicts[candidate_b][objective_id]:
            dominate = True
        elif population_fitness_dicts[candidate_b][objective_id] > population_fitness_dicts[candidate_a][objective_id]:
            return False
    return dominate


def order_by_sparsity(candidate_solutions, pop_costs):
    """
    A function which calculates the sparsity of a rank of candidate solutions.

    Args:
        candidate_solutions (list): A rank of candidate solutions.
        pop_costs (dict): A dictionary which containts the current population as keys and a list of their objective costs as values

    Returns:
        F: The list of candidate solutions ordered by their sparsity.
    """

    sparsity = list()
    objective_fitnesses = list()

    for candidate in candidate_solutions:
        sparsity.append(0)
        objective_fitnesses.append(pop_costs[candidate])

    objective_ids = list(objective_fitnesses[0])
    for obj_id in range(len(objective_ids)):
        pop_single_obj_fitnesses = [x[obj_id] for x in objective_fitnesses]
        # TODO check ordering
        # pop_costs.sort(reverse=True)
        single_obj_sorted_candidates = [x for _, x in
                                        sorted(zip(pop_single_obj_fitnesses, candidate_solutions), reverse=True,
                                               key=lambda tu: tu[0])]
        objectives_sorted_by_single_obj = sorted(objective_fitnesses, key=operator.itemgetter(obj_id), reverse=True)
        # TODO check if this yields whole range since pop_single_obj_fitnesses seems to not be sorted
        o_range = pop_single_obj_fitnesses[-1] - pop_single_obj_fitnesses[0]
        if o_range == 0:
            o_range = 1

        sparsity[0] = np.inf  # element with lowest obj values gets infinite sparsity
        sparsity[-1] = np.inf  # same with highest obj values

        for j in range(1, len(single_obj_sorted_candidates) - 1):
            sparsity[j] = sparsity[j] + ((objectives_sorted_by_single_obj[j + 1][obj_id] -
                                          objectives_sorted_by_single_obj[j - 1][obj_id]) / o_range)

    candidate_solutions = [x for _, x in
                           sorted(zip(sparsity, single_obj_sorted_candidates), reverse=True, key=lambda tu: tu[0])]
    return candidate_solutions


def tournament_selection(pop_costs, tournament_size=2, size=2):
    """
    A function which picks a number of random candidate solutions from the popultion and computes, which is the best one regarding rank and sparsity.
    Args:
        pop_costs (dict): A dictionary which containts the current population as keys and a list of their objective costs as values

    Returns:
        best: A candidate solution with a good rank and sparsity.
    """
    ranks = front_rank_assignment(pop_costs)
    population = list(pop_costs.keys())
    pop_sparsety_ordered = order_by_sparsity(population, pop_costs)
    best_list = []
    for _ in range(size):
        best = random.choice(population)
        for i in range(1, tournament_size):
            competitor = random.choice(population)
            rank_competitor = find_item_in_several_lists(ranks, competitor)[0]
            rank_best = find_item_in_several_lists(ranks, best)[0]
            if rank_competitor < rank_best:
                best = competitor
            elif rank_competitor == rank_best:
                if pop_sparsety_ordered.index(competitor) < pop_sparsety_ordered.index(best):
                    best = competitor
        best_list.append(best)
    return best_list


def fitness_proportionate_selection(pop_costs):
    """
    A function which picks a candidate solution with a propability proportionate to their fitness value.
    Args:
        pop_costs (dict): A dictionary which containts the current population as keys and a list of their objective costs as values

    Returns:
        Best: A candidate solution
    """
    objective_sums = list()
    candidates = list()

    # TODO proportionate to sum of objectives - different scales? re-evaluate.
    for objective_values in list(pop_costs.values()):
        objective_sums.append(sum(objective_values))

    # TODO checks if centre is 0 and shifts mean to 1; try to understand intuition.
    if sum(objective_sums) == 0:
        objective_sums = [elem + 1 for elem in objective_sums]

    # TODO consider np.cumsum;
    for i in range(1, len(objective_sums)):
        objective_sums[i] = objective_sums[i] + objective_sums[i - 1]

    for j in range(2):
        # TODO objective_sums[len(objective_sums) - 1] == objective_sums[-1] ?
        n = random.uniform(0, objective_sums[len(objective_sums) - 1])
        for i in range(1, len(objective_sums)):
            if objective_sums[i - 1] < n <= objective_sums[i]:
                new_candidate = list(pop_costs.keys())[i]
                candidates.append(new_candidate)
        if len(candidates) == j:
            new_candidate = list(pop_costs.keys())[0]
            candidates.append(new_candidate)

    # TODO consider return candidates
    return candidates[0], candidates[1]


def stochastic_universal_sampling(pop_costs, size=2):
    """
    A function which picks a candidate solution with a propability proportionate to their fitness value. If a solutio
    Args:
        pop_costs (dict): A dictionary which containts the current population as keys and a list of their objective costs as values

    Returns:
        Best: A candidat solution
    """
    objective_sums = list()
    candidates = list()

    for objective_values in list(pop_costs.values()):
        objective_sums.append(sum(objective_values))

    if sum(objective_sums) == 0:
        objective_sums = [elem + 1 for elem in objective_sums]

    for i in range(1, len(objective_sums)):
        objective_sums[i] = objective_sums[i] + objective_sums[i - 1]

    n = random.uniform(0, (objective_sums[-1] / size))
    for j in range(size):
        i = 0
        while objective_sums[i] < n:
            i = i + 1
        n = n + (objective_sums[-1] / size)
        new_candidate = list(pop_costs.keys())[i]
        candidates.append(new_candidate)

    return candidates[0], candidates[1]


# TODO: construct mutation decorator to lose f_i_range and magic_mutation_size
def line_recombination(candidate_a, candidate_b, f_i_range, magic_mutation_size=0.01):
    """
    A function, which takes two candidate solutions and crosses them over to generate two new candidate solutions.

    Args:
        candidate_a (Vm): A candidate solution.
        candidate_b (Vm): A candidate solution.
        f_i_range (float): The value range of the features and - if applicable - interactions

    Returns:
        Mutation(A_new), Mutation(B_new): Two lists, with the new candidate solutions.
    """
    a_feature_interaction_value_vector = candidate_a.get_feature_interaction_value_vector()
    b_feature_interaction_value_vector = candidate_b.get_feature_interaction_value_vector()
    alpha = np.random.rand(len(a_feature_interaction_value_vector), 1).T

    # TODO: split line into several statements
    a_feature_interaction_value_vector_new = np.add(np.multiply(alpha, a_feature_interaction_value_vector),
                                                    np.multiply(np.subtract(1, alpha),
                                                                b_feature_interaction_value_vector))
    b_feature_interaction_value_vector_new = np.add(np.multiply(alpha, b_feature_interaction_value_vector),
                                                    np.multiply(np.subtract(1, alpha),
                                                                a_feature_interaction_value_vector))
    a_new = deepcopy(candidate_a)
    a_new.set_feature_interaction_value_vector(a_feature_interaction_value_vector_new)
    b_new = deepcopy(candidate_b)
    b_new.set_feature_interaction_value_vector(b_feature_interaction_value_vector_new)
    a_mutated = mutate(a_new, 0, magic_mutation_size * f_i_range)
    b_mutated = mutate(b_new, 0, magic_mutation_size * f_i_range)
    return a_mutated, b_mutated


def sim_binary_co(A, B, f_i_range):
    """
    A function, which takes two candidate solutions and crosses them over to generate two new candidate solutions.

    Args:
        A (Vm): A candidate solution.
        B (Vm): A candidate solution.
        f_i_range (float): The value range of the features and - if applicable - interactions

    Returns:
        Mutation(A_new), Mutation(B_new): Two lists, with the new candidate solutions.
    """
    magic_one_percent_mutation_range = 0.01
    eta = 2
    A_new = deepcopy(A)
    B_new = deepcopy(B)
    rand = random.random()

    if rand <= 0.5:
        beta = math.pow((2.0 * rand), (1 / (eta + 1)))
    else:
        beta = math.pow((1 / (2.0 * (1 - rand))), (1 / (eta + 1)))

    features_a = A_new.get_feature_influences()
    features_b = B_new.get_feature_influences()
    for (key_a, val_a), (key_b, val_b) in zip(features_a.items(), features_b.items()):
        features_a[key_a] = 0.5 * (((1 + beta) * val_a) + ((1 - beta) * val_b))
        features_b[key_b] = 0.5 * (((1 - beta) * val_a) + ((1 + beta) * val_b))
    A_new.set_feature_influences(features_a)
    B_new.set_feature_influences(features_b)

    if A.uses_interactions():
        interactions_a = A_new.get_interaction_influences()
        interactions_b = B_new.get_interaction_influences()
        for (key_a, val_a), (key_b, val_b) in zip(interactions_a.items(), interactions_b.items()):
            interactions_a[key_a] = 0.5 * (((1 + beta) * val_a) + ((1 - beta) * val_b))
            interactions_b[key_b] = 0.5 * (((1 - beta) * val_a) + ((1 + beta) * val_b))
        A_new.set_interaction_influences(interactions_a)
        B_new.set_interaction_influences(interactions_b)

    mutated_a = mutate(A_new, 0, magic_one_percent_mutation_range * f_i_range)
    mutated_b = mutate(B_new, 0, magic_one_percent_mutation_range * f_i_range)
    return mutated_a, mutated_b


def mutate(A, mean, features_interactions_range, mutation_prob=1):  # intermediate recombination
    """
    A function, which takes a candidate solution and mutates them to generate a new candidate solution.

    Args:
        A (Vm): A candidate solution.
        features_interactions_range (float): The value range of the features and - if applicable - interactions

    Returns:
        A: A list with the new, mutated candidate solution.
    """
    A_mutated = deepcopy(A)

    mu = mean
    sigma = features_interactions_range  # standard deviation

    features_a = A.get_feature_influences()
    for key, val in features_a.items():
        if mutation_prob >= np.random.random_sample():
            s = np.random.normal(mu, sigma, 1)[0]
            features_a[key] = val + s
    A.set_feature_influences(features_a)

    if A.uses_interactions():
        interactions_a = A.get_interaction_influences()
        for key, val in interactions_a.items():
            if mutation_prob >= np.random.random_sample():
                s = np.random.normal(mu, sigma, 1)[0]
                interactions_a[key] = val + s
        A.set_interaction_influences(interactions_a)

    return A_mutated


def find_item_in_several_lists(my_list, item):
    # TODO this seems to yield (id_of_sublist, index_of_item_in_sublist for each sublist in my_list
    return [(ind, my_list[ind].index(item)) for ind in range(len(my_list)) if item in my_list[ind]]


def compute_fulfilled_objectives(avm, avm_modified, config, source_avm_performances):
    # Get selected change operations and their probability
    change_operations = config['Scope_for_Changes']['Change_Operations']
    change_probs = {}
    for operation in change_operations:
        change_probs[operation] = float(config[str(operation)]['Probability'])

    # get elements of dictionary, which where objected to modification
    if config['Scope_for_Changes']['Change_Feature'] == "most_influential":
        relevance_treshold = float(config['Scope_for_Changes']['Relevance_Treshhold'])
        feature_dict = most_influential(avm.get_feature_influences(), relevance_treshold)
        new_feature_dict = most_influential(avm_modified.get_feature_influences(), relevance_treshold)
    else:
        feature_dict = avm.get_feature_influences()
        new_feature_dict = avm_modified.get_feature_influences()
    model_dict = dict(feature_dict)
    new_model_dict = dict(new_feature_dict)

    if avm.uses_interactions():
        if config['Scope_for_Changes']['Change_Interaction'] == "most_influential":
            relevance_treshold = float(config['Scope_for_Changes']['Relevance_Treshhold'])
            interactions_dict = most_influential(avm.get_interaction_influences(), relevance_treshold)
            new_interactions_dict = most_influential(avm_modified.get_interaction_influences(), relevance_treshold)
            # model_dict.update(interactions_dict)
            # new_model_dict.update(new_interactions_dict)
        else:
            interactions_dict = avm.get_interaction_influences()
            new_interactions_dict = avm_modified.get_interaction_influences()
        model_dict.update(interactions_dict)
        new_model_dict.update(new_interactions_dict)

    all_fitness = {}
    for operation in change_operations:
        fitness = {
            'Noise_small': noise_small_objective,
            'Noise_big': noise_big_objective,
            'Linear_Transformation': linear_transformation_objective,
            'Negation': negation_objective,
        }[operation](new_model_dict, model_dict, change_probs[operation], config)

        all_fitness[operation] = fitness

    # TODO make same interface for Change_* as for other operations
    if 'Change_Feature' in config['Scope_for_Changes'] and config['Scope_for_Changes']['Change_Feature'] != "none":
        try:
            f_perc = float(config['Scope_for_Changes']['Change_Feature_percentage'])
        except:
            sys.exit("Change_Feature_percentage must be a float. Please check your configuration file!")
        fitness = feature_objective(new_feature_dict, feature_dict, f_perc)
        all_fitness['Change_Feature'] = fitness
    if avm.uses_interactions() and \
            config['Scope_for_Changes']['Change_Interaction'] != "none":
        try:
            i_perc = float(config['Scope_for_Changes']['Change_Interaction_percentage'])
        except:
            sys.exit("Change_Interaction_percentage must be a float. Please check your configuration file!")
        fitness = interactions_objective(new_interactions_dict, interactions_dict, i_perc)
        all_fitness['Change_Interaction'] = fitness

    if 'VariantCorrelation' in config:
        variant_corr_fitness = variant_corr_objective(avm_modified, source_avm_performances, config)
        all_fitness["VariantCorrelation"] = variant_corr_fitness

    all_fitnesses_list = list(all_fitness.values())
    return all_fitnesses_list


def feature_objective(new_model_dict, model_dict, p):
    differences = 0

    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] != model_dict[elem]:
            differences = differences + 1
    share_of_change = differences / len(list(model_dict.keys()))
    feature_fitness = 1 - abs(p - share_of_change)
    return feature_fitness


def interactions_objective(new_model_dict, model_dict, p):
    differences = 0

    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] != model_dict[elem]:
            differences = differences + 1
    share_of_change = differences / len(list(model_dict.keys()))
    interactions_fitness = 1 - abs(p - share_of_change)
    return interactions_fitness


def noise_small_objective(new_model_dict, model_dict, p, config):
    sigma = float(config['Noise_small']['Standard_deviation'])
    noise_fitness = noise_objective(new_model_dict, model_dict, p, sigma)
    return noise_fitness


def noise_big_objective(new_model_dict, model_dict, p, config):
    sigma = float(config['Noise_big']['Standard_deviation'])
    noise_fitness = noise_objective(new_model_dict, model_dict, p, sigma)
    return noise_fitness


def noise_objective(new_model_dict, model_dict, p, sigma):
    changed_fts = [elem for elem in list(new_model_dict.keys()) if new_model_dict[elem] != model_dict[elem]]
    # if only one feature has changed, the following will fail
    if len(changed_fts) < 2:
        return 0
    change_ratio = len(changed_fts) / len(list(model_dict.keys()))
    new_vals = [new_model_dict[ft] for ft in changed_fts]
    old_vals = [model_dict[ft] for ft in changed_fts]
    noise_deltas = list(np.array(new_vals) - np.array(old_vals))
    # noise_samples = sps.norm.rvs(loc=0, scale=sigma, size=500)
    noise_samples = sps.norm.rvs(loc=0, scale=sigma, size=len(noise_deltas))

    pears_distance = sps.pearsonr(noise_deltas, noise_samples)[0]
    # # plt.hist(noise_deltas, bins=50);plt.show()
    # norm_cdf = lambda x: sps.norm.cdf(x, scale=sigma)
    # ks_goodness_of_fit = sps.kstest(noise_deltas, norm_cdf)
    #
    # differences = 0
    # for elem in list(new_model_dict.keys()):
    #     if new_model_dict[elem] != model_dict[elem] and \
    #             (new_model_dict[elem] < model_dict[elem] - (model_dict[elem] * sigma) or
    #              new_model_dict[elem] > model_dict[elem] + (model_dict[elem] * sigma)):
    #         differences = differences + 1
    # change_ratio = differences / len(list(model_dict.keys()))

    noise_fitness = 1 - abs(p - change_ratio)
    ks_fitness = 1 - pears_distance
    return ks_fitness


def variant_corr_objective(avm_estimated, source_avm_performances, config):
    # TODO add guard
    target_corr = float(config["VariantCorrelation"]["target_correlation"])
    performances_vm_estimated = avm_estimated.calc_performance_for_validation_variants()
    # TODO move into objective function
    avm_sim = compute_similarities(source_avm_performances, performances_vm_estimated, config)
    fitness = abs(target_corr - avm_sim)
    return fitness


def linear_transformation_objective(new_model_dict, model_dict, p, config):
    differences = 0
    operation = str(config['Linear_Transformation']['Operation'])
    operand = float(config['Linear_Transformation']['Operand'])
    transform = {
        'addition': lambda x: x + x * operand,
        'substraction': lambda x: x - x * operand,
        'multiplication': lambda x: x * (x * operand),
        'division': lambda x: x / (x * operand)
    }.get(operation)

    for elem in list(new_model_dict.keys()):
        new_elem = transform(model_dict[elem])
        if new_model_dict[elem] != model_dict[elem] and \
                (new_elem == new_model_dict[elem]):
            differences = differences + 1
    change_ratio = differences / len(list(model_dict.keys()))
    lin_trans_fitness = 1 - abs(p - change_ratio)
    return lin_trans_fitness


def negation_objective(new_model_dict, model_dict, p, config):
    differences = 0
    for elem in list(new_model_dict.keys()):
        if new_model_dict[elem] == model_dict[elem] * -1:
            differences = differences + 1
    change_ratio = differences / len(list(model_dict.keys()))
    negation_fitness = 1 - abs(p - change_ratio)
    return negation_fitness


def breed_KT(archive_fitness, pop_size, config):
    # , pop_size, config
    """
    A function which takes the population of candidate solutions and uses it to generate a new set of candidate solutions.

    Args:
        P (dict): A dictionary which containts the current population as keys and a list of their objective costs as values
        P_dict(dict): A dictionary which containts the current population as keys and the current population split into features and (if applicable) interactions as dictionaries, where the keys are the names and the values are the values
        pop_size (int): The number of candidate solutions in the population.

    Returns:
        offsprings: A numpy matrix with the generated, new candidate solutions. The number of new candidate solutions is defined by the pop_size
    """
    offsprings = list()
    choice_sel_algo = str(config['NSGAII']['Selection_Algorithm'])
    choice_recombination_algo = str(config['NSGAII']['Recombination_Algorithm'])
    interaction_option = list(archive_fitness)[0].uses_interactions()

    # TODO find out why half population size is chosen or make configurable
    for i in range(0, pop_size, 4):
        # TODO make several lines out of this
        # TODO consider strategy pattern for selection algs
        parent_a, parent_b = {
            'tournament_selection': tournament_selection,
            'fitness_proportionate_selection': fitness_proportionate_selection,
            'stochastic_universal_sampling': stochastic_universal_sampling
        }.get(choice_sel_algo)(archive_fitness)
        # TODO make several lines out of this
        # TODO consider strategy pattern for cross-overs
        child_a, child_b = {
            'one_point_CO': one_point_co,
            'two_point_CO': two_point_co,
            'universal_CO': universal_co
        }[choice_recombination_algo](parent_a, parent_b, interaction_option)
        offsprings.append(child_a)
        offsprings.append(child_b)
    while len(offsprings) > pop_size / 2:
        offsprings = offsprings[:-1]

    return offsprings


def one_point_co(candidate_a, candidate_b, interaction_option):
    new_a = deepcopy(candidate_a)
    new_b = deepcopy(candidate_b)
    influences_a = new_a.get_feature_influences()
    influences_b = new_b.get_feature_influences()
    c = np.random.randint(len(influences_a))
    for _, (influence_a_name, influence_a_val), (influence_b_name, influence_b_val) \
            in zip(range(c), influences_a.items(), influences_b.items()):
        new_a.set_feature_influence(influence_b_name, influence_b_val)
        new_b.set_feature_influence(influence_a_name, influence_a_val)
    # TODO this does a second cross-over for interactions. re-evaluate this choice
    if interaction_option:
        interaction_influences_a = new_a.get_interaction_influences()
        interaction_influences_b = new_b.get_interaction_influences()
        c = np.random.randint(len(interaction_influences_a))
        for _, (influence_a_name, influence_a_val), (influence_b_name, influence_b_val) \
                in zip(range(0, c), interaction_influences_a.items(), interaction_influences_b.items()):
            new_a.set_interaction_influence(influence_b_name, influence_b_val)
            new_b.set_interaction_influence(influence_a_name, influence_a_val)
    return new_a, new_b


def two_point_co(candidate_a, candidate_b, interaction_option):
    new_a = deepcopy(candidate_a)
    new_b = deepcopy(candidate_b)
    influences_a = new_a.get_feature_influences()
    feature_names = list(influences_a)
    lower_cut_features = np.random.randint(len(feature_names))
    upper_cut_features = np.random.randint(len(feature_names))
    if lower_cut_features > upper_cut_features:
        lower_cut_features, upper_cut_features = upper_cut_features, lower_cut_features
    for pos in range(lower_cut_features, upper_cut_features):
        feature_name = feature_names[pos]
        old_feature_influence_a = new_a.get_feature_influence(feature_name)
        old_feature_influence_b = new_b.get_feature_influence(feature_name)
        new_a.set_feature_influence(feature_name, old_feature_influence_b)
        new_b.set_feature_influence(feature_name, old_feature_influence_a)

    # TODO this does a second cross-over for interactions. re-evaluate this choice
    if interaction_option:
        interaction_influences_a = new_a.get_interaction_influences()
        # TODO could shorten this with low, up = sorted(randint(...,2))
        lower_cut_interactions = np.random.randint(len(interaction_influences_a))
        upper_cut_interactions = np.random.randint(len(interaction_influences_a))
        if lower_cut_interactions > upper_cut_interactions:
            lower_cut_interactions, upper_cut_interactions = upper_cut_interactions, lower_cut_interactions
        interaction_names = list(interaction_influences_a)
        for pos in range(lower_cut_interactions, upper_cut_interactions):
            interaction_name = interaction_names[pos]
            old_feature_influence_a = new_a.get_interaction_influence(interaction_name)
            old_feature_influence_b = new_b.get_interaction_influence(interaction_name)
            new_a.set_interaction_influence(interaction_name, old_feature_influence_b)
            new_b.set_interaction_influence(interaction_name, old_feature_influence_a)
    return new_a, new_b


def universal_co(candidate_a, candidate_b, interaction_option, crossover_prob=0.5):
    new_a = deepcopy(candidate_a)
    new_b = deepcopy(candidate_b)
    influences_a = new_a.get_feature_influences()
    influences_b = new_b.get_feature_influences()
    for (influence_a_name, influence_a_val), (influence_b_name, influence_b_val) \
            in zip(influences_a.items(), influences_b.items()):
        t = np.random.random_sample()
        if t > crossover_prob:
            new_a.set_feature_influence(influence_b_name, influence_b_val)
            new_a.set_feature_influence(influence_a_name, influence_a_val)
    if interaction_option:
        interaction_influences_a = new_a.get_interaction_influences()
        interaction_influences_b = new_b.get_interaction_influences()
        for (influence_a_name, influence_a_val), (influence_b_name, influence_b_val) \
                in zip(interaction_influences_a.items(), interaction_influences_b.items()):
            t = np.random.random_sample()
            if t > crossover_prob:
                new_a.set_interaction_influence(influence_b_name, influence_b_val)
                new_a.set_interaction_influence(influence_a_name, influence_a_val)

    return new_a, new_b


def most_influential(dict_values, threshold):
    """
    A function which takes a list of items and returns the values that are larger than (threshold*100)% of the dataset

    Args:
        list_values (list): A list with float values

    Returns:
        A list with values that are larger than 75% of the dataset
    """
    list_values = list(dict_values.values())
    list_values.sort(reverse=True)
    culled_list = list_values[0:round(len(list_values) * threshold)]
    influential = {key: dict_values[key] for key in dict_values if dict_values[key] in culled_list}
    return influential


def compare_front_fitness(front):
    """
    A function, which calculates the mean total fitness of a list of candidate solutions

    Args:
        front (dict): A dictioray of candidate solutions.

    Returns:
        A float value, which represent the mean total similarity of a list of candidate solutions.
    """
    Total_O = list()

    for obj in list(front.values()):
        Total_O.append(sum(obj))

    Max_Fitness = max(Total_O)
    Mean_Fitness = sum(Total_O) / len(Total_O)

    return Max_Fitness, Mean_Fitness


def kde(data_list, size, bandwidth_=None, cv=3):
    """
    A function which to perform a kernel density estimation.

    Args:
        data (list): All values, which are used to perform the kernel density estimation. The intial distribution.

    Returns:
        A tuple of lists which contain x-axis coordinates for plotting the results (index [0])
        and the estimated values for the estimated distribution (index [1])

    """
    if not bandwidth_:
        bandwidth_ = 'auto'
    if str(bandwidth_) != "auto":
        auto_bandwith = False
        try:
            kde_bandwidth = float(bandwidth_)
        except:
            sys.exit("KDE_bandwidth must be float. Please check your configuration file!")
    else:
        auto_bandwith = True

    data_np = np.atleast_2d(data_list).swapaxes(0, 1)

    # use grid search cross-validation to optimize the bandwidth
    if auto_bandwith:
        params = {
            'bandwidth': np.logspace(-2, 2, 200)
        }
        grid = GridSearchCV(KernelDensity(), params, cv=cv, )
        grid.fit(data_np)
        bandwidth = grid.best_estimator_.bandwidth
    else:
        bandwidth = kde_bandwidth
    data__min = data_np[data_np != 0].min()
    data__max = data_np[data_np != 0].max()
    data_plot = np.linspace(data__min, data__max, size)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data_np)
    log_dens = kde.score_samples(data_plot)

    return data_plot, np.exp(log_dens)
