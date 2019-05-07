import yaml
import seaborn as sns
import numpy as np
import os
import datetime
import sys
import random
import pycosat
from random import shuffle
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import argparse

from thor.nsga2 import kde
from thor.nsga2 import front_rank_assignment, order_by_sparsity, compute_fulfilled_objectives, breed_KT, \
    most_influential, \
    compare_front_fitness, Nsga2

sns.set()


# ----------
# CONCATENATE FEATURES AND INTERACTIONS
# ----------

def concatenate(list_a, list_b):
    """
    A function to convert two lists into arrays and concatenate them.
    
    Args:
        list_a (list): Values for all features.
        list_b (list): Values for all interactions.
        
    Returns:
        An array with the concatenated feature and interaction values.
    
    """
    m_f = np.asarray(list_a)
    m_i = np.asarray(list_b)
    f_and_i = np.append(m_f, m_i)
    f_and_i = np.asmatrix(f_and_i)

    return f_and_i


class AvmGenerator:
    def __init__(self, conf_yaml):
        self.config = conf_yaml
        self.saver = Saver(self.config)

        # try:
        self.valid_variants_size = int(conf_yaml['Variants']['NumberOfVariants'])
        # except:
        #     sys.exit("NumberOfVariants must be an integer. Please check your configuration file!")

        sampling_method = conf_yaml['Variants']['Sampling_Method']

        # GET ATTRIBUTES FROM CONFIG-FILE
        avm_yaml = conf_yaml['AttributedModel']
        vm_yaml = conf_yaml['NonAttributedModel']
        self.n_jobs = int(self.config['NumberOfThreads']) if 'NumberOfThreads' in self.config else Vm.DEFAULT_JOBS
        sampling_yaml = self.config['Variants']
        self.avm = Vm(avm_yaml, self.valid_variants_size, sampling_yaml, n_jobs=self.n_jobs, is_attributed=True)
        self.vm = Vm(vm_yaml, self.valid_variants_size, sampling_yaml, n_jobs=self.n_jobs, is_attributed=False)
        print("Finished with creating variants")

    def run(self):
        best_front_dict = self.optimize()
        output_dir = self.saver.store_results(best_front_dict, self.avm)
        return output_dir

    # ----------
    # GENERATE INTERACTIONS
    # ----------

    def optimize(self):
        print('START OPTIMIZING')
        if self.config['AttributedModel']['With_Variants']:
            # NSGA-II
            print("Starting NSGA-II")
            nsga2_optimizer = Nsga2(self.config, self.n_jobs)
            best_front_dict = nsga2_optimizer.nsga2(self.avm, self.vm)
        # else:
        #     # Just KDE
        #     e_feature_list = estimation(feature_list, feature_list_for_estimation)
        #     e_feature_list_pure = list(e_feature_list.values())
        #     BestFront = [e_feature_list_pure]
        #
        #     # if provided: get initial values for estimated interactions
        #     if conf_yaml['AttributedModel']['With_Interactions'] == True:
        #         e_interaction_list = estimation(interactions_list, interaction_list_for_estimation)
        #         e_interactions_list_pure = list(e_interaction_list.values())
        #         BestFront = concatenate(e_feature_list_pure, e_interactions_list_pure).tolist()
        return best_front_dict


class AvmModificator():
    def __init__(self, conf_yaml):
        self.config = conf_yaml
        self.saver = Saver(self.config)
        # ========================
        # GET ATTRIBUTES FROM CONFIG-FILE
        # ========================
        avm_yaml = conf_yaml['AttributedModel']

        try:
            self.valid_variants_size = int(conf_yaml['Variants']['NumberOfVariants'])
        except:
            sys.exit("NumberOfVariants must be an integer. Please check your configuration file!")

        # sampling_method = str(self.config['Variants']['Sampling_Method'])
        sampling_yaml = self.config['Variants']
        self.n_jobs = int(self.config['NumberOfThreads']) if 'NumberOfThreads' in self.config else Vm.DEFAULT_JOBS
        self.avm = Vm(avm_yaml, self.valid_variants_size, sampling_yaml, n_jobs=self.n_jobs, is_attributed=True)
        print('created AVM')
        #
        # # ========================
        # # PREPARE FEATURE AND INTERACTION LISTS
        # # ========================
        # feature_list_pure = list(feature_list.values())
        #
        # if self.config['Model']['With_Interactions'] == True:
        #     interactions_list_pure = list(interactions_list.values())
        #     valid_complete_variants = append_interactions(valid_variants, feature_list, interactions_list)
        #     f_and_i = concatenate(feature_list_pure, interactions_list_pure)
        #     nsga_data = [feature_list, interactions_list]
        # else:
        #     valid_complete_variants = valid_variants
        #     f_and_i = np.asmatrix(feature_list_pure)
        #     nsga_data = [feature_list]

    def optimize(self):
        # ========================
        # START OPTIMIZING
        # ========================
        print("Starting NSGA-II")

        nsga2_optimizer = Nsga2(self.config, self.n_jobs)

        best_front_dict = nsga2_optimizer.nsga2_KT(self.avm)
        # BestFront_dict = self.nsga2_KT(nsga_data, valid_complete_variants, conf_yaml)
        return best_front_dict

    def run(self):
        best_front_dict = self.optimize()
        # ========================
        # COMMON AND DEAD FEATURES
        # ========================
        if 'Find_common_and_dead_features' in self.config['Search_Space'] and self.config['Search_Space'][
            'Find_common_and_dead_features']:
            bad_regions = self.avm.bad_region()
            print(bad_regions)

            # ========================
        # CREATE COHERENT DATASET
        # ========================
        fitness_scores = {curr_avm: curr_avm.calc_performance_for_validation_variants() for curr_avm in best_front_dict}
        # if self.avm.get_interaction_influences() is not None:
        #     old_data = [feature_list_pure, interactions_list_pure, fitness_scores]
        # else:
        #     old_data = [feature_list_pure, fitness_scores]
        best_front_dict_no_obj_names = {}
        for vm in best_front_dict:
            best_front_dict_no_obj_names[vm] = list(best_front_dict[vm])

        output_dir = self.saver.store_results(best_front_dict_no_obj_names, self.avm)
        return output_dir


class AvmComparison:
    def __init__(self):
        pass

    def plotting_KT(self, old_data, new_data, filepath, config):
        """
        A function which takes the original and modified features, interactions and fitness values and compares them with them help of plot diagrams

        Args:
            old_data (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the original data
            new_data (list): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the modified data

        """
        # instantiating stuff
        try:
            amount_bins = int(config['KDE']['NumberOfBins'])
        except:
            sys.exit("NumberOfBins must be an integer. Please check your configuration file!")

        # PREPARE THE DATA
        # feature values
        old_F = old_data[0]
        new_F = new_data[0]
        kde_old_F = kde(old_F, len(old_data[0]))
        kde_new_F = kde(new_F, len(new_data[0]))
        bin_old_F = np.linspace(min(old_F), max(old_F), amount_bins)
        bin_new_F = bin_old_F

        # variant fitness values
        old_V = old_data[-1]
        new_V = new_data[-1]
        kde_old_V = kde(old_V, old_data[-1].size)
        kde_new_V = kde(new_V, new_data[-1].size)
        bin_old_V = np.linspace(old_V[old_V != 0].min(), old_V[old_V != 0].max(), amount_bins)
        bin_new_V = bin_old_V

        if str(config['Model']['With_Interactions']) == "True":
            # real interaction values
            old_I = list(old_data[1])
            new_I = list(new_data[1])
            kde_old_I = kde(old_I, len(old_data[1]))
            kde_new_I = kde(new_I, len(new_data[1]))
            bin_old_I = np.linspace(min(old_I), max(old_I), amount_bins)
            bin_new_I = np.linspace(min(new_I), max(new_I), amount_bins)

        # INITIALIZE PLOT
        if str(config['Model']['With_Interactions']) == "True":
            fig = plt.figure(figsize=(30, 30))
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
            fig = plt.figure(figsize=(30, 20))
            oF = fig.add_subplot(231)
            nF = fig.add_subplot(232)
            F = fig.add_subplot(233)

            oV = fig.add_subplot(234)
            nV = fig.add_subplot(235)
            V = fig.add_subplot(236)

            # PLOT THE DATA
        oF.set_title("old feature values")
        oF.hist(old_F, bins=bin_old_F, fc="#a58a66", density=True, alpha=0.5)
        oF.plot(kde_old_F[0][:, 0], kde_old_F[1], linewidth=2, color="#ba824c", alpha=1)
        oF.set_xlabel('value')
        oF.set_ylabel('density')

        nF.set_title("new feature values")
        nF.hist(new_F, bins=bin_new_F, fc="#669ba5", density=True, alpha=0.5)
        nF.plot(kde_new_F[0][:, 0], kde_new_F[1], linewidth=2, color="#43676d", alpha=1)
        nF.set_xlabel('value')
        nF.set_ylabel('density')

        F.set_title("old and new feature values")
        F.hist(new_F, bins=bin_new_F, fc="#a58a66", density=True, alpha=0.5)
        F.hist(old_F, bins=bin_old_F, fc="#669ba5", density=True, alpha=0.5)
        F.plot(kde_new_F[0][:, 0], kde_new_F[1], linewidth=2, color="#43676d", alpha=1)
        F.plot(kde_old_F[0][:, 0], kde_old_F[1], linewidth=2, color="#ba824c", alpha=1)
        F.set_xlabel('value')
        F.set_ylabel('density')

        #######
        if str(config['Model']['With_Interactions']) == "True":
            oI.set_title("old interaction values")
            oI.hist(old_I, bins=bin_old_I, fc="#a58a66", density=True, alpha=0.5)
            oI.plot(kde_old_I[0][:, 0], kde_old_I[1], linewidth=2, color="#ba824c", alpha=1)
            oI.set_xlabel('value')
            oI.set_ylabel('density')

            nI.set_title("new interaction values")
            nI.hist(new_I, bins=bin_new_I, fc="#669ba5", density=True, alpha=0.5)
            nI.plot(kde_new_I[0][:, 0], kde_new_I[1], linewidth=2, color="#43676d", alpha=1)
            nI.set_xlabel('value')
            nI.set_ylabel('density')

            I.set_title("old and new interaction values")
            I.hist(old_I, bins=bin_old_I, fc="#a58a66", density=True, alpha=0.5)
            I.hist(new_I, bins=bin_new_I, fc="#669ba5", density=True, alpha=0.5)
            I.plot(kde_old_I[0][:, 0], kde_old_I[1], linewidth=2, color="#ba824c", alpha=1)
            I.plot(kde_new_I[0][:, 0], kde_new_I[1], linewidth=2, color="#43676d", alpha=1)
            I.set_xlabel('value')
            I.set_ylabel('density')

        ######

        oV.set_title("old variant values")
        oV.hist(old_V, bins=bin_old_V, fc="#a58a66", density=True, alpha=0.5)
        oV.plot(kde_old_V[0][:, 0], kde_old_V[1], linewidth=2, color="#ba824c", alpha=1)
        oV.set_xlabel('value')
        oV.set_ylabel('density')

        nV.set_title("new variant values")
        nV.hist(new_V, bins=bin_new_V, fc="#669ba5", density=True, alpha=0.5)
        nV.plot(kde_new_V[0][:, 0], kde_new_V[1], linewidth=2, color="#43676d", alpha=1)
        nV.set_xlabel('value')
        nV.set_ylabel('density')

        V.set_title("old and new variant values")
        V.hist(old_V, bins=bin_old_V, fc="#a58a66", density=True, alpha=0.5)
        V.hist(new_V, bins=bin_new_V, fc="#669ba5", density=True, alpha=0.5)
        V.plot(kde_old_V[0][:, 0], kde_old_V[1], linewidth=2, color="#ba824c", alpha=2)
        V.plot(kde_new_V[0][:, 0], kde_new_V[1], linewidth=2, color="#43676d", alpha=2)
        V.set_xlabel('value')
        V.set_ylabel('density')

        # save the plot
        plt.savefig(filepath + '/plots.png', bbox_inches='tight')
        plt.savefig(filepath + '/plots.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()


class Vm:
    DEFAULT_JOBS = 1

    def __init__(self, yml, variant_set_size, sampling_yaml, is_attributed=False, using_interactions=False,
                 n_jobs=-1):
        self.variant_set_size = variant_set_size
        self.is_attributed = is_attributed
        self.yml = yml
        dimacs_path = yml['DIMACS-file']
        self.n_jobs = n_jobs
        sampling_method = sampling_yaml['Sampling_Method']
        self.sampling_method = sampling_method
        self.num_variants = sampling_yaml['NumberOfVariants']
        self.perm_method = sampling_yaml['Permutation_Method'] if 'Permutation_Method' in sampling_yaml else None
        # self.is_attributed = False if 'New_Interactions_Specs' in yml else False

        feature_influence_file = yml['Feature-file']
        self.constraints = self.parse_dimacs(dimacs_path)
        self.feature_influences = self.parsing_text(feature_influence_file) if feature_influence_file else None
        # ========================
        # USE CONSTRAINTS TO GENERATE VALID VARIANTS
        print('USE CONSTRAINTS TO GENERATE VALID VARIANTS')
        # ========================

        # TODO: uniform interface

        if sampling_method != "random":
            self.valid_variants = self.get_valid_variants(self.constraints, len(self.feature_influences.keys()) - 1,
                                                          self.feature_influences)
        else:
            self.valid_variants = self.get_valid_variants(self.constraints, self.variant_set_size,
                                                          self.feature_influences)

        if is_attributed:
            self.interactions_specs = None
            if 'Interactions-file' in yml:
                # implies is_attributed == True
                interactions_influence_file = yml['Interactions-file']
                self.interactions_influence = self.parsing_text(
                    interactions_influence_file) if interactions_influence_file else None

                self.valid_complete_variants = self.annotate_interaction_coverage(self.valid_variants,
                                                                                  self.feature_influences,
                                                                                  self.interactions_influence)

            else:
                self.interactions_influence = None
                self.valid_complete_variants = self.valid_variants

        else:

            self.interactions_specs = yml['New_Interactions_Specs'] if 'New_Interactions_Specs' in yml else None
            if self.interactions_specs:
                # try:
                self.interactions_specs = list(map(int, self.interactions_specs))
                # except:
                #     sys.exit("Interaction_Specs must be a sequence of integers. Please check your configuration file!")
                self.interactions_influence = self.new_interactions(self.constraints,
                                                                    self.feature_influences,
                                                                    self.interactions_specs, self.n_jobs)

                self.valid_complete_variants = self.annotate_interaction_coverage(self.valid_variants,
                                                                                  self.feature_influences,
                                                                                  self.interactions_influence)
            else:
                self.interactions_influence = None
                self.valid_complete_variants = self.valid_variants

        print("initialized Vm")

    def bad_region(self):
        """
        A function which tries to find bad regions in a SAT problems search space.

        Args:
            self

        Returns:
            List of Features and their setting, which result in unsatisfied assignments for the SAT
        """
        bad_regions = []
        num_features = len(self.get_feature_influences())
        constraint_list = self.constraints
        for i in range(1, num_features + 1):
            c_copy = list(constraint_list)
            c_copy.append([i])
            if pycosat.solve(c_copy) == "UNSAT":
                bad_regions.append(-i)

            c_copy = list(constraint_list)
            c_copy.append([-i])
            if pycosat.solve(c_copy) == "UNSAT":
                bad_regions.append(i)
        return bad_regions

    def set_feature_influence(self, name, influence):
        self.feature_influences[name] = influence

    def set_interaction_influence(self, name, influence):
        self.interactions_influence[name] = influence

    def get_feature_influence(self, name):
        return self.feature_influences[name]

    def get_interaction_influence(self, name):
        return self.interactions_influence[name]

    def get_feature_influences(self):
        return self.feature_influences

    def get_interaction_influences(self):
        return self.interactions_influence

    def set_feature_influences(self, feature_influcences):
        self.feature_influences = feature_influcences

    def set_interaction_influences(self, interactions_influence):
        self.interactions_influence = interactions_influence

    def get_feature_num(self):
        return len(self.feature_influences)

    def get_interaction_num(self):
        if self.interactions_influence:
            n = self.interactions_influence
        else:
            n = self.interactions_specs
        return len(n)

    def uses_interactions(self):
        specified_inderactions = self.interactions_influence is not None
        specs_annotated = self.interactions_specs is not None
        result = specified_inderactions or specs_annotated
        return result

    def get_feature_interaction_value_vector(self):
        feature_influence_vals = list(self.get_feature_influences().values())
        if self.interactions_influence:
            interactions_list_pure = list(self.get_interaction_influences().values())
            feature_interaction_value_vector = concatenate(feature_influence_vals, interactions_list_pure)
        else:
            feature_interaction_value_vector = np.asmatrix(feature_influence_vals)
        return feature_interaction_value_vector

    def set_feature_interaction_value_vector(self, feature_interaction_value_vector):
        i = 0
        feature_interaction_value_list = feature_interaction_value_vector.ravel().tolist()[0]
        for key in self.get_feature_influences():
            self.set_feature_influence(key, feature_interaction_value_list[i])
            i += 1
        if self.interactions_influence:
            for key in self.get_interaction_influences():
                self.set_interaction_influence(key, feature_interaction_value_list[i])
                i += 1

    def get_feature_dump(self):
        lines = []
        for feature_name, influence in self.get_feature_influences().items():
            line = '{}: {}'.format(feature_name, influence)
            lines.append(line)
        dump_str = str(os.linesep).join(lines)
        return dump_str

    def get_interaction_dump(self):
        lines = []
        for feature_code, influence in self.get_interaction_influences().items():
            line = '{}: {}'.format(feature_code, influence)
            lines.append(line)
        dump_str = str(os.linesep).join(lines)
        return dump_str

    def parse_dimacs(self, path):
        """
        A function to parse a provided DIMACS-file.

        Args:
            path (str): The DIMACS-file's file path

        Returns:
            A list of lists containing all of the DIMACS-file's constrains. Each constrain is represented by a seperate sub-list.

        """
        dimacs = list()
        dimacs.append(list())
        with open(path) as mfile:
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

    def get_valid_variants(self, constraint_list, size, feature_influences):
        """
        A function to compute the valid variants of a model.

        Args:
            constraint_list (list): All constrains provided for the model.
            size (int): The desired number of variants for the model.

        Returns:
            A numpy matrix with variants, which satisfy the provided constrains. Each row represents one variant.
        """

        new_c = constraint_list.copy()

        perm_method = self.perm_method

        assert (perm_method in ["complete", "clauses", "no_permutation"]), (
            "Options for Permutation_Method are: complete, clauses, no_permutation")
        assert (self.sampling_method in ["random", "feature-wise", "pair-wise", "neg-feature-wise", "neg-pair-wise"]), (
            "Options for Sampling_Method are: random, feature-wise, neg-feature-wise, pair-wise, neg-pair-wise")

        sampling = {
            'feature-wise': lambda x, i, j: x.append([i]),
            'pair-wise': lambda x, i, j: x.extend([[i], [j]]),
            'neg-feature-wise': lambda x, i, j: x.append([-(i)]),
            'neg-pair-wise': lambda x, i, j: x.extend([[-i], [-j]])
        }.get(self.sampling_method)

        sol_collection = list()

        # substract root feature
        largest_dimacs_literal = len(feature_influences) - 1
        if not np.any([largest_dimacs_literal in sub_list for sub_list in constraint_list]):
            dummy_constraint = [largest_dimacs_literal, -1 * largest_dimacs_literal]
            new_c.append(dummy_constraint)

        if perm_method == "no_permutation" and self.sampling_method == "random":
            solutions = list(itertools.islice(pycosat.itersolve(new_c), size))
            for elem in solutions:
                solution = Vm.transform2binary(elem)
                sol_collection.append(solution)
        else:
            for i in range(0, size):
                if perm_method == "clauses" or perm_method == "complete":
                    shuffle(new_c)  # shuffle the constraints
                    if perm_method == "complete":
                        for constraint in new_c:  # shuffle feature assignment in constraints
                            shuffle(constraint)
                c_copy = list(new_c)
                if self.sampling_method != "random":
                    sampling(c_copy, i + 1, (i + 1) % (size) + 1)

                solution = pycosat.solve(c_copy)
                if solution != "UNSAT":
                    new_c.append([j * -1 for j in solution])
                    solution = Vm.transform2binary(solution)
                    sol_collection.append(solution)

        m_sol_list = np.asmatrix(sol_collection)

        return m_sol_list

    # # TODO
    # def calc_fitness_values(self, f_and_i, valid_complete_variants):
    #     # ========================
    #     # CALCULATE THE FITNESS OF THE AVM FOR EVERY VALID VARIANT
    #     # ========================
    #     print('CALCULATE THE FITNESS OF THE AVM FOR EVERY VALID VARIANT')
    #     fitness_scores = self.calc_performance(valid_complete_variants, f_and_i, f_and_i.shape[1])
    #     self..append(fitness_scores)

    # def prepare_features_interactions(self, ):
    #     #
    #     # PREPARE FEATURE AND INTERACTION LISTS
    #     print('PREPARE FEATURE AND INTERACTION LISTS')
    #
    #     if self.is_attributed:
    #         if self.using_interactions:
    #             # avm = [feature_list, interactions_list]
    #         else:
    #             # avm = [feature_list]
    #             # nsga_data = [feature_list, feature_list_for_estimation]
    #
    #     else:
    #         if self.using_interactions:
    #             e_valid_complete_variants = self.annotate_interaction_coverage(valid_variants_for_estimation,
    #                                                                            feature_list_for_estimation,
    #                                                                            interaction_list_for_estimation)
    #
    #
    #             avm = [feature_list, interactions_list]
    #             nsga_data = [feature_list, feature_list_for_estimation, interactions_list,
    #                          interaction_list_for_estimation]
    #         else:
    #             e_valid_complete_variants = valid_variants_for_estimation
    #
    #             f_and_i = np.asmatrix(feature_list_pure)
    #             avm = [feature_list]
    #             nsga_data = [feature_list, feature_list_for_estimation]
    #
    #
    #     return avm, e_valid_complete_variants, f_and_i, interaction_list_for_estimation, nsga_data,\
    #            valid_complete_variants

    # check if interaction is true and false for at least one variant,
    # as well as if the interaction members are true and false in at least one variant
    def check_interaction(self, c, f, random_features):
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

    def get_feature_and_influence_vector(self):
        feature_list_pure = list(self.feature_influences.values())
        if self.interactions_influence:
            interactions_list_pure = list(self.interactions_influence.values())
            feature_and_influence_vector = concatenate(feature_list_pure, interactions_list_pure)
        else:
            feature_and_influence_vector = np.asmatrix(feature_list_pure)
        return feature_and_influence_vector

    def new_interactions(self, constraint_list, f, specs, n_jobs):
        """
        A function to generate new interactions between features. The generation is handled by threads. The amount of threads is dependend on the number of different interaction degrees.

        Args:
            constraint_list (list): The previously aquired list (of lists) with all constraints of a model.
            f (dict): The models features.
            specs (list): The amount of new interactions, followed by value pairs for ratio in percent and interaction degree, e.g. [100, 50, 2, 50, 3].

        Returns:
            a dictionary with the new interactions as keys.
        """
        total_amount = specs[0]
        interaction_ratio = list(specs[1::2])
        interaction_degree = list(specs[2::2])

        all_new_interactions = dict()
        splitted_new_interactions = dict()
        for elem in interaction_degree:
            splitted_new_interactions["dict" + str(elem)] = {}

        if n_jobs > 0:
            number_of_threads = n_jobs
        else:
            number_of_threads = os.cpu_count()

        # some sweet, sweet error handling:
        assert (sum(interaction_ratio) == 100), (
            "The interaction ratios must sum up to 100. Currently they sum up to: ", sum(interaction_dist))

        def worker(amount):
            new_interactions = dict()
            for elem in range(len(interaction_degree)):
                if elem == len(interaction_degree) - 1:
                    amount_new_int = amount
                else:
                    amount_new_int = round(amount / 100.0 * interaction_ratio[elem])
                    amount = amount - amount_new_int

                while amount_new_int > len(splitted_new_interactions["dict" + str(interaction_degree[elem])]):
                    legit_int = False
                    while legit_int == False:
                        random_feature = list(np.random.choice(list(f.keys())[1:], interaction_degree[elem]))
                        if self.check_interaction(constraint_list, f, random_feature):
                            legit_int = True
                            random_feature = sorted(random_feature)
                            interaction = ""
                            for i in random_feature:
                                interaction = interaction + str(i) + "#"
                            interaction = interaction[:-1]
                            splitted_new_interactions["dict" + str(interaction_degree[elem])][interaction] = ""
                # new_interactions.update(new_int_degree_subdict)
            # return new_interactions

        pool = ThreadPool()
        l = [total_amount] * (number_of_threads)

        pool.map(worker, l)

        for elem in range(len(interaction_degree)):
            desired_amount = total_amount * interaction_ratio[elem] / 100
            while desired_amount < len(splitted_new_interactions["dict" + str(interaction_degree[elem])]):
                rchoice = random.choice(list(splitted_new_interactions["dict" + str(interaction_degree[elem])].keys()))
                del splitted_new_interactions["dict" + str(interaction_degree[elem])][rchoice]
            all_new_interactions.update(splitted_new_interactions["dict" + str(interaction_degree[elem])])

        pool.close()
        pool.join()

        print("Finished with creating interactions")

        return all_new_interactions

    def parsing_text(self, m):
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

    def parsing_variants(self, m):
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

    # ----------
    # HELPER FUNCTIONS
    # ----------
    @staticmethod
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

    def annotate_interaction_coverage(self, variants, feature_influences, interaction_influences):
        """
        A function which check for each variant, if they satisfy the previously provided (or estimated) interactions.
        It does so by looking up the involved features for each interaction and checking if those features are set to 1 for
        the respective variant. If so, the program appends a 1 (interaction satisfied) to the variant,
        else it append a 0 (interaction not satisfied).

        Args:
            variants (numpy matrix): All previously computed variants, which satisfy the provided constrains
            feature_influences (dict): All features with their names as keys and their values as values
            interaction_influences (dict): All interactions with feature tuples as keys and their values as values

        Returns:
            A numpy matrix with variants and information about which interactions they satisfy.
            Each row represents one variant and its interactions information.

        """

        valid_interaction = np.array([[1]])

        def check_for_interaction(row):
            for elem in interaction_influences.keys():
                valid_interaction[0, 0] = 1
                tokens = elem.split("#")
                for feature in tokens:
                    index = list(feature_influences.keys()).index(feature) - 1
                    if row[0, index] == 0:
                        valid_interaction[0, 0] = 0
                        break
                row = np.concatenate((row, valid_interaction), axis=1)  # np.insert(row, -1, valid_interaction)
            return row

        variants = np.apply_along_axis(check_for_interaction, axis=1, arr=variants)

        return variants

    # ----------
    # PERFORMANCE CALCULATION
    # ----------

    def calc_performance(self, variants):
        """
        A function to calculate the fitness/cost (depending on the model's application area) of all previously computed variants.

        Args:
            variants (numpy matrix): All previously computed variants with information about which interaction they satisfy
            f_and_i (numpy matrix): The provided or estimated values for all features and interactions

        Returns:
            An array of all variant fitnesses/costs.
        """
        feature_and_influence_vector = self.get_feature_and_influence_vector()
        root = np.ravel(feature_and_influence_vector)[0]
        variants = np.transpose(variants)
        # len_ratio = len_f_and_i / feature_and_influence_vector.shape[1]
        feature_and_influence_vector = np.delete(feature_and_influence_vector, 0, 1)
        m_fitness = np.dot(feature_and_influence_vector, variants)
        # if len_ratio != 1:
        #     m_fitness = m_fitness * len_ratio
        m_fitness = np.add(m_fitness, root)
        m_fitness = np.asarray(m_fitness)
        m_fitness = m_fitness.ravel()
        return m_fitness

    # ----------
    # PERFORMANCE CALCULATION
    # ----------

    def calc_performance_for_validation_variants(self):
        """
        A function to calculate the fitness/cost (depending on the model's application area) of all previously computed variants.

        Args:
             f_and_i (numpy matrix): The provided or estimated values for all features and interactions

        Returns:
            An array of all variant fitnesses/costs.
        """
        # variants = self.valid_variants
        variants = self.valid_complete_variants
        feature_and_influence_vector = self.get_feature_and_influence_vector()
        root = np.ravel(feature_and_influence_vector)[0]
        variants = np.transpose(variants)
        # len_ratio = len_f_and_i / feature_and_influence_vector.shape[1]
        feature_and_influence_vector = np.delete(feature_and_influence_vector, 0, 1)
        m_fitness = np.dot(feature_and_influence_vector, variants)
        # if len_ratio != 1:
        #     m_fitness = m_fitness * len_ratio
        m_fitness = np.add(m_fitness, root)
        m_fitness = np.asarray(m_fitness)
        m_fitness = m_fitness.ravel()

        return m_fitness


def main():
    random.seed()
    # ========================
    # GET LOCATION OF CONFIG-FILE
    # ========================
    parser = argparse.ArgumentParser(description='Thor2')
    parser.add_argument('path', metavar='config file path', type=str, help="the config's file path")
    args = parser.parse_args()
    config_location = args.path

    with open(config_location, 'r') as ymlfile:
        yml_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    for use_case_str in yml_cfg:
        cfg = yml_cfg[use_case_str]
        print(cfg)
        print("Performing ", use_case_str)

        n_jobs = cfg["NumberOfThreads"]
        result_choice = cfg["ResultsToBeSaved"]
        custom_result_specs = cfg["ResultsCustomSpecs"] if "ResultsCustomSpecs" in cfg else None
        output_dir = cfg["DirectoryToSaveResults"] if "DirectoryToSaveResults" in cfg else None

        avm_settings = cfg['AttributedModel']
        dimacs_file = avm_settings['DIMACS-file']
        feature_file = avm_settings['Feature-file']

        # avm = Vm(dimacs_file, feature_file, )

        if use_case_str == "AVM-Generation":
            avmgGen = AvmGenerator(cfg)
            output_path = avmgGen.run()
            print("The program terminated as expected.")
            print("Results saved to {}".format(output_path))

        elif use_case_str == "AVM-Modification":
            avmMod = AvmModificator(cfg)
            output_path = avmMod.run()
            print("The program terminated as expected.")
            print("Results saved to {}".format(output_path))
        else:
            print("Usage of wrong use-case")
            print("The two possible use-cases are:")
            print("AVM-Generation")
            print("AVM-Modification")
            print("")
            print("Please check your configuration files for the right notation!")


class Saver:
    def __init__(self, config):
        self.config = config

    # ----------
    # WRITING TO FILE
    # ----------

    def store_text(self, directory, dump_str, filename):
        """
        A function to write the new features and interactions to their respective txt-files.

        Args:
            directory (str): The path to the new txt-file
            old_data (dict): The dict with feature/interaction names
            new_data (dict): The dict with new/estimated feature/interaction values
            filename (str): Name for the new file

        """
        # TODO clean up os directory methods
        new_file = directory + "/" + filename + ".txt"

        print("Storing to {}".format(new_file))
        with open(new_file, 'w') as ffile:
            ffile.write(dump_str)

    # ----------
    # PLOTTING
    # ----------

    def store_plot(self, avm, vm, filepath):
        """
        A function which takes the given and estimated feature, interaction and fitness values and compares them with them help of plot diagrams

        Args:
            avm (Vm): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the attributed model's variants
            vm (Vm): A list that contains the feature values (dict), if provided interaction values (dict) and the fitness values/costs of the non-attributed model's variants
            filepath (str): path to the results folder

        """

        # instantiating stuff
        try:
            amount_bins = int(self.config['KDE']['NumberOfBins'])
        except:
            sys.exit("NumberOfBins must be an integer. Please check your configuration file!")

        # PREPARE THE DATA
        # real feature values
        values_rF = list(avm.get_feature_influences().values())
        values_kde_F = kde(values_rF, len(values_rF))
        values_eF = list(vm.get_feature_influences().values())
        bin_F = np.linspace(min(values_rF), max(values_rF), amount_bins)
        bin_eF = np.linspace(min(values_eF), max(values_eF), amount_bins)

        # real variant fitness values
        values_rV = avm.calc_performance_for_validation_variants()
        values_eV = vm.calc_performance_for_validation_variants()
        values_kde_V = kde(values_rV, values_eV.size)
        bin_V = np.linspace(values_rV[values_rV != 0].min(), values_rV[values_rV != 0].max(), amount_bins)
        bin_eV = np.linspace(values_eV[values_eV != 0].min(), values_eV[values_eV != 0].max(), amount_bins)

        with_interactions = avm.get_interaction_influences() is not None

        if with_interactions:
            # real interaction values
            values_rI = list(avm.get_interaction_influences().values())
            values_kde_I = kde(values_rI, len(vm.get_interaction_influences()))
            values_eI = list(vm.get_interaction_influences().values())
            bin_I = np.linspace(min(values_rI), max(values_rI), amount_bins)
            bin_eI = np.linspace(min(values_eI), max(values_eI), amount_bins)

            # INITIALIZE PLOT
        if with_interactions:
            fig = plt.figure(figsize=(30, 30))
            rF = fig.add_subplot(331)
            kdeF = fig.add_subplot(332)
            eF = fig.add_subplot(333)

            rI = fig.add_subplot(334)
            kdeI = fig.add_subplot(335)
            eI = fig.add_subplot(336)

            rV = fig.add_subplot(337)
            kdeV = fig.add_subplot(338)
            eV = fig.add_subplot(339)

        else:
            fig = plt.figure(figsize=(30, 20))
            rF = fig.add_subplot(231)
            kdeF = fig.add_subplot(232)
            eF = fig.add_subplot(233)

            rV = fig.add_subplot(234)
            kdeV = fig.add_subplot(235)
            eV = fig.add_subplot(236)

            # PLOT THE DATA
        rF.set_title("real Features")
        rF.hist(values_rF, bins=bin_F, fc="grey", density=True)
        rF.set_xlabel('value')
        rF.set_ylabel('density')

        kdeF.set_title("kde Features")
        kdeF.plot(values_kde_F[0][:, 0], values_kde_F[1], linewidth=2, color="grey", alpha=1)
        kdeF.hist(values_rF, bins=bin_F, density=True, fc="black", alpha=0.1)
        kdeF.set_xlabel('value')
        kdeF.set_ylabel('density')

        eF.set_title("estimated Features")
        eF.hist(values_eF, bins=bin_eF, density=True, fc="grey")
        eF.hist(values_rF, bins=bin_F, density=True, fc="black", alpha=0.1)
        eF.set_xlabel('value')
        eF.set_ylabel('density')

        #######
        if with_interactions:
            rI.set_title("real Interactions")
            rI.hist(values_rI, bins=bin_I, density=False, fc="grey", weights=np.ones(len(values_rI)) / len(values_rI))
            rI.set_xlabel('value')
            rI.set_ylabel('density')

            kdeI.set_title("kde Interactions")
            kdeI.plot(values_kde_I[0][:, 0], values_kde_I[1], linewidth=2, color="grey", alpha=1)
            kdeI.hist(values_rI, bins=bin_I, fc='black', alpha=0.1, density=True)
            kdeI.set_xlabel('value')
            kdeI.set_ylabel('density')

            eI.set_title("estimated Interactions")
            eI.hist(values_eI, bins=bin_eI, density=False, fc="grey", weights=np.ones(len(values_eI)) / len(values_eI))
            eI.hist(values_rI, bins=bin_I, density=False, fc="black", weights=np.ones(len(values_rI)) / len(values_rI),
                    alpha=0.1)
            eI.set_xlabel('value')
            eI.set_ylabel('density')

        ######

        rV.set_title("real Variants")
        rV.hist(values_rV, bins=bin_V, density=False, fc="grey", weights=np.divide(1, values_rV))
        rV.set_xlabel('value')
        rV.set_ylabel('density')

        kdeV.set_title("kde Variants")
        kdeV.plot(values_kde_V[0][:, 0], values_kde_V[1], linewidth=2, color="grey", alpha=1)
        kdeV.hist(values_rV, bins=bin_V, fc='black', alpha=0.1, density=True)
        kdeV.set_xlabel('value')
        kdeV.set_ylabel('density')

        eV.set_title("estimated Variants")
        eV.hist(values_eV, bins=bin_eV, density=False, fc="grey", weights=np.divide(1, values_eV))
        eV.hist(values_rV, bins=bin_V, density=False, fc="black", weights=np.divide(1, values_rV), alpha=0.1)
        eV.set_xlabel('value')
        eV.set_ylabel('density')

        # save the plot
        plt.savefig(filepath + 'plots.png', bbox_inches='tight')
        plt.savefig(filepath + 'plots.pdf', bbox_inches='tight')

        # show the plot
        # plt.show()
        plt.clf()
        plt.close()

    def define_results(self, best_front__dict, results_to_be_saved, results_custom_specs=None):
        assert (results_to_be_saved in ["all", "overall-best", "custom"]), (
            "Options for ResultsToBeSaved are: all, overall-best, custom")
        if results_to_be_saved == "all":
            best_front = list(best_front__dict.keys())

        elif results_to_be_saved == "overall-best":
            obj_values = list(best_front__dict.values())
            maximum = obj_values[0]
            for elem in obj_values:
                if sum(elem) > sum(maximum):
                    maximum = elem
            for solution, values in best_front__dict.items():
                if values == maximum:
                    best_front = [solution]

        elif results_to_be_saved == "custom":

            def weighted_sum(obj, specs):
                weighted = 0
                for i in range(0, len(obj)):
                    weighted = weighted + (obj[i] * specs[i])
                return weighted

            obj_values = list(best_front__dict.values())
            maximum = obj_values[0]
            for elem in obj_values:
                if weighted_sum(elem, results_custom_specs) > weighted_sum(maximum, results_custom_specs):
                    maximum = elem
            for solution, values in best_front__dict.items():
                if values == maximum:
                    best_front = [solution]

        return best_front

    def store_results(self, best_front_dict, avm):
        # ========================
        # SAVE THE RESULTS
        # ========================
        conf_yaml = self.config
        print("Finished with calculating results")
        print("Start saving results")
        # define name and path of the directory
        if str(conf_yaml['DirectoryToSaveResults']) != "auto":
            directory = str(conf_yaml['DirectoryToSaveResults'])
        else:
            directory = datetime.datetime.now().strftime("AVM-Generation_Results/Gen_results-%Y-%m-%d_%H%M%S")
        if not os.path.exists(directory):
            os.makedirs(directory)
        # define results to be saved
        result_selection = conf_yaml["ResultsToBeSaved"]
        result_custom_specs = conf_yaml["ResultsCustomSpecs"] if "ResultsCustomSpecs" in conf_yaml else None
        best_front = self.define_results(best_front_dict, result_selection, result_custom_specs)
        # save results
        for i, cur_vm in enumerate(best_front):
            if not os.path.exists(directory + "/result" + str(i + 1)):
                os.makedirs(directory + "/result" + str(i + 1))
            # write results into a txt-file:
            filedirectory = directory + "/result" + str(i + 1)

            # cur_vm = best_front[i]
            e_feature_list = cur_vm.get_feature_influences()
            # vm = list([e_feature_list])
            e_interactions_list = cur_vm.get_interaction_influences()  # may be None
            e_fitness_scores = cur_vm.calc_performance_for_validation_variants()
            # vm.append(e_fitness_scores)

            self.store_text(filedirectory, cur_vm.get_feature_dump(), "new_features")
            if e_interactions_list is not None:
                self.store_text(filedirectory, cur_vm.get_interaction_dump(), "new_interactions")

            # PLOTTING, SO WE CAN LOOK AT SOMETHING
            self.store_plot(avm, cur_vm, filedirectory + "/")
        print("Finished with saving results")
        return os.path.abspath(filedirectory)

    def store_results_modification(self, best_front_dict, avm):
        #     # save results
        #     feature_list_pure_new = BestFront[i][:len(feature_list_pure)]
        #     writing_text(filedirectory + "/", feature_list, feature_list_pure_new, "new_features")
        #     if self.config['Model']['With_Interactions'] == True:
        #         interactions_list_pure_new = BestFront[i][len(feature_list_pure):]
        #         f_and_i_new = concatenate(feature_list_pure_new, interactions_list_pure_new)
        #         writing_text(filedirectory + "/", interactions_list, interactions_list_pure_new, "new_interactions")
        #     else:
        #         f_and_i_new = np.asmatrix(feature_list_pure_new)
        #
        #     fitness_scores_new = performance(valid_complete_variants, f_and_i_new, f_and_i.shape[1])
        #
        #     if self.config['Model']['With_Interactions'] == True:
        #         new_data = [feature_list_pure_new, interactions_list_pure_new, fitness_scores_new]
        #     else:
        #         new_data = [feature_list_pure_new, fitness_scores_new]
        #
        #     if self.config['Search_Space']['Find_bad_regions'] == True:
        #         f = open(directory + "/bad_regions.txt", "w+")
        #         f.write(', '.join(map(str, np.array(bad_regions))))
        #         f.close()
        #
        #     # PLOTTING, SO WE CAN LOOK AT SOMETHING
        #     plotting_KT(old_data, new_data, filedirectory + "/")
        #
        # print("Finished with saving results")

        conf_yaml = self.config
        print("Finished with calculating results")
        print("Start saving results")
        # define name and path of the directory
        if str(conf_yaml['DirectoryToSaveResults']) != "auto":
            directory = str(conf_yaml['DirectoryToSaveResults'])
        else:
            directory = datetime.datetime.now().strftime("AVM-Generation_Results/Gen_results-%Y-%m-%d_%H%M%S")
        if not os.path.exists(directory):
            os.makedirs(directory)
        # define results to be saved
        result_selection = conf_yaml["ResultsToBeSaved"]
        result_custom_specs = conf_yaml["ResultsCustomSpecs"] if "ResultsCustomSpecs" in conf_yaml else None
        best_front = self.define_results(best_front_dict, result_selection, result_custom_specs)
        # save results
        for i, cur_vm in enumerate(best_front):
            if not os.path.exists(directory + "/result" + str(i + 1)):
                os.makedirs(directory + "/result" + str(i + 1))
            # write results into a txt-file:
            filedirectory = directory + "/result" + str(i + 1)
            # cur_vm = best_front[i]
            e_feature_list = cur_vm.get_feature_influences()
            # vm = list([e_feature_list])
            e_interactions_list = cur_vm.get_interaction_influences()  # may be None
            e_fitness_scores = cur_vm.calc_performance_for_validation_variants()
            # vm.append(e_fitness_scores)
            self.store_text(filedirectory, cur_vm.get_feature_dump(), "new_features")
            if e_interactions_list is not None:
                self.store_text(filedirectory, cur_vm.get_interaction_dump(), "new_interactions")
            # PLOTTING, SO WE CAN LOOK AT SOMETHING
            self.store_plot(avm, cur_vm, filedirectory + "/")
        print("Finished with saving results")
        return os.path.abspath(filedirectory)


if __name__ == "__main__":
    main()
