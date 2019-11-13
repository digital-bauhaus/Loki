import os
import numpy as np
import pycosat
import itertools
from random import shuffle

# from thor.thor2 import is_attributed, dimacs_path, feature_influence_file


class ThorAvm:
    DIMACS_FILE_CUES = ['dimacs', 'constraints']
    FEATURE_FILE_CUES = ['feature']
    INTERACTION_FILE_CUES = ['interaction']

    def __init__(self, dimacs_path, feature_file, interaction_file=None, sampling_method=None, perm_method=None):
        self.feature_influences = self.parse_influence_text(feature_file) if feature_file else None
        self.constraints = self.parse_dimacs(dimacs_path)
        self.interactions_influence = None

        self.sampling_method = "random" if sampling_method is None else sampling_method
        self.perm_method = "complete" if perm_method is None else perm_method

        if not dimacs_path or not feature_file:  # or not interaction_file:
            self.print("Assuming all files are in same folder as this python script (No complete set of paths passed)")
            own_path = os.path.dirname(__file__)
            for cur_file in os.listdir(own_path):
                low_dir = str(cur_file).lower()
                if np.any([cue in low_dir for cue in ThorAvm.DIMACS_FILE_CUES]):
                    self.dimacs_path = os.path.join(cur_file)
                if np.any([cue in low_dir for cue in ThorAvm.FEATURE_FILE_CUES]):
                    self.feature_file = os.path.join(cur_file)
                if np.any([cue in low_dir for cue in ThorAvm.INTERACTION_FILE_CUES]):
                    self.interaction_file = os.path.join(cur_file)
        else:
            self.dimacs_path = dimacs_path
            self.feature_file = feature_file
            self.interaction_file = interaction_file
        self.dimacs = self.parse_dimacs(self.dimacs_path)

        self.feature_influences = self.parse_influence_text(self.feature_file) if self.feature_file else None
        if interaction_file:
            self.interactions_influence = self.parse_influence_text(
                self.interaction_file) if interaction_file else None
        else:
            self.interactions_influence = None

    def print(self, content):
        pre = str(self)
        template = '[{}] {}'
        output = template.format(pre, content)
        print(output)

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
        # root = np.ravel(feature_and_influence_vector)[0]
        variants = np.transpose(variants)
        # feature_and_influence_vector = np.delete(feature_and_influence_vector, 0, 1)
        m_fitness = np.dot(feature_and_influence_vector, variants)
        # m_fitness = np.add(m_fitness, root)
        m_fitness = np.asarray(m_fitness)
        m_fitness = m_fitness.ravel()
        return m_fitness

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

    def get_feature_influence(self, name):
        return self.feature_influences[name]

    def get_interaction_influence(self, name):
        return self.interactions_influence[name]

    def get_feature_influences(self):
        return self.feature_influences

    def get_interaction_influences(self):
        return self.interactions_influence

    def get_feature_num(self):
        return len(self.feature_influences)

    def get_interaction_num(self):
        if self.interactions_influence:
            n = self.interactions_influence
        else:
            n = 0
        return len(n)

    def uses_interactions(self):
        specified_inderactions = self.interactions_influence is not None
        result = specified_inderactions
        return result

    def get_feature_interaction_value_vector(self):
        feature_influence_vals = list(self.get_feature_influences().values())
        if self.interactions_influence:
            interactions_list_pure = list(self.get_interaction_influences().values())
            feature_interaction_value_vector = concatenate(feature_influence_vals, interactions_list_pure)
        else:
            feature_interaction_value_vector = np.asmatrix(feature_influence_vals)
        return feature_interaction_value_vector

    def get_feature_and_influence_vector(self):
        feature_list_pure = list(self.feature_influences.values())
        if self.interactions_influence:
            interactions_list_pure = list(self.interactions_influence.values())
            feature_and_influence_vector = concatenate(feature_list_pure, interactions_list_pure)
        else:
            feature_and_influence_vector = np.asmatrix(feature_list_pure)
        return feature_and_influence_vector

    def parse_influence_text(self, m):
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

    def get_valid_variants(self, size):
        """
        A function to compute the valid variants of a model.

        Args:
            constraint_list (list): All constrains provided for the model.
            size (int): The desired number of variants for the model.

        Returns:
            A numpy matrix with variants, which satisfy the provided constrains. Each row represents one variant.
        """
        new_c = self.constraints.copy()
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
        largest_dimacs_literal = len(self.feature_influences) - 1
        if not np.any([largest_dimacs_literal in sub_list for sub_list in self.constraints]):
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
                    sampling(c_copy, i + 1, (i + 1) % size + 1)

                solution = pycosat.solve(c_copy)
                if solution != "UNSAT":
                    new_c.append([j * -1 for j in solution])
                    solution = ThorAvm.transform2binary(solution)
                    sol_collection.append(solution)
        m_sol_list = np.asmatrix(sol_collection)
        return m_sol_list

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
