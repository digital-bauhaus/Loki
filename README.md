# Thor2
A tool to genererate authentic attributed variability models (AVM) and modify existings ones. 
With an AVM as input, it employs NSGA-II, a genetic algorithm, to determine attribute attribute values s.a. performance for an unattributed variability model (VM). With such models, algorithms working on AVM may be evaluated on large-scale software systems in feasible time.

## Getting started
Here are a few tips and recommendations on how to use thor2 in its current state.

### Dependencies
Berore running thor2, you need to install required Python modules, which are listed in [requirements.txt](requirements.txt). To install them, run `pip install -r requirements.txt`

### Run thor2
Run `python thor2.py my-conf.yaml`, where `python` is a Python 3 interpreter and  `my-conf.yaml` is your configuration file.

### Configure thor2
Thor2 uses a configuration file to carry user constraints and environment parameters. Being written in [YAML](https://yaml.org/),  all config options are specified within one task node. As seen in the example configs [generation-example.yaml](generation-example.yaml) and [modification-example.yaml](modification-example.yaml), the task node is `AVM-Generation` if you want to generate attribute values s.a. performance for an VM and `AVM-Modification` if you want to modify an AVM. 

It is advisable to start with the example configurations and test if your AVM files have the supported syntax for thor2. Next, you can adjust the settings to your needs, following the configuration option descriptions below. Note, however, that the configuration of thor2 is currently under reconstruction; hence, the current syntax might deviate slightly.

#### Generating AVM

##### [AttributedModel]
With_Variants:
Specification, whether the models work with constraints.
Possible Values: True, False

DIMACS-file:
The path, relative to thor2.py for the attributed model's DIMACS file with its constraints

Feature-file:
The path, relative to thor2.py for the text file, which containts the attributed model's features and feature values.
Each feature must be in a separate line, followed by a colon and its value.
example:
root: -291.062087712032
Feature_1: 1137.83599218529
Feature_2: 253.918568781176
Feature_3: 206.384366153135
 
With_Interactions:
Specification, whether the models work with interactions.
Possible Values: True, False

Interactions-file:
The path, relative to thor2.py for the text file, which contains the attributed model's interactions and interaction values.
Each interaction must be in a separate line. The involved features must be seperated by a hash/pound, followed by a colon and the interaction's value.
example:
Feature_1#Feature_2: 12.8163475344619
Feature_1#Feature_3: -110.381179569712
Feature_2#Feature_4#Feature_5: -98.3197288814209


#####  [NonAttributedModel]
DIMACS-file:
The path, relative to thor2.py for the non-attributed model's dimacs-file with its constraints

Feature-file:
The path, relative to thor2.py for the text file, which contains the non-attributed model's features.
Each feature must be in a separate line.

New_Interactions_Specs:
Specification, how many new interactions should be created in total, followed by value pairs for ratio in percent and interaction degree.
Example:
1000 new interaction shall be generated in total, 50% must involve 2 features, 30% must involve 3 features, 20% must involve 4 features. This instruction turns into:
1000, 50, 2, 30, 3, 20, 4

  

#####  [Variants]
Sampling_Method:
The method used to generate the set of variants, which will be used to calculate the model's performance distribution.
Possible settings:
random: finds the specified number of variant without any aditional restrictions (hint: this is SAT-based random sampling)
feature-wise: find one variant per feature where this feature is selected
neg-feature-wise: find one variant per feature where this feature is not selected
pair-wise: find one variant per feature pair, that is, a variant, in which two features are both selected
neg-pair-wise: find one variant per feature pair, that is, a variant in which two features are not selected

NumberOfVariants:
The number of solutions that shall be generated based on the provided DIMACS files. The variants are used for calculating performance values, which in turn are used for calculating the similarity between the generated model's performance distribution and the given model's performance distribution. 

Permutation_Method:
The method used to permutate the models' set of constraints to increase randomness in the SAT-based random variants.
Possible settings:
no_permutation: the constraints won't be permutated
clauses: the order of the constraint clauses are permutated
complete: the order of the constraint clauses and the order of the literals inside the clauses are permutated

  

##### [NSGAII]
Population_Size:
The number of candidate solutions which are generated, bred, and optimized. 

Maximum_Generations:
The upper bound of iterations for the NSGA-II algorithm. If the solutions converge before reaching this bound the algorithm will terminate earlier.
The default setting is "auto", which uses as many iterations as needed until the solution converges.

Selection_Algorithm:
The algorithm that chooses parents for the next generation.
Currently available: tournament_selection, fitness_proportionate_selection, stochastic_universal_sampling

Recombination_Algorithm:
The algorithm that is used for recombining the parents to generate new offspring
Currently available: Line_Recombination, simulated_binary_CO

Similarity_Measures:
The statistical tests to calculate the similarity between the given and the generated models. At least one must be selected. If several are selected, they have to be seperated by comma.
currently available: AD (Anderson-Darling-Test), ED (Euclidean distance), KS (Kolmogorov-Smirnov test), PCC (Pearson's correlation coefficient)
Example: AD, ED, PCC 

##### [Miscellaneous]
NumberOfThreads:
The number of threads to be used while generating interactions.
The default setting is "auto", which generates as many threads as cores are available. This can be changed to any integer value.

KDE_bandwidth:
The bandwidth, i.e. smoothing parameter, that is used for kernel density estimation.
The default setting is "auto", which calculates an appropriate bandwidth on the basis of cross-validation. This can be changed to any float value > 0.

NumberOfBins:
Number of bins for plotting the histogram. The more bins the more precise the histogram.

DirectoryToSaveResults: auto
The path to the directory, where the results will be saved.
The default setting is "auto", which generates a folder where thor2.py is located. The folder's name will be "results-[current date and time]". This can be changed to any folder on the machine. The specified folder will be created if it does not exist already.

ResultToSave:
Specification for which results shall be saved.
Possible settings:
all: saves all results
overall-best: saves the overall best result
custom: use custom weighting for the objectives to calcultate the best result

ResultsCustomSpecs:
Specification for how the different objective values shall be weighted.
If the model uses interactions, the sequence for the different objective values is: Features, Interactions, Variants
If the model does not use interactions, the sequence for the different objective values is: Features, Variants
The values' sum must round up 100. Only applicable if ResultToSave is set to "custom".
Examples:
The model uses interactions. The objective value for features shall be 50% of the total weight and the objective values for interactions and variants shall each be 25% of the total weight. This instruction turns into:
50, 25, 25
The model does not use interactions. The objective value for features shall be weighted 2 times as much as the objective value for the variants. This instruction turns into:
66.6, 33.3



#### Modifying AVM

#####  [Model]
With_Variants:
Specification, whether the models work with constraints.
Possible Values: True, False

DIMACS-file:
The path, relative to thor2.py for the attributed model's DIMACS file with its constraints

Feature-file:
The path, relative to thor2.py for the text file, which contains the attributed model's features and feature values.
Each feature must be in a separate line, followed by a colon and its value.
Example:
root: -291.062087712032
Feature_1: 1137.83599218529
Feature_2: 253.918568781176
Feature_3: 206.384366153135
 
With_Interactions:
Specification, whether the models work with interactions.
Possible Values: True, False

Interactions-file:
The path, relative to thor2.py for the text file, which contains the attributed model's interactions and interaction values.
Each interaction must be in a separate line. The involved features must be seperated by a hash/pound, followed by a colon and the interaction's value.
example:
Feature_1#Feature_2: 12.8163475344619
Feature_1#Feature_3: -110.381179569712
Feature_2#Feature_5: -98.3197288814209
  

##### [Variants]
Sampling_Method:
The method used to generate the set of variants, which will be used to calculate the model's performance distribution
Possible settings:
random: finds the specified number of variant without any additional restrictions (hint: SAT-based random sampling)
feature-wise: find one variant per feature where this feature is selected
neg-feature-wise: find one variant per feature where this feature is not selected
pair-wise: find one variant per feature pair, that is, a variant, in which two features are both selected
neg-pair-wise: find one variant per feature pair, that is, a variant in which two features are not selected

NumberOfVariants:
The number of solutions that shall be generated based on the provided DIMACS files. The variants are used for calculating performance values, which in turn are used for calculating the similarity.

Permutation_Method:
The method used to permutate the models' set of constraints to obtain more randomness in SAT-based random sampling.
Possible settings:
no_permutation: the constraints won't be permutated
clauses: the order of the constraint clauses are permutated
complete: the order of the constraint clauses and the order of the literals inside the clauses are permutated

##### [NSGAII]
Population_Size:
The number of candidate solutions that are generated, bred and optimized.

Maximum_Generations:
The upper bound of iterations for the NSGA-II algorithm. If the solutions converge before reaching this bound the algorithm will terminate earlier.
The default setting is "auto", which uses as many iterations as needed until the solution converges.
  
Selection_Algorithm:
The algorithm that chooses parents for the next generation.
Currently available: tournament_selection, fitness_proportionate_selection, stochastic_universal_sampling 

Recombination_Algorithm:
The algorithm that is used for recombining the parent to generate new offspring.
Currently available: one_point_CO, two_point_CO, universal_CO

##### [Scope_for_Changes]
Change_Feature:
Specification, which kind of feature changeds will be performed.
Possible Values: all, most-influential, none

Change_Feature_percentage:
The percentage of features from the selected specification, which will be modified.
Value must be a float between 0 and 1

Change_Interaction:
Specification, which kind of interaction shall be modified.
Possible Values: all, most-influential, none

Change_Interaction_percentage:
The percentage of interactions from the selected specification, which will be modified.
Value must be a float between 0 and 1

Relevance_Treshhold:
Defining which percent of features and interaction with the highest values are considered influencial.
Value must be a float between 0 and 1

Change_Operation:
The operations which shall be performed on the features and/or interactions. At least one must be selected.
Possible Values: Noise_small, Noise_big, Linear_Transformation, Negation

  

##### [Noise_small]

Probability:
The probability for adding Gaussian noise to a feature or interaction.

Mean:
The mean value for the normal distribution, which is used to generate noise.

Standard_deviation:
The value for the standard deviation for the normal distribution, which is used to generate noise.

[Noise_big]
Probability:
The probability for adding noise to a feature or interaction.

Mean:
The mean value for the normal distribution, which is used to generate noise.

Standard_deviation:
The value for the standard deviation for the normal distribution, which is used to generate noise.

  

##### [Linear_Transformation]
Probability:
The probability for performing a linear transformation on the feature or interaction.

Operation:
Specification, which operation will be performed
Possible Values: addition, substraction, division, multiplication

Operand:
The operand for the linear transformation. Can be any kind of float/real value.

##### [Negation]

Probability:
The probability for inverting the value of a feature or interaction.

##### [Search_Space]
Find_common_and_dead_features:
Specification, whether the programm should search for features that are selected (common) or unselected (dead) in every variant
Possible Values: True, False 

##### [Miscellaneous]
NumberOfThreads:
The number of threads to be used while generating interactions. The default setting is "auto", which generates as many threads as cores are available. This can be changed to any integer value.

KDE_bandwidth:
The bandwidth, i.e., smoothing parameter, that shall be used for the kernel density estimation. The default setting is "auto", which calculates an appropriate bandwidth on the basis of cross-validation. This can be changed to any float value > 0.

NumberOfBins:
Number of bins for plotting the histogram. The more bins the more precise the histogram.
DirectoryToSaveResults: auto

The path to the directory, where the results will be saved. The default setting is "auto", which generates a folder where thor2.py is located. The folder's name will be results-[current date and time]. This can be changed to any folder on the machine. The specified folder will be created if it does not exist already.

ResultToSave:
Specification for which results shall be saved. The default setting is "auto", which saves all results. It is also possible to just save the best result (overall-best) or use a custom weighting for the different objectives (custom).
Possible Values: all, overall-best, custom

## Trouble Shooting
There is the possibility of encountering the following error message with Anaconda, when running thor2:

```dask.async.IndexError: pop from empty list```

To solve this problem, please replace the file "base.py", which can be found under `Anaconda3\Lib\site-packages\sklearn` with the corrected "base.py" provided in the thor2-folder
