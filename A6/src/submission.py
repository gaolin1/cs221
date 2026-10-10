#!/usr/bin/python
"""
XCS221 Homework 6: Bayesian Networks
"""

from util import (
    BayesianNetwork, BayesianNode, init_zero_conditional_probability_tables,
    normalize_counts, load_annotation_csv, plot_annotator_cpts, plot_label_cpt
)
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict

############################################################
# Problem 2a: Converting Phylogenetic Tree to Bayesian Network

def initialize_phylogenetic_tree(mutation_rate: float, genome_length: int=1) -> BayesianNetwork:
    """
    Initialize the phylogenetic tree as a Bayesian network using the BayesianNode and
    Bayesian Network classes in util.py.
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1].")
    pass
    # ### START CODE HERE ###
    # initializes dna sequence
    dna_domain = ["A", "C", "T", "G"]
    domain_length = len(dna_domain)
    thomas_bayus = BayesianNode("Thomas bayus", dna_domain, None, None)
    # fills the table with 0.333
    mutate_table = np.full(shape=(domain_length, domain_length), fill_value=mutation_rate/3)
    # then fill in the diagonal e.g. "C-C" with 0.9 so the sum of P is 1
    np.fill_diagonal(mutate_table, 1 - mutation_rate)
    humblus_studentus = BayesianNode("Humblus studentus", dna_domain, [thomas_bayus], mutate_table)
    aryamus_bayus = BayesianNode("Aryamus bayus", dna_domain, [thomas_bayus], mutate_table)
    kenius_bayus = BayesianNode("Kenius bayus", dna_domain, [aryamus_bayus], mutate_table)
    # ### END CODE HERE ###
    network = BayesianNetwork([aryamus_bayus, humblus_studentus, thomas_bayus, kenius_bayus], batch_size=genome_length)
    return network

############################################################
# Problem 2b: Sampling from Bayesian Networks

def forward_sampling(network: BayesianNetwork) -> Dict[str, str]:
    """
    Sample a single observation from the given Bayesian network.

    Use the topological ordering of variables in network.order and sample each
    variable according to its conditional probability distribution given the values
    of its parents (which have already been sampled).

    Args:
        network: A BayesianNetwork object containing nodes to sample from

    Returns:
        A dictionary mapping variable names to their sampled values
    """
    samples: Dict[str, List[str]] = {node.name: [] for node in network.order}

    for idx in range(network.batch_size):
        assignment: Dict[str, str] = {}
        # ### START CODE HERE ###
        # the ouput looks something like [[node_name as key, [idx_one, idx_two]]]
        for node in network.order:
            node_name = node.name
            choices = node.domain
            probabilty = []
            for choice in choices:
                if not node.parents:
                    choice_probability = node.get_probability(value=choice)[idx]
                else:
                    parents = node.parents
                    parent_dict = {}
                    for parent in parents:
                        parent_name = parent.name
                        parent_dict[parent_name] = assignment[parent_name]
                    choice_probability = node.get_probability(value=choice, parent_values=parent_dict)
                probabilty.append(choice_probability)
            node_choice = np.random.choice(choices, p=probabilty)
            assignment[node_name] = node_choice
            samples[node_name].append(node_choice)
        # ### END CODE HERE ###

    return samples

############################################################
# Problem 2c: Computing Joint Probability

def compute_joint_probability(
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    batch_indices: Optional[List[int]] = None
) -> float:
    """
    Compute the joint probability of a given assignment to all variables in the network.

    Args:
        network: A BayesianNetwork object
        assignment: Dictionary mapping variable names to their assigned values
        batch_indices: Optional list of batch indices to compute the joint probability for
            (default is [0, 1, ..., batch_size-1], so all the indices)

    Returns:
        The joint probability as a float
    """
    
    # ### START CODE HERE ###
    joint_probability = 1
    if batch_indices is None:
        running_indices = list(range(max(1, network.batch_size)))
    else:
        running_indices = batch_indices
    
    for indice in range(len(running_indices)):
        idx = running_indices[indice]
        for node in network.order:
            node_name = node.name 
            assignment_value = assignment[node_name][indice]
            # parents are empty if parent
            if not node.parents:
                assignemnt_probability = node.get_probability(value=assignment_value)[idx]
            else:
                parents = node.parents
                parents_dict = {}
                for parent in parents:
                    parent_name = parent.name
                    parent_value = assignment[parent_name][indice]
                    parents_dict[parent_name] = parent_value
                assignemnt_probability = node.get_probability(value=assignment_value, parent_values=parents_dict)
            joint_probability = joint_probability * assignemnt_probability
    return joint_probability
    # ### END CODE HERE ###

# ############################################################
# # Problem 2d: Test forward sampling

def test_forward_sampling():
    np.random.seed(123)
    random.seed(123)
    pass
    # ### START CODE HERE ###
    test_network = initialize_phylogenetic_tree(mutation_rate=0.1, genome_length=10)
    sample = forward_sampling(network=test_network)
    joint_probability = compute_joint_probability(network=test_network, assignment=sample)
    # ### END CODE HERE ###
    print(sample)
    print(f"{joint_probability:.10%}")

# Uncomment to test forward sampling
# test_forward_sampling()

############################################################
# Problem 2e: Rejection Sampling

def rejection_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_samples: int
) -> Dict[Any, float]:
    """
    Use rejection sampling to estimate the likelihoods for each outcome in the 
    target variable conditioned on the given assignments (a dictionary mapping
    variable names to their assigned values), i.e. P(target_variable | conditioned_on_assignments).

    Args:
        network: A BayesianNetwork object
        target_variable: The name of the variable to estimate the likelihoods for
        conditioned_on_assignments: A dictionary mapping variable names to their assigned values
        num_samples: The number of samples to draw

    Returns:
        A dictionary mapping outcomes to their likelihoods, conditioned on the given assignments.
    """
    
    # ### START CODE HERE ###
    reject_sample = {}
    final_sample = {}
    on_condition_samples_count = 0
    for sample_idx in range(num_samples):
        true_sample = False
        sample = forward_sampling(network=network)
        for conditional_variable in conditioned_on_assignments.keys():
            sample_on_condition = sample[conditional_variable]
            if sample_on_condition == conditioned_on_assignments[conditional_variable]:
                true_sample = True
            else:
                true_sample = False
                break
        if true_sample:
            target_assignment = tuple(sample[target_variable])
            if target_assignment in reject_sample:
                reject_sample[target_assignment] += 1
            else:
                reject_sample[target_assignment] = 1
            on_condition_samples_count += 1
    for target in reject_sample.keys():
        target_probability = reject_sample[target]/on_condition_samples_count
        final_sample[target] = target_probability
    return final_sample
    # ### END CODE HERE ###

############################################################
# Problem 2f: Gibbs Sampling

def gibbs_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_iterations: int,
    initial_state: Dict[str, List[str]] = None
) -> Dict[Any, float]:
    """
    Estimate P(target_variable | conditioned_on_assignments) via Gibbs sampling.

    - Initialization: random ancestral sample (forward_sampling)
    - Use compute_joint_probability to score single-variable proposals
      (simple and robust for small networks)
    - Record the Thomas bayus genome after each full sweep
    """
    # random initialization, then set evidence variables
    state = forward_sampling(network) if initial_state is None else initial_state
    for evidence_var, evidence_val in conditioned_on_assignments.items():
        state[evidence_var] = evidence_val

    # resample all non-evidence vars each sweep
    resample_nodes = [n for n in network.order if n.name not in conditioned_on_assignments]
    counts = defaultdict(int)

    for _ in range(num_iterations):
        # ### START CODE HERE ###
        for resample_node in resample_nodes:
            choices = resample_node.domain
            resample_node_name = resample_node.name
            for idx in range(network.batch_size):
                choice_probabilities = []
                for choice in choices:
                    sample_state = state
                    sample_state[resample_node_name][idx] = choice
                    choice_probability = compute_joint_probability(network=network, assignment=sample_state)
                    choice_probabilities.append(choice_probability)
                # normalize to 1
                choice_probability_total = sum(choice_probabilities)
                resample_probability = choice_probabilities/choice_probability_total
                new_choice = np.random.choice(choices,p=resample_probability)
                state[resample_node_name][idx] = new_choice
        target_value = tuple(state[target_variable])
        if target_value in counts:
            counts[target_value] += 1
        else:
            counts[target_value] = 1
        # ### END CODE HERE ###
    total_samples = sum(counts.values())
    return {val: counts[val] / total_samples for val in counts.keys()}

def test_gibbs_vs_rejection(
    num_steps: int=10000,
    seed: int=0,
    mutation_rate: float=0.1,
    genome_length: int=4,
):
    def exact_inference():
        same = 1.0 - mutation_rate
        diff = mutation_rate / 3.0
        p_a = same * same + 3 * diff * diff
        p_not_a = diff * same + same * diff + 2 * diff * diff
        expected = (p_a ** 3) * p_not_a
        return expected
    np.random.seed(seed)
    random.seed(seed)
    network = initialize_phylogenetic_tree(mutation_rate=mutation_rate, genome_length=genome_length)
    kenius_bayus_genome = ['A'] * genome_length

    vals = gibbs_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_iterations=num_steps)
    gibbs_p_aaac = vals.get(('A', 'A', 'A', 'C'), 0.0)

    rejection_vals = rejection_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_samples=num_steps)
    rejection_p_aaac = rejection_vals.get(('A', 'A', 'A', 'C'), 0.0)

    print(f"{'Gibbs':>10} {gibbs_p_aaac:.4f}")
    print(f"{'Rejection':>10} {rejection_p_aaac:.4f}")
    print(f"{'Exact':>10} {exact_inference():.4f}")

# Uncomment to test Gibbs vs. rejection sampling
#test_gibbs_vs_rejection(num_steps=100, mutation_rate=0.1, genome_length=4)
#test_gibbs_vs_rejection(num_steps=10000, mutation_rate=0.1, genome_length=4)

############################################################
# Problem 3d: Bayesian network for annotators

def bayesian_network_for_annotators(num_annotators: int, dataset_size: int=1) -> BayesianNetwork:
    """
    Return the Bayesian network for the annotators.
    """
    # ### START CODE HERE ###
    outcome_domain = ["good", "bad"]
    domain_length = len(outcome_domain)
    data_node = BayesianNode("Y", outcome_domain, None, None)
    annotator_original_table = np.full(shape=(domain_length, domain_length), fill_value=0.4)
    np.fill_diagonal(annotator_original_table, val=0.6)
    nodes_list = [data_node]
    for annotator_index in range(num_annotators):
        annotator_name = "A_" + str(annotator_index)
        annotator_table = annotator_original_table.copy()
        annotator_node = BayesianNode(annotator_name, outcome_domain, [data_node], annotator_table)
        nodes_list.append(annotator_node)
    network = BayesianNetwork(nodes_list, batch_size=dataset_size)
    return network
    # ### END CODE HERE ###

############################################################
# Problem 3e: Maximum likelihood estimation


def accumulate_assignment(
    counts: Dict[str, np.ndarray],
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    weight: float = 1.0,
    batch_indices: Optional[List[int]] = None,
) -> None:
    """
    Add weighted counts for a fully or partially observed assignment.
    """
    batch_size = len(list(assignment.values())[0])
    for i in range(batch_size) if batch_indices is None else range(len(batch_indices)):
        idx = batch_indices[i] if batch_indices is not None else i
        assignment_i = {k: v[i] for k, v in assignment.items()}
        for node in network.nodes:
            # ### START CODE HERE ###
            node_name = node.name
            node_value = assignment_i[node_name]
            node_index = node.domain.index(node_value)
            if not node.parents:
                counts[node_name][idx][node_index] += weight
            else:
                node_parent_index = node.parent_assignment_indices(assignment_i)
                counts[node_name][node_parent_index][node_index] += weight
            # ### END CODE HERE ###


def mle_estimation(network: BayesianNetwork, data: List[Dict[str, List[str]]], lambda_param: float = 1.0) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE.
    """
    
    # ### START CODE HERE ###
    for assignment_idx, assignment in enumerate(data):
        if assignment_idx == 0:
            current_count = init_zero_conditional_probability_tables(network=network)
        accumulate_assignment(counts=current_count, network=network, assignment=assignment)
    lambda_count = {}
    for parameter, parameter_value in current_count.items():
        new_value = parameter_value + lambda_param
        lambda_count[parameter] = new_value
    normalize_counts(network=network, counts=lambda_count)
    return network
    # ### END CODE HERE ###

############################################################
# Problem 3f: Maximum likelihood estimation for annotators

def mle_estimation_for_annotators(data: List[Dict[str, List[str]]]) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE for the annotators.
    """
    
    # ### START CODE HERE ###
    num_annotators = 0
    data_point = data[0]
    data_point_keys = list(data_point.keys())
    total_labels = len(data_point_keys)
    if "Y" in data_point_keys:
        num_annotators = total_labels - 1
    else:
        num_annotators = total_labels
    dataset_size = len(data_point[data_point_keys[0]])
    annotator_network = bayesian_network_for_annotators(num_annotators=num_annotators, dataset_size=dataset_size)
    mle_estimation(network=annotator_network, data=data)
    return annotator_network
    # ### END CODE HERE ###

def test_mle_estimation_for_annotators():
    data = load_annotation_csv('data/annotations.csv', include_labels=True)
    trained = mle_estimation_for_annotators(data)
    plot_annotator_cpts(trained, "plots/annotators.png")

#test_mle_estimation_for_annotators()

############################################################
# Problem 4a: Expectation step

def e_step(
    network: BayesianNetwork, data: List[Dict[str, List[str]]]
) -> Tuple[List[Dict[str, List[str]]], List[float], List[List[int]]]:
    """
    Create the dataset of fully-observed weighted observations given some hidden variables, for the EM algorithm.
    """
    # ### START CODE HERE ###
    completed_list = []
    q_weights_list = []
    position_list = []
    for data_point in data:
        data_lines = []
        data_point_keys = data_point.keys()
        data_point_keys_list = list(data_point_keys)
        data_point_size = len(data_point[data_point_keys_list[0]])
        # build data lines down on positions
        for position_index in range(data_point_size):
            position_data_point = {}
            for data_key in data_point_keys_list:
                position_data_point[data_key] = [data_point[data_key][position_index]]
            data_lines.append(position_data_point)
        # loop over individual data lines
        for data_line_index, data_line in enumerate(data_lines):
            missing_nodes = {}
            for node in network.nodes:
                node_name = node.name
                if node_name not in data_line.keys():
                    choices = node.domain
                    missing_nodes[node_name] = choices
            missing_choices = product(*missing_nodes.values())
            missing_probabilities = []
            for missing_choice_result in missing_choices:
                sample_copy = data_line.copy()
                for missing_choice_index, missing_choice in enumerate(missing_choice_result):
                    missing_node = list(missing_nodes.keys())[missing_choice_index]
                    sample_copy[missing_node] = [missing_choice]
                completed_list.append(sample_copy)
                position_list.append([data_line_index])
                probability = compute_joint_probability(network=network, assignment=sample_copy, batch_indices=[data_line_index])
                missing_probabilities.append(probability)
            missing_sum = sum(missing_probabilities)
            q_weights_list.extend(missing_probabilities / missing_sum)
    return [completed_list, q_weights_list, position_list]
    # ### END CODE HERE ###

############################################################
# Problem 4b: Maximization step


def m_step(
    network: BayesianNetwork,
    all_completions: List[Dict[str, List[str]]],
    all_weights: List[float],
    all_indices: List[List[int]],
) -> BayesianNetwork:
    """
    Update the CPTs of the Bayesian network using expected counts.
    """
    
    # ### START CODE HERE ###
    initial_counts = init_zero_conditional_probability_tables(network=network)
    for data_index, completed_data in enumerate(all_completions):
        if data_index == 0:
            current_counts = initial_counts
        current_weight = all_weights[data_index]
        current_indice = all_indices[data_index]
        accumulate_assignment(counts=current_counts, network=network, assignment=completed_data, weight=current_weight, batch_indices=current_indice)
    normalize_counts(network=network, counts=current_counts)
    return network
    # ### END CODE HERE ###


############################################################
# Problem 4c: Expectation maximization

def em_learn(network: BayesianNetwork, data: List[Dict[str, str]], num_iterations: int) -> BayesianNetwork:
    """
    Run the EM algorithm for a given number of iterations.
    """
    # ### START CODE HERE ###
    for _ in range(num_iterations):
        all_completed, all_weights, all_indices = e_step(network=network, data=data)
        network = m_step(network=network, all_completions=all_completed, all_weights=all_weights, all_indices=all_indices)
    return network
    # ### END CODE HERE ###

def test_em_learn():
    network = bayesian_network_for_annotators(num_annotators=3, dataset_size=100)
    data = load_annotation_csv('data/annotations.csv', include_labels=False)
    trained = em_learn(network, data, num_iterations=100)
    plot_annotator_cpts(trained, "plots/annotators_em.png")
    plot_label_cpt(trained, "plots/labels_em.png")

# Uncomment to test EM learning
test_em_learn()
