# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Landscape tuning."""

import itertools
from pprint import pprint
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
from scipy.special import comb

import potts_model
import sampling
import experiment
import utils


def get_fraction_adaptive_singles(landscape: potts_model.PottsModel) -> float:
    """Returns the fraction of singles that are adaptive."""
    all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence, landscape.vocab_size)
    fraction_adaptive = (landscape.evaluate(all_singles) >= 0).sum() / all_singles.shape[0]
    return fraction_adaptive


def get_doubles_df(landscape, threshold, adaptive) -> pd.DataFrame:
    """Returns a dataframe with ['fitness', 'a_fitness', 'b_fitness', 'residual'] keys
    That includes all combinations of singles that are above/below a given fitness threshold
    """
    wt_seq = landscape.wildtype_sequence
    all_singles = sampling.get_all_single_mutants(wt_seq, landscape.vocab_size)
    df = experiment.get_fitness_df(all_singles, landscape.evaluate, wt_seq)
    if adaptive:
        singles = df[df.fitness >= threshold].sequence
    else:
        singles = df[df.fitness < threshold].sequence

    doubles = []
    single_a = []
    single_b = []
    for a, b in itertools.combinations(singles, 2):
        # check that this is a true double
        if utils.get_mutation_positions(a, wt_seq) == utils.get_mutation_positions(b, wt_seq):
            continue
        double = utils.add_seqs(a, b, wt_seq)[0]
        doubles.append(double)
        single_a.append(a)
        single_b.append(b)

    doubles_df = experiment.get_fitness_df(doubles, landscape.evaluate, wt_seq)

    single_a_fitness = landscape.evaluate(np.vstack(single_a))
    single_b_fitness = landscape.evaluate(np.vstack(single_b))
    doubles_df['a_fitness'] = single_a_fitness
    doubles_df['b_fitness'] = single_b_fitness

    residual = doubles_df.fitness - doubles_df.a_fitness - doubles_df.b_fitness + landscape.evaluate(wt_seq)
    doubles_df['residual'] = residual
    return doubles_df


def get_epistasis_stats(landscape: potts_model.PottsModel,
                        threshold: float = 0, adaptive: bool = True) -> Tuple[float, float]:
    """Returns statistics about epistasis for combinations of singles.

    Mean epistasis effect size is defined as the average effect of epistasis when two single mutants
    from the selected set are combined. The rate of reciprocal sign epistasis is defined as the fraction
    of epistatic interactions that have effects opposing the effects of the constituent singles. (i.e. the rate
    of negative epistasis for adaptive singles in combination).

    Args:
      landscape: The landscape.
      threshold: The threshold fitness for singles.
      adaptive: If True, thresholded singles will have fitness >= `threshold`. If False,
        selected singles will have fitness < `threshold`.

    Returns:
      A Tuple of length 2 where element [0] is the mean epistasis effect size
      and element [1] is the rate of reciprocal sign epistasis.
    """
    doubles_df = get_doubles_df(landscape, threshold, adaptive)
    residual = doubles_df.residual

    mean_epistasis_effect = np.mean(residual)
    if adaptive:
        rate_reciprocal_epistasis = (residual < 0).sum() / residual.shape[0]
    else:
        rate_reciprocal_epistasis = (residual > 0).sum() / residual.shape[0]
    return mean_epistasis_effect, rate_reciprocal_epistasis


def get_mean_single_effect(landscape: potts_model.PottsModel, threshold: float = 0, adaptive: bool = True) -> float:
    """Returns average effect size of singles.

    Args:
      landscape: The landscape.
      threshold: The threshold fitness for singles.
      adaptive: If True, thresholded singles will have fitness >= `threshold`. If False,
        selected singles will have fitness < `threshold`.

    Returns:
      The average effect size of the selected singles.
    """
    all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence, landscape.vocab_size)
    df = experiment.get_fitness_df(all_singles, landscape.evaluate, landscape.wildtype_sequence)
    if adaptive:
        mean_single_effect = df[df.fitness >= threshold].fitness.mean()
    else:
        mean_single_effect = df[df.fitness < threshold].fitness.mean()
    return mean_single_effect


def get_epistatic_horizon(landscape: potts_model.PottsModel) -> float:
    """Returns the epistatic horizon for the given landscape.

    The "epistatic horizon" is defined as the distance K from the wildtype at which, on average,
    epistatic contributions outweigh linear contributions from adaptive singles. This is the average
    distance we can expect a greedy algorithm to perform well on the given landscape.

    Let $s_{+}$ be the average adaptive single mutant effect. let $e_{+, +}$ be the average
    epistatic effect for a pair of adaptive singles. Then K is defined as :

    $$
    K = \\dfrac{e_{+, +} - 2 s_{+}}
               {e_{+, +}}
    $$

    Returns:
      The epistatic horizon.
    """
    mean_adaptive_epistasis, _ = get_epistasis_stats(landscape, threshold=0.0, adaptive=True)
    mean_adaptive_single = get_mean_single_effect(landscape, threshold=0, adaptive=True)
    epistatic_horizon = (mean_adaptive_epistasis - 2 * mean_adaptive_single) / mean_adaptive_epistasis
    return epistatic_horizon


def get_single_std(landscape: potts_model.PottsModel) -> float:
    """Returns the standard deviation of single mutant effects."""
    all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence, landscape.vocab_size)
    return np.std(landscape.evaluate(all_singles))


def get_landscape_stats(landscape: potts_model.PottsModel) -> dict:
    """Returns a dictionary of landscape statistics."""
    reciprocal_adaptive_epistasis_effect, fraction_reciprocal_adaptive_epistasis = get_epistasis_stats(
        landscape, threshold=0.0, adaptive=True)
    stats_dict = {'fraction_adaptive_singles': get_fraction_adaptive_singles(landscape),
                  'reciprocal_adaptive_epistasis_effect': reciprocal_adaptive_epistasis_effect,
                  'fraction_reciprocal_adaptive_epistasis': fraction_reciprocal_adaptive_epistasis,
                  'epistatic_horizon': get_epistatic_horizon(landscape),
                  'std_singles': get_single_std(landscape)}
    return stats_dict


def get_normalizing_field_scale(landscape: potts_model.PottsModel) -> float:
    std_singles = get_single_std(landscape)
    return 1 / std_singles


def get_single_mut_offset(landscape: potts_model.PottsModel, fraction_adaptive_singles: float) -> float:
    all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence, landscape.vocab_size)
    single_fitness = landscape.evaluate(all_singles)
    single_mut_offset = -1 * np.quantile(single_fitness, q=1 - fraction_adaptive_singles)
    return single_mut_offset


def get_epi_offset(landscape: potts_model.PottsModel, fraction_reciprocal_adaptive_epistasis: float) -> float:
    doubles_df = get_doubles_df(landscape, threshold=0.0, adaptive=True)
    # we want fraction to remain negative
    epi_offset = -1 * np.quantile(doubles_df.fitness, q=fraction_reciprocal_adaptive_epistasis)
    return epi_offset


def get_coupling_scale(landscape: potts_model.PottsModel,
                       epistatic_horizon: float,
                       field_scale: float = 1.0,
                       single_mut_offset: float = 0.0,
                       epi_offset: float = 0.0) -> float:
    """Returns the scaling factor that would result in `epistatic_horizon`

    Requires solving the equation for coupling_scale:

    K * field_scale (s_+ + field_scale) + (K choose 2) * coupling_scale (e_+ + epi_offset) = 0"""
    adaptive_threshold = 0.0
    is_adaptive = True

    mean_adaptive_single_effect = get_mean_single_effect(
        landscape, threshold=adaptive_threshold, adaptive=is_adaptive) + single_mut_offset
    untuned_epistasis_effect = get_doubles_df(
        landscape, threshold=adaptive_threshold, adaptive=is_adaptive).residual.mean()
    mean_epistasis_effect = untuned_epistasis_effect + epi_offset

    K = epistatic_horizon
    numerator = - K * field_scale * mean_adaptive_single_effect
    denominator = comb(K, 2) * mean_epistasis_effect
    coupling_scale = numerator / denominator
    return coupling_scale


# TODO(nthomas) fraction_reciprocal_deleterious_epistasis:
#   The fraction of deleterious(-, -) doubles that exhibit positive epistasis.
def get_tuning_kwargs(landscape: potts_model.PottsModel,
                      fraction_adaptive_singles: Optional[float] = None,
                      fraction_reciprocal_adaptive_epistasis: Optional[float] = None,
                      epistatic_horizon: Optional[float] = None,
                      normalize_to_singles: bool = True) -> Dict[str, float]:
    """Returns the landscape tuning parameters.

    Args:
      landscape: A landscape.
      fraction_adaptive_singles: The fraction of singles that achieve a fitness > wildtype. If unset,
        this fraction is preserved in the initial landscape
      fraction_reciprocal_adaptive_epistasis: The fraction of adaptive (+, +) doubles that exhibit negative
        epistasis.

      epistatic_horizon: average distance at which epi effects out weigh singles
      normalize_to_singles: A boolean, when True, ensures that the standard deviation of single mutant effects
        is 1.0.


    Returns:
      A dict of tuning parameters
      A tuple of [shift, shift, scale, scale] tuning parameters.
    """
    # default tuning params
    coupling_scale = 1.0
    field_scale = 1.0
    single_mut_offset = 0.0
    epi_offset = 0.0

    print('Untuned landscape stats:')
    pprint(get_landscape_stats(landscape))

    # compute tuning parameters
    if normalize_to_singles:
        field_scale = get_normalizing_field_scale(landscape)

    if fraction_adaptive_singles:
        single_mut_offset = get_single_mut_offset(landscape, fraction_adaptive_singles)

    if fraction_reciprocal_adaptive_epistasis:
        epi_offset = get_epi_offset(landscape, fraction_reciprocal_adaptive_epistasis)

    if epistatic_horizon:
        coupling_scale = get_coupling_scale(landscape, epistatic_horizon, field_scale, single_mut_offset, epi_offset)

    tuning_kwargs = {'coupling_scale': coupling_scale,
                     'field_scale': field_scale,
                     'single_mut_offset': single_mut_offset,
                     'epi_offset': epi_offset}
    return tuning_kwargs
