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

"""Tests for synthetic_protein_landscapes.tuning."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import tuning
import potts_model
import sampling


class TuningParamsTest(parameterized.TestCase):

    def _get_params(self, seed):
        """Weight matrix and field vector."""
        rng = np.random.default_rng(seed)
        weight_matrix = rng.normal(size=(4, 4, 20, 20))
        # make symmetric
        weight_matrix = weight_matrix + np.moveaxis(weight_matrix, (0, 1, 2, 3), (1, 0, 3, 2))
        field_vec = rng.normal(size=(4, 20))
        return weight_matrix, field_vec

    def _get_landscape(self, seed, wt_seq=[0, 0, 0, 0], **kwargs):
        """Return a small PottsModel landscape."""
        weight_matrix, field_vec = self._get_params(seed)
        return potts_model.PottsModel(weight_matrix, field_vec, wt_seq=wt_seq, **kwargs)

    @parameterized.named_parameters(
        dict(
            testcase_name='wt_0',
            seed=0,
            wt_seq=[0, 0, 0, 0],
        ),
        dict(
            testcase_name='wt_1',
            seed=1,
            wt_seq=[1, 1, 1, 1],
        ),
    )
    def test_normalize_to_singles(self, wt_seq, seed):
        untuned_landscape = self._get_landscape(wt_seq=wt_seq, seed=seed)
        tuning_kwargs = tuning.get_tuning_kwargs(untuned_landscape, normalize_to_singles=True)
        tuned_landscape = self._get_landscape(wt_seq=wt_seq, seed=seed, **tuning_kwargs)

        all_single_fitness = tuned_landscape.evaluate(
            sampling.get_all_single_mutants(wt_seq, tuned_landscape.vocab_size))
        np.testing.assert_equal(np.std(all_single_fitness), 1.0)

    @parameterized.named_parameters(
        dict(
            testcase_name='tuning_2',
            seed=2,
            wt_seq=[0, 0, 0, 0],
            desired_stats_dict={'fraction_adaptive_singles': 0.1,
                                'fraction_reciprocal_adaptive_epistasis': 0.4,
                                'epistatic_horizon': 10}
        ),
        dict(
            testcase_name='tuning_1',
            seed=3,
            wt_seq=[1, 1, 1, 1],
            desired_stats_dict={'fraction_adaptive_singles': 0.15,
                                'fraction_reciprocal_adaptive_epistasis': 0.52,
                                'epistatic_horizon': 20},
        ),
    )
    def test_tuned_stats(self, wt_seq, seed, desired_stats_dict):
        untuned_landscape = self._get_landscape(wt_seq=wt_seq, seed=seed)

        tuning_kwargs = tuning.get_tuning_kwargs(
            untuned_landscape,
            normalize_to_singles=True,
            fraction_adaptive_singles=desired_stats_dict['fraction_adaptive_singles'],
            fraction_reciprocal_adaptive_epistasis=desired_stats_dict['fraction_reciprocal_adaptive_epistasis'],
            epistatic_horizon=desired_stats_dict['epistatic_horizon'])

        tuned_landscape = self._get_landscape(wt_seq=wt_seq, seed=seed, **tuning_kwargs)
        actual_stats_dict = tuning.get_landscape_stats(tuned_landscape)
        for stat in desired_stats_dict.keys():
            self.assertEqual(desired_stats_dict[stat], actual_stats_dict[stat])


if __name__ == '__main__':
    absltest.main()
