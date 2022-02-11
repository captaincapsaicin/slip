import itertools
import json

defaults = {
    'mogwai_filepath': "/global/home/users/nthomas/git/slip/data/3er7_1_A_model_state_dict.npz",
    'potts_field_scale': 1,
    'potts_single_mut_offset': 0,
    'potts_epi_offset': 0,
    'vocab_size': 20,
    'training_set_min_num_mutations': 0,
    'model_random_seed': 0,
    'metrics_random_split_fraction': 0.8,
    'metrics_random_split_random_seed': 0,
    'metrics_distance_split_radii': [3, 4, 5],
    'training_set_max_num_mutations': 15,
    'training_set_num_samples': 5000,
}

options = {
    'training_set_random_seed': list(range(20)),
    'potts_coupling_scale': [1.0, 3.3, 6.0, 29.5],  # k = inf, 10, 6, 2
    'training_set_include_singles': [True, False],
    'model_name': ['linear', 'cnn']
}


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def main():
    json_lines = [json.dumps(dict(l, **defaults))
                  for l in product_dict(**options)]
    with open('regression_params.json', 'w') as f:
        f.write('\n'.join(json_lines))



if __name__ == '__main__':
    main()
