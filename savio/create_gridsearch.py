from datetime import datetime
import itertools

from pathlib import Path

import json

LOG_DIRECTORY = '/global/scratch/projects/fc_songlab/nthomas/slip/log/'

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


def update_with_defaults(existing_dict, defaults):
    return dict(existing_dict, **defaults)


def write_regression_params(options, defaults, outfile):
    """Writes a json of regression parameters.

    Outfile should be the eventual run directory."""
    json_lines = [json.dumps(update_with_defaults(d, defaults)) for d in product_dict(**options)]

    with open(outfile, 'w') as f:
        f.write('\n'.join(json_lines))


def get_batch_id():
    """Returns a batch id based on the time."""
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


SBATCH_TEMPLATE = 'run_experiment_template.txt'
def write_sbatch_script(batch_id, template_filepath, out_filepath):
    """Reads in a `template` and fills in the batch_id where necessary."""
    with open(template_filepath, 'r') as f:
        text = f.read()
    text = text.format(batch_id=batch_id)
    with open(out_filepath, 'w') as f:
        f.write(text)


def write_readable_options_and_defaults(options, defaults, directory):
    with open(directory / Path('defaults.json'), 'w') as f:
        json.dump(defaults, f)
    with open(directory / Path('options.json'), 'w') as f:
        json.dump(options, f)


command_string = "while read i ; do sbatch {job_directory}/run_experiment.sh \"$i\"; done < {job_directory}/regression_params.json"
def get_command_string(job_directory):
    return command_string.format(job_directory=job_directory)


def main():
    # create batch ID
    batch_id = get_batch_id()
    # create directory for the job
    job_directory = Path(LOG_DIRECTORY) / Path(batch_id)
    job_directory.mkdir()

    # write regression_params to the job directory
    outfile = job_directory / Path('regression_params.json')
    write_regression_params(options, defaults, outfile)

    # read in the experiment template and add the batch ID
    outfile = job_directory / Path('run_experiment.sh')
    write_sbatch_script(batch_id, SBATCH_TEMPLATE, outfile)

    # write the options into a text file for human readability
    write_readable_options_and_defaults(options, defaults, job_directory)

    print(get_command_string(job_directory))


if __name__ == '__main__':
    main()
