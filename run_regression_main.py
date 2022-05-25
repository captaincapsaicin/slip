"""Entry point for running regression experiments."""
import argparse
from pathlib import Path

import json

import pandas as pd

import experiment


def main(kwargs_json, output_dir, job_id):
    """Main function for sbatch run.

    Args:
      kwargs_json: a json string with regression experiment configuration parameters
      output_dir: string indicating directory to put metrics
      job_id: a unique identifier for the job
    """
    kwargs = json.loads(kwargs_json)
    print(kwargs)
    metrics = experiment.run_regression_experiment(**kwargs)
    print(metrics)

    # update output with input kwargs
    for metric_type in metrics.keys():
        for kwarg, val in kwargs.items():
            metrics[metric_type][kwarg] = val

    # write metrics
    with open(Path(output_dir) / Path(job_id + '_metrics.json'), 'w') as f:
        pd.DataFrame(metrics).T.to_json(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take in a line of json')
    parser.add_argument('kwargs_json', type=str, help='a line of json')
    parser.add_argument('output_dir', type=str, help='the output directory for metrics')
    parser.add_argument('job_id', type=str, help='the job id')
    args = parser.parse_args()
    main(args.kwargs_json, args.output_dir, args.job_id)
