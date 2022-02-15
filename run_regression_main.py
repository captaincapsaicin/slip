import argparse
from pathlib import Path

import json

import experiment

def main(kwargs_json, output_dir, job_id):
    kwargs = json.loads(kwargs_json)
    metrics = experiment.run_regression_experiment(**kwargs)
    print(kwargs)
    print(metrics)

    output = {'metrics': metrics,
              'kwargs': kwargs}

    with open(Path(output_dir) / Path(job_id + '_metrics.json'), 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take in a line of json')
    parser.add_argument('kwargs_json', type=str, help='a line of json')
    parser.add_argument('output_dir', type=str, help='the output directory for metrics')
    parser.add_argument('job_id', type=str, help='the job id')
    args = parser.parse_args()
    main(args.kwargs_json, args.output_dir, args.job_id)

