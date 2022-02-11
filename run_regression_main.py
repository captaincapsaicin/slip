import argparse
import json

import experiment

def main(kwargs_json):
    kwargs = json.loads(kwargs_json)
    metrics = experiment.run_regression_experiment(**kwargs)
    print(metrics)  # TODO(nthomas) make this output to a log file somewhere... along with the parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take in a line of json')
    parser.add_argument('kwargs_json', type=str, help='a line of json')
    args = parser.parse_args()
    main(args.kwargs_json)

