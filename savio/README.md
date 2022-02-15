# Load the correct environment
```
module unload python
module load ml/tensorflow/2.5.0-py37
source activate slip
```

# Create gridsearch parameters
Specify the desired options in create_gridsearch.py

```
vi create_gridsearch.py
```

# Create the gridsearch directory for job logging

```
$ python create_gridsearch.py
while read i ; do sbatch $DIR/run_experiment.sh "$i"; done < $DIR/regression_params.json
```
Creates a directory for the logfiles, and prints the sbatch command for running the gridsearch

# Submit jobs per parameter set
Iterate over the lines in that file to create one job per task

```
while read i ; do sbatch $DIR/run_experiment.sh "$i"; done < $DIR/regression_params.json
```

# Look up the results somehow.
```
$DIR/*.out
```
