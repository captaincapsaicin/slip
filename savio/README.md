# Create parameters
Specify the desired options in create_gridsearch.py

```
python create_gridsearch.py
```
Creates a file called regression_params.json

# Submit jobs per parameter set
Iterate over the lines in that file to create one job per task

```
for line in `cat regression_params.json`; do sbatch run_experiment.sh "$line"; done
```

# Look up the results somehow.
