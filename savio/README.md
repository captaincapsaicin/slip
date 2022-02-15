# Create parameters
Specify the desired options in create_gridsearch.py

```
python create_gridsearch.py
```
Creates a file called regression_params.json

# Submit jobs per parameter set
Iterate over the lines in that file to create one job per task

```
while read i ; do sbatch run_experiment.sh $i; done < regression_params.json
```

# Look up the results somehow.
```
```
