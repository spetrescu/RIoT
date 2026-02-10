# Running code for Use case 2 experiment
This is for reproducing the results of Table 4 in the paper.

## Install reqs
Create a venv using `python3 -m venv env`, subsequently activate it `source env/bin/activate`, and `pip install -r requirements.txt`

## Run `training.py` (takes roughly 2 minutes to obtain results -- as we parse 120 days of data from site F)
Assuming you have set up your venv, to run the code, simply execute `python training.py`

## Visualize `plot.py`
Assuming you have set up your venv, to see visualizations of proxy estimations for energy savings (both in terminal as table and figures), simply execute `python plot.py`