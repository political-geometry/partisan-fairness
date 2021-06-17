# Measuring partisan fairness

## The basics

This code accompanies the chapter "Measuring partisan fairness" by Mira Bernstein and Olivia Walch in _Political Geometry_. That chapter examines some of the metrics that have been proposed for quantifying the fairness of a districting plan. 

Most of the code is in Jupyter Notebooks (`.ipynb` ), with in-notebook explanations of what's being calculated and shown. For code that's more utilitarian/plot-centric, look to `utilities.py` and `noninteractive_utilities.py` (with the latter reproducing many functions introduced in the `.ipynb` notebooks for easier importing into other files).

## Getting started

The simplest way to interact with the notebooks is to [download and install Jupyter](https://jupyter.readthedocs.io/en/latest/install.html).

Once you've got a notebook running, you can change values and run cells individually or all at once. Standard warning of notebooks applies: Changing values in individual cells can affect the output of other cells. Keep track of the state of your environment, or reboot the kernel to make sure you know what's going into each of the plots.

## Table of Contents

Each notebook is aimed at illustrating a key concept relevant to measuring partisan fairness.

* `plotting_seats_and_votes.ipynb`: Introduction to proportionality, vote shares, and the seats-votes plane
* `massachusetts_and_new_york.ipynb`: Looks at how geography matters in two states with very similar Republican vote shares but very different seat shares
* `partisan_symmetry.ipynb`: Playing with the concept of partisan symmetry and **uniform partisan swing**
* `efficiency_gap.ipynb`: Showcases the definition, as well as some of the issues with, the **efficiency gap**
* `precinct_histograms.ipynb`: Examines how voter preferences vary at the precinct level across different states
* `ensemble_partisan_metrics.ipynb`: Plots histograms of partisan metrics taken from ensembles
* `make_figures_for_chapter.ipynb`:  Resource to reproduce plots and tables (in LaTeX form) from the chapter


## The data

Some of the data needed to run the notebooks is in the `data/` subdirectory, but we've held off on duplicating data that's already stored in other Github repositories. You'll want to download that elsewhere and copy it into your `data` directory. Links to the relevant data sources are below: 

* [Congressional Election results from The Daily Kos](https://docs.google.com/spreadsheets/d/1whYBonfwlgTGnYl7U_IH31G0JNYQ9QBIjDfqkZHkW-0/edit#gid=0)
* [The MIT Election Data & Science Lab](https://electionlab.mit.edu)
* [Shapefiles from MGGG States](https://github.com/mggg-states)
* [Ensembles (from the _Political Geometry_ Introduction)](https://github.com/political-geometry/chapter-0-introduction)
