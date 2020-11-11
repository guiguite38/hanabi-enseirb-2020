# hanabi-enseirb-2020


The aim is to rise to the challenge offered by Deepmind [[1]](#Bard_2020). \
One promising lead is offered by Facebook [[2]](#lerer2019improving).

## Environment installation

See the instructions in ```environment/README.md```.
If you run into the error:
  - ```ModuleNotFoundError: No module named 'skbuild'``` you need to install [scikit-build](https://scikit-build.readthedocs.io/en/latest/installation.html)
  - ```CMake error```, try running ```environment/clean_all.sh``` then try again

## Objectives

Our first main goal is to rise to the first challenge, which is when the game happens with the same agent playing with himself.

Then, we'll move on to the next challenge which is to play with different agents.

# Architecture

```
+ environment/        contains Deepmind's hanabi environment
+ players/            contains the players we made
```

# Authors

With equal contributions:
  - Guillaume Grosse
  - Matéo Mahaut
  - Théo Matricon
  - Quentin Lanneau

# Bibliography


- **[1]**<a id="Bard_2020"></a>
  *The Hanabi challenge: A new frontier for AI research* \
  Bard, Nolan and Foerster, Jakob N. and Chandar, Sarath and Burch, Neil and Lanctot, Marc and Song, H. Francis and Parisotto, Emilio and Dumoulin, Vincent and Moitra, Subhodeep and Hughes, Edward and et al. \
  from Artificial Intelligence vol. 20, March 2020, http://dx.doi.org/10.1016/j.artint.2019.103216 \
  DOI: 10.1016/j.artint.2019.103216


-  **[2]** <a id="lerer2019improving"></a>
      *Improving Policies via Search in Cooperative Partially Observable Games* \
      Adam Lerer and Hengyuan Hu and Jakob Foerster and Noam Brown \
      2019, https://arxiv.org/abs/1912.02318
