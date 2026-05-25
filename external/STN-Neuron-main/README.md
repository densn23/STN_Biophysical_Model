# STN-Neuron

This is the NEURON python model for the optimized STN multicompartment neuron. The MatingPool.pickle is the pool optimized by a custom genetic algorithm.

## Using the code

1. Install dependencies
Start terminal (Mac) or equivalents on other operating systems under the project folder.
run `pip install -r requirements.txt`

2. Compile the mod files
- go to folder `sth` using `cd sth` (Mac or Linux command, use equivalents on other operating systems).
- run 'nrnivmodl' or 'python compile.py' to compile the mod files.

3. Run simulation
- run `sim_neuron.py` using command `python sim_neuron.py` for optimized STN neuron model using GW morphology.
- run `sim_detail.py` using command `python sim_detail.py` for detailed morphology from Chu et al., 2017.

## Notes
`stn_neuron.py` is for model using GW morphology and has is supposed to work with `sim_neuron.py` for parameter value and distribution assignment.

`stn_detail.py` is for detailed morphologies from Chu et al., 2017. They are supposed to work with `sim_detail.py`. Morphology filename is assigned in function main() in `sim_detail.py`. You can change it to other morphologies.