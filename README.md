# Lakas
Game parameter optimizer using nevergrad framework.


## A. Setup
### Specific installation example
A guide to install on [windows 10](https://github.com/fsmosca/Lakas/wiki/Windows-10-setup).

### General installation guide
* Install python 3.8 or later
* Install nevergrad
  * pip install nevergrad
* Install hiplot for plotting
  * pip install hiplot
  

## B. Supported NeverGrad Optimizers
* [OnePlusOne](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.ParametrizedOnePlusOne)
* [TBPSA](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.ParametrizedTBPSA) (Test-based population-size adaptation)
* [Bayessian Optimization](https://facebookresearch.github.io/nevergrad/optimizers_ref.html?highlight=logger#nevergrad.optimization.optimizerlib.ParametrizedBO)


## C. Sample optimization
* [Optimization comparison](https://github.com/fsmosca/Lakas/wiki/Optimization-Comparison)


## D. Resuming a cancelled optimization
Use the option  
`--output-data-file oneplusone.dat ...`  
to save the optimization data into the file `oneplusone.dat`. You may resume the optimization by using the option  
`--input-data-file oneplusone.dat ...`

#### Note
If you use oneplusone optimizer, you should use it to resume the oneplusone optimization.  
Example:  
`python lakas.py --output-data-file oneplusone.dat --optimizer oneplusone ...`  

After 2 budgets or so, optimization is cancelled. To resume:  
`python lakas.py --input-data-file oneplusone.dat --output-data-file oneplusone.dat --optimizer oneplusone ...`  

The 2 budgets stored in `oneplusone.dat` are still there and new budgets will be added on that same data file.

If your optimizer is tbpsa save it to a different file.  
`python lakas.py --output-data-file tbpsa.dat --optimizer tbpsa ...`  

## E. Credits
* [NeverGrad](https://github.com/facebookresearch/nevergrad)
* [Cutechess](https://github.com/cutechess/cutechess)
* [Stockfish](https://stockfishchess.org/)
* [HiPlot](https://github.com/facebookresearch/hiplot)
