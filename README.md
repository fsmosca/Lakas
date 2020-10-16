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
The loss is calculated by running an engine match between `test engine` and `base engine`, where the `test engine` will use the param values recommended by the optimizer and the `base engine` will use the initial or default param values. If `test engine` wins by a result of say 0.52 from `(wins + draw/2) / games`, that result is minimized to `1.0 - 0.52` or 0.48. That is then reported to the optimizer. In this tuning example, the `base engine` will always use the init or default param. In the next budget the `test engine` will use the new recommended param values from the optimizer while the `base engine` will continue to use the init or default param.

In this optimization we use the oneplusone by using `--optimizer oneplusone ...`. Type `python lakas.py -h` to see more [options](https://github.com/fsmosca/Lakas/wiki/Help).

#### Command line
```
python lakas.py --optimizer oneplusone --depth 4 --base-time-sec 30 --budget 1000 --games-per-budget 1000 --concurrency 6 --engine ./engines/stockfish-modern/stockfish.exe --input-param "{'RazorMargin': {'init':527, 'lower':100, 'upper':700}, 'FutMargin': {'init':227, 'lower':50, 'upper':400}, 'KingAttackWeights[2]': {'init':81, 'lower':0, 'upper':200}, 'eMobilityBonus[0][7]': {'init':20, 'lower':-100, 'upper':100}, 'mMobilityBonus[1][7]': {'init':63, 'lower':-100, 'upper':150}, 'eMobilityBonus[3][21]': {'init':168, 'lower':-100, 'upper':300}, 'eThreatByRook[1]': {'init':46, 'lower':0, 'upper':200}, 'mThreatByMinor[3]': {'init':77, 'lower':0, 'upper':200}, 'eThreatByKing': {'init':89, 'lower':0, 'upper':200}, 'eThreatByPawnPush': {'init':39, 'lower':0, 'upper':200}}" --opening-file ./start_opening/ogpt_chess_startpos.epd
```

#### Input param
```
2020-10-10 15:34:08,341 | INFO  | input param: OrderedDict([('FutMargin', {'init': 227, 'lower': 50, 'upper': 400}), ('KingAttackWeights[2]', {'init': 81, 'lower': 0, 'upper': 200}), ('RazorMargin', {'init': 527, 'lower': 100, 'upper': 700}), ('eMobilityBonus[0][7]', {'init': 20, 'lower': -100, 'upper': 100}), ('eMobilityBonus[3][21]', {'init': 168, 'lower': -100, 'upper': 300}), ('eThreatByKing', {'init': 89, 'lower': 0, 'upper': 200}), ('eThreatByPawnPush', {'init': 39, 'lower': 0, 'upper': 200}), ('eThreatByRook[1]', {'init': 46, 'lower': 0, 'upper': 200}), ('mMobilityBonus[1][7]', {'init': 63, 'lower': -100, 'upper': 150}), ('mThreatByMinor[3]', {'init': 77, 'lower': 0, 'upper': 200})])
```

#### Basic optimization input info
```
2020-10-10 15:34:08,341 | INFO  | total budget: 1000
2020-10-10 15:34:08,342 | INFO  | games_per_budget: 1000
2020-10-10 15:34:08,342 | INFO  | tuning match move control: base_time_sec: 30, inc_time_sec: 0.05, depth=4
2020-10-10 15:34:08,348 | INFO  | parameter dimension: 10
2020-10-10 15:34:08,349 | INFO  | optimizer: oneplusone, noise_handling: optimistic, mutation: gaussian, crossover: False
```

#### Budget 1
```
2020-10-10 15:34:08,355 | INFO  | budget: 1
2020-10-10 15:34:08,356 | INFO  | recommended param: {'FutMargin': 227, 'KingAttackWeights[2]': 81, 'RazorMargin': 527, 'eMobilityBonus[0][7]': 20, 'eMobilityBonus[3][21]': 168, 'eThreatByKing': 89, 'eThreatByPawnPush': 39, 'eThreatByRook[1]': 46, 'mMobilityBonus[1][7]': 63, 'mThreatByMinor[3]': 77}
2020-10-10 15:34:08,356 | INFO  | best param: {'FutMargin': 227, 'KingAttackWeights[2]': 81, 'RazorMargin': 527, 'eMobilityBonus[0][7]': 20, 'eMobilityBonus[3][21]': 168, 'eThreatByKing': 89, 'eThreatByPawnPush': 39, 'eThreatByRook[1]': 46, 'mMobilityBonus[1][7]': 63, 'mThreatByMinor[3]': 77}
2020-10-10 15:34:08,356 | INFO  | init param: {'FutMargin': 227, 'KingAttackWeights[2]': 81, 'RazorMargin': 527, 'eMobilityBonus[0][7]': 20, 'eMobilityBonus[3][21]': 168, 'eThreatByKing': 89, 'eThreatByPawnPush': 39, 'eThreatByRook[1]': 46, 'mMobilityBonus[1][7]': 63, 'mThreatByMinor[3]': 77}
2020-10-10 15:34:08,356 | INFO  | recommended vs init
2020-10-10 15:36:47,417 | INFO  | actual result: 0.49800 @1000 games, minimized result: 0.50200, point of view: recommended
```

#### Budget 2
```
2020-10-10 15:36:47,434 | INFO  | budget: 2
2020-10-10 15:36:47,436 | INFO  | recommended param: {'FutMargin': 125, 'KingAttackWeights[2]': 56, 'RazorMargin': 509, 'eMobilityBonus[0][7]': -64, 'eMobilityBonus[3][21]': 118, 'eThreatByKing': 69, 'eThreatByPawnPush': 119, 'eThreatByRook[1]': 160, 'mMobilityBonus[1][7]': 74, 'mThreatByMinor[3]': 42}
2020-10-10 15:36:47,437 | INFO  | best param: {'FutMargin': 227, 'KingAttackWeights[2]': 81, 'RazorMargin': 527, 'eMobilityBonus[0][7]': 20, 'eMobilityBonus[3][21]': 168, 'eThreatByKing': 89, 'eThreatByPawnPush': 39, 'eThreatByRook[1]': 46, 'mMobilityBonus[1][7]': 63, 'mThreatByMinor[3]': 77}
2020-10-10 15:36:47,437 | INFO  | init param: {'FutMargin': 227, 'KingAttackWeights[2]': 81, 'RazorMargin': 527, 'eMobilityBonus[0][7]': 20, 'eMobilityBonus[3][21]': 168, 'eThreatByKing': 89, 'eThreatByPawnPush': 39, 'eThreatByRook[1]': 46, 'mMobilityBonus[1][7]': 63, 'mThreatByMinor[3]': 77}
2020-10-10 15:36:47,438 | INFO  | recommended vs init
2020-10-10 15:39:36,659 | INFO  | actual result: 0.49000 @1000 games, minimized result: 0.51000, point of view: recommended

...
```


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
