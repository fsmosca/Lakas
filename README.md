# Lakas
Game parameter optimizer using nevergrad framework.


### Setup
* Install python 3.8 or later
* Install nevergrad
  * pip install nevergrad
  

### Supported NeverGrad Optimizers
* [OnePlusOne](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.ParametrizedOnePlusOne)
* [TBPSA](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.ParametrizedTBPSA) (Test-based population-size adaptation)
* [Bayessian Optimization](https://facebookresearch.github.io/nevergrad/optimizers_ref.html?highlight=logger#nevergrad.optimization.optimizerlib.ParametrizedBO)


### Sample optimization
#### Command line
```
python lakas.py --optimizer oneplusone --depth 4 --base-time-sec 30 --budget 1000 --games-per-budget 1000 --concurrency 6 --engine ./engines/stockfish-modern/stockfish.exe --input-param "{'RazorMargin': {'init':527, 'lower':100, 'upper':700}, 'FutMargin': {'init':227, 'lower':50, 'upper':400}, 'KingAttackWeights[2]': {'init':81, 'lower':0, 'upper':200}, 'eMobilityBonus[0][7]': {'init':20, 'lower':-100, 'upper':100}, 'mMobilityBonus[1][7]': {'init':63, 'lower':-100, 'upper':150}, 'eMobilityBonus[3][21]': {'init':168, 'lower':-100, 'upper':300}, 'eThreatByRook[1]': {'init':46, 'lower':0, 'upper':200}, 'mThreatByMinor[3]': {'init':77, 'lower':0, 'upper':200}, 'eThreatByKing': {'init':89, 'lower':0, 'upper':200}, 'eThreatByPawnPush': {'init':39, 'lower':0, 'upper':200}}" --opening-file ./start_opening/ogpt_chess_startpos.epd
```


### Credits
* [NeverGrad](https://github.com/facebookresearch/nevergrad)
* [Cutechess](https://github.com/cutechess/cutechess)
* [Stockfish](https://stockfishchess.org/)
