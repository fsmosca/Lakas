:: File: bayesopt_ucb.bat
:: Version 1.0

:: Title: Example 1, using bayesopt optimizer with ucb utility kind.


:: Usage:
:: If you want to use this batch file, copy this file
:: in the same folder with lakas.py

:: Type python lakas.py -has
:: To see more options and its usage.


:: Command line option descriptions

:: lakas.py
:: The program that will run the optimization.

:: --concurrency 1
:: Concurrecy is set to 1, but if your cpu has more threads
:: you can increase this to finish the optimization faster. During optimization
:: game matches will be played to get the loss. If --games-per-budget is high
:: it would take time to finish a single budget. If concurrency is 2, there
:: will be 2 games that will be run in parallel or concurrently.

:: --optimizer bayesopt
:: The optimizer to use from nevergrad optimizer libary.
:: Ref. 1: https://facebookresearch.github.io/nevergrad/optimizers_ref.html?highlight=bo#nevergrad.optimization.optimizerlib.ParametrizedBO
:: Ref. 2: https://github.com/fmfn/BayesianOptimization

:: --bo-utility-kind ucb
:: An option for bayesopt optimizer.

:: --output-data-file bayesopt_ucb.dat
:: This optimization will be saved in bayesopt_ucb.dat file. If for some
:: reason you cancel the optimization or there is power interruption,
:: you can use this file to continue the optimization.
:: Usage:
:: <old command line> --input-data-file bayesopt_ucb.dat

:: --optimizer-log-file opt_log_plot_ucb.txt
:: The optimization log for plotting will be saved in this file.
:: After the optimization you can browse on the iteractive plot save in 
:: opt_log_plot_ucb.txt.html.

:: --depth 6
:: The game matches to produce the loss will be played at depth 6 limit.

:: --base-time-sec 30
:: The time control used in the optimization match. Since the depth is 6,
:: this time control will not be followed as most probably depth 6
:: is triggered first.

:: --inc-time-sec 0.05
:: The increment per move, part of time control.

:: --budget 100
:: The number of times the loss or game matches are called.
:: Example 1: one parameter to be optimized: 
:: 'PawnValue': {'init':100, 'lower':50, 'upper':150}
:: param_space = upper - lower = 150 - 50 = 100
:: A param_pace equal to budget will be ideal.
:: Example 2: two parameters to be optimized: 
:: 'PawnValue': {'init':100, 'lower':50, 'upper':150}
:: 'KnightValue': {'init':300, 'lower':250, 'upper':350}
:: param_space_p = upper - lower = 150 - 50 = 100
:: param_space_n = upper - lower = 350 - 250 = 100
:: Total param_pace = param_space_p x param_space_n = 100x100 = 10k
:: Ideally budget should be 10k. But optimizers are smart, it can suggest
:: good parameters and reject bad parameters after couple of budgets.
:: So we don't need a budget of 10k. Notice the range of upper less lower,
:: if this is low, the param space that the optimizer will consider
:: will be greatly reduced. Initially we may assume that param_space is 10.
:: This depends on the range. If there are 2 params, 
:: total param_space=10x10=100. So our budget is 100. If the plot of the
:: optimizer is showing good result on higher budgets, that would mean that
:: it can still improve with higher budgets. In this case just run again
:: the optimization using:
:: <old command line> --input-data-file bayesopt_ucb.dat
:: so that the data in bayesopt_ucb.dat from previous optimization will be
:: used by the optimizer.

:: --games-per-budget 200
:: The number of games to be conducted in every match to get the loss.
:: To increase accuracy and reduce the result of a match noise,
:: increase this value.

:: --engine ./engines/stockfish-modern/stockfish.exe
:: The location and filename of the engine used in engine match.

:: --input-param "{'RazorMargin': {'init':527, 'lower':100, 'upper':700}, 'KingAttackWeights[2]': {'init':81, 'lower':0, 'upper':150}, 'eMobilityBonus[0][7]': {'init':20, 'lower':-20, 'upper':50}, 'mMobilityBonus[1][7]': {'init':63, 'lower':-50, 'upper':150}, 'eMobilityBonus[3][21]': {'init':168, 'lower':50, 'upper':250}, 'eThreatByRook[1]': {'init':46, 'lower':0, 'upper':150}, 'mThreatByMinor[3]': {'init':77, 'lower':0, 'upper':150}, 'eThreatByKing': {'init':89, 'lower':10, 'upper':150}, 'eThreatByPawnPush': {'init':39, 'lower':0, 'upper':100}}"
:: The parameters of the engine that will be optimized.

:: --opening-file ./start_opening/ogpt_chess_startpos.epd
:: The starting file to start the game matches.



python lakas.py --concurrency 1 --optimizer bayesopt --bo-utility-kind ucb --output-data-file bayesopt_ucb.dat --input-data-file bayesopt_ucb.dat --optimizer-log-file opt_log_plot_ucb.txt --depth 6 --base-time-sec 30 --inc-time-sec 0.05 --budget 100 --games-per-budget 200 --engine ./engines/stockfish-modern/stockfish.exe --input-param "{'RazorMargin': {'init':527, 'lower':100, 'upper':700}, 'KingAttackWeights[2]': {'init':81, 'lower':0, 'upper':150}, 'eMobilityBonus[0][7]': {'init':20, 'lower':-20, 'upper':50}, 'mMobilityBonus[1][7]': {'init':63, 'lower':-50, 'upper':150}, 'eMobilityBonus[3][21]': {'init':168, 'lower':50, 'upper':250}, 'eThreatByRook[1]': {'init':46, 'lower':0, 'upper':150}, 'mThreatByMinor[3]': {'init':77, 'lower':0, 'upper':150}, 'eThreatByKing': {'init':89, 'lower':10, 'upper':150}, 'eThreatByPawnPush': {'init':39, 'lower':0, 'upper':100}}" --opening-file ./start_opening/ogpt_chess_startpos.epd --opening-file-format epd

pause
