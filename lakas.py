#!/usr/bin/env python


"""Lakas

A game parameter optimizer using nevergrad framework"""


__author__ = 'fsmosca'
__script_name__ = 'Lakas'
__version__ = 'v0.23.0'
__credits__ = ['joergoster', 'musketeerchess', 'nevergrad', 'teytaud']


import os
import sys
import argparse
import ast
import copy
from collections import OrderedDict
from subprocess import Popen, PIPE
from pathlib import Path
import logging
import platform

import nevergrad as ng
import psutil


log_formatter = logging.Formatter("%(asctime)s | %(levelname)-5.5s | %(message)s")
log_formatter2 = logging.Formatter("%(asctime)s | %(process)6d | %(levelname)-5.5s | %(message)s")


def setup_logger(name, log_file, log_formatter, level=logging.INFO, console=False, mode='w'):
    handler = logging.FileHandler(log_file, mode=mode)
    handler.setFormatter(log_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    logger.propagate = False

    if console:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(log_formatter)
        logger.addHandler(consoleHandler)

    return logger


logger = setup_logger(
    'lakas_logger', 'lakas_log.txt', log_formatter,
    level=logging.INFO, console=True, mode='a')


logger2 = setup_logger(
    'match_logger', 'lakas_match.txt', log_formatter2,
    level=logging.INFO, console=False)


def find_process_id_by_name(process_name):
    process_object = []

    #Iterate over the all the running process
    for proc in psutil.process_iter():
       try:
           pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])

           # Check if process name contains the given name string.
           if process_name.lower() in pinfo['name'].lower() :
               process_object.append(pinfo)

       except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess):
           pass

    return process_object


def log_cpu(proc_list, msg=''):
    """
    proc_list = (proc, pid, name)
    """
    if len(proc_list) < 1:
        return

    os_name = platform.system()  # Linux, indows or ''
    num_threads = psutil.cpu_count(logical=True)

    for (p, pid, name) in proc_list:
        if os_name.lower() == 'windows':
            cpu_pct = p.cpu_percent(interval=None) / num_threads
        else:
            cpu_pct = p.cpu_percent(interval=None)
        logger2.info(f'{msg:43s}, proc_id: {pid}, cpu_usage%: {cpu_pct:0.0f}, num_threads: {num_threads}, proc_name: {name}')


class Objective:
    def __init__(self, optimizer, engine_file, input_param, init_param,
                 opening_file, opening_file_format, best_param, best_loss,
                 games_per_budget=100, depth=1000, concurrency=1,
                 base_time_sec=5, inc_time_sec=0.05,
                 match_manager='cutechess', variant='normal',
                 best_result_threshold=0.5, use_best_param=False, hashmb=64,
                 common_param=None, deterministic_function=False,
                 optimizer_name=None, spsa_scale=500000, proc_list=[]):
        self.optimizer = optimizer
        self.engine_file = engine_file
        self.input_param = input_param
        self.init_param = init_param
        self.games_per_budget = games_per_budget
        self.best_loss = best_loss
        self.depth = depth
        self.concurrency = concurrency
        self.base_time_sec = base_time_sec
        self.inc_time_sec = inc_time_sec
        self.opening_file = opening_file
        self.opening_file_format = opening_file_format
        self.match_manager = match_manager
        self.variant = variant
        self.best_result_threshold = best_result_threshold
        self.use_best_param = use_best_param
        self.hashmb = hashmb
        self.common_param = common_param
        self.deterministic_function = deterministic_function
        self.optimizer_name = optimizer_name
        self.spsa_scale = spsa_scale

        if len(best_param):
            self.best_param = copy.deepcopy(best_param)
        else:
            self.best_param = copy.deepcopy(init_param)

        if self.best_loss is None:
            self.best_loss = 1.0 - best_result_threshold

        self.test_param = {}

        self.proc_list = proc_list

    def run(self, **param):

        recommendation = self.optimizer.provide_recommendation()

        log_cpu(self.proc_list, msg=f'budget {self.optimizer.num_ask}, after asking recommendation')

        opt_best_param = recommendation.value
        opt_curr_best_value = self.optimizer.current_bests

        logger.info(f'budget: {self.optimizer.num_ask}')

        # Options for test engine.
        test_options = ''
        for k, v in param.items():
            test_options += f'option.{k}={v} '
            self.test_param.update({k: v})

        logger.info(f'recommended param: {self.test_param}')

        # Add common param. It should not be included in the test param.
        if self.common_param is not None:
            for k, v in self.common_param.items():
                test_options += f'option.{k}={v} '

        test_options = test_options.rstrip()

        # Options for base engine.
        base_options = ''
        if self.use_best_param:
            for k, v in self.best_param.items():
                base_options += f'option.{k}={v} '
        else:
            for k, v in self.init_param.items():
                base_options += f'option.{k}={v} '

        # Add common param.
        if self.common_param is not None:
            for k, v in self.common_param.items():
                base_options += f'option.{k}={v} '

        base_options = base_options.rstrip()

        logger.info(f'best param: {opt_best_param[1]}')

        # optimistic for non-deterministic and average for deterministic.
        if not self.deterministic_function:
            curr_best_loss = opt_curr_best_value["pessimistic"].mean
        else:
            curr_best_loss = opt_curr_best_value["average"].mean

        # Scale down the spsa loss for display.
        if self.optimizer_name == 'spsa':
            curr_best_loss = curr_best_loss/self.spsa_scale

        logger.info(f'best loss: {curr_best_loss}')

        logger.info(f'init param: {self.init_param}')

        if self.common_param is not None:
            logger.info(f'common param: {self.common_param}')

        if self.use_best_param:
            logger.info(f'recommended vs best')
        else:
            logger.info(f'recommended vs init')

        log_cpu(self.proc_list, msg='before a match starts')

        result = engine_match(self.engine_file, test_options, base_options,
                              self.opening_file, self.opening_file_format,
                              games=self.games_per_budget,
                              depth=self.depth, concurrency=self.concurrency,
                              base_time_sec=self.base_time_sec,
                              inc_time_sec=self.inc_time_sec,
                              match_manager=self.match_manager,
                              variant=self.variant, hashmb=self.hashmb)

        min_res = 1.0 - result

        log_cpu(self.proc_list, msg='after the match')

        logger.info(f'actual result: {result:0.5f} @{self.games_per_budget} games,'
                    f' minimized result: {min_res:0.5f},'
                    ' point of view: recommended\n')

        # Modify the loss that is reported to the optimizer as
        # the base engine will be using the current best param.
        if self.use_best_param:
            if min_res < 1.0 - self.best_result_threshold:
                self.best_loss = self.best_loss - (1.0 - min_res) * 0.001
                min_res = self.best_loss
                self.best_param = copy.deepcopy(self.test_param)
            else:
                min_res = self.best_result_threshold + min_res * 0.0001

        log_cpu(self.proc_list, msg='just before sending the result to optimizer')

        return min_res


def set_param(input_param):
    """Converts input param to a dict of param_name: init value"""
    new_param = {}
    for k, v in input_param.items():
        new_param.update({k: v['init']})

    return new_param


def read_result(line: str, match_manager) -> float:
    """
    Read result output line from match manager.
    cutechess:
      Score of e1 vs e2: 39 - 28 - 64  [0.542] 131
    duel:
      Score of e1 vs e2: [0.542] 131
    """
    if match_manager == 'cutechess':
        num_wins = int(line.split(': ')[1].split(' -')[0])
        num_draws = int(line.split(': ')[1].split('-')[2].strip().split()[0])
        num_games = int(line.split('] ')[1].strip())
        result = (num_wins + num_draws / 2) / num_games
    elif match_manager == 'duel':
        result = float(line.split('[')[1].split(']')[0])
    else:
        logger.exception(f'match manager {match_manager} is not supported.')
        raise

    return result


def get_match_commands(engine_file, test_options, base_options,
                       opening_file, opening_file_format, games, depth,
                       concurrency, base_time_sec, inc_time_sec, match_manager,
                       variant, hashmb):
    if match_manager == 'cutechess':
        tour_manager = Path(Path.cwd(), './tourney_manager/cutechess/cutechess-cli.exe')
    else:
        tour_manager = 'python -u ./tourney_manager/duel/duel.py'

    test_name = 'test'
    base_name = 'base'
    pgn_output = 'nevergrad_games.pgn'

    command = f' -concurrency {concurrency}'
    command += ' -tournament round-robin'

    if variant != 'normal':
        command += f' -variant {variant}'

    if match_manager == 'cutechess':
        command += f' -pgnout {pgn_output} fi'
        command += f' -each tc=0/0:{base_time_sec}+{inc_time_sec} depth={depth}'
        command += f' -engine cmd={engine_file} name={test_name} {test_options} proto=uci option.Hash={hashmb}'
        command += f' -engine cmd={engine_file} name={base_name} {base_options} proto=uci option.Hash={hashmb}'
        command += f' -rounds {games//2} -games 2 -repeat 2'
        command += ' -recover'
        command += ' -debug'
        command += f' -openings file={opening_file} order=random format={opening_file_format}'
        command += ' -resign movecount=6 score=700 twosided=true'
        command += ' -draw movenumber=30 movecount=6 score=1'
    else:
        command += f' -pgnout {pgn_output}'
        if depth != 1000:
            command += f' -each tc=0/0:{base_time_sec}+{inc_time_sec} depth={depth}'
        else:
            command += f' -each tc=0/0:{base_time_sec}+{inc_time_sec}'
        command += f' -engine cmd={engine_file} name={test_name} {test_options}'
        command += f' -engine cmd={engine_file} name={base_name} {base_options}'
        command += f' -rounds {games} -repeat 2'
        command += f' -openings file={opening_file}'

    return tour_manager, command


def engine_match(engine_file, test_options, base_options, opening_file,
                 opening_file_format, games=10, depth=1000, concurrency=1,
                 base_time_sec=5, inc_time_sec=0.05, match_manager='cutechess',
                 variant='normal', hashmb=64) -> float:
    result = ''

    tour_manager, command = get_match_commands(
        engine_file, test_options, base_options, opening_file,
        opening_file_format, games, depth, concurrency, base_time_sec,
        inc_time_sec, match_manager, variant, hashmb)

    # Execute the command line to start the match.
    process = Popen(str(tour_manager) + command, stdout=PIPE, text=True)
    for eline in iter(process.stdout.readline, ''):
        line = eline.strip()
        logger2.info(line)
        if line.startswith(f'Score of {"test"} vs {"base"}'):
            result = read_result(line, match_manager)
            if 'Finished match' in line:
                break

    if result == '':
        raise Exception('Error, there is something wrong with the engine match.')

    return result


def lakas_oneplusone(instrum, name, input_data_file,
                     noise_handling='optimistic',
                     mutation='gaussian', crossover=False, budget=100):
    """
    Ref.: https://facebookresearch.github.io/nevergrad/optimizers_ref.html?highlight=logger#nevergrad.families.ParametrizedOnePlusOne
    """
    # Continue from previous session by loading the previous data.
    if input_data_file is not None:
        loaded_optimizer = ng.optimizers.ParametrizedOnePlusOne()
        optimizer = loaded_optimizer.load(input_data_file)
        logger.info(f'optimizer: {name}, previous budget: {optimizer.num_ask}\n')
    else:
        # If input noise handling is a tuple, i.e "(optimistic, 0.01)".
        if '(' in noise_handling:
            noise_handling = ast.literal_eval(noise_handling)

        logger.info(f'optimizer: {name}, '
                    f'noise_handling: {noise_handling}, '
                    f'mutation: {mutation}, crossover: {crossover}\n')

        my_opt = ng.optimizers.ParametrizedOnePlusOne(
            noise_handling=noise_handling, mutation=mutation, crossover=crossover)

        optimizer = my_opt(parametrization=instrum, budget=budget)

    return optimizer


def lakas_tbpsa(instrum, name, input_data_file, naive=True,
                initial_popsize=None, budget=100):
    """
    Ref.: https://facebookresearch.github.io/nevergrad/optimizers_ref.html?highlight=logger#nevergrad.families.ParametrizedTBPSA
    """
    if input_data_file is not None:
        loaded_optimizer = ng.optimizers.ParametrizedTBPSA()
        optimizer = loaded_optimizer.load(input_data_file)
        logger.info(f'optimizer: {name}, previous budget: {optimizer.num_ask}\n')
    else:
        logger.info(f'optimizer: {name}, naive: {naive}, initial_popsize: {initial_popsize}\n')
        my_opt = ng.optimizers.ParametrizedTBPSA(naive=naive,
                                                 initial_popsize=initial_popsize)
        optimizer = my_opt(parametrization=instrum, budget=budget)

    return optimizer


def lakas_spsa(instrum, name, input_data_file, budget=100):
    """
    Ref.: https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.SPSA
    """
    if input_data_file is not None:
        loaded_optimizer = ng.optimizers.SPSA(instrum, budget=budget)
        optimizer = loaded_optimizer.load(input_data_file)
        logger.info(f'optimizer: {name}, previous budget: {optimizer.num_ask}\n')
    else:
        logger.info(f'optimizer: {name}\n')
        optimizer = ng.optimizers.SPSA(instrum, budget=budget)

    return optimizer


def lakas_cmaes(instrum, name, input_data_file, budget=100):
    """
    Ref.: https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.ParametrizedCMA
    """
    # Continue from previous session by loading the previous data.
    if input_data_file is not None:
        loaded_optimizer = ng.optimizers.ParametrizedCMA()
        optimizer = loaded_optimizer.load(input_data_file)
        logger.info(f'optimizer: {name}, previous budget: {optimizer.num_ask}\n')
    else:
        logger.info(f'optimizer: {name}\n')
        my_opt = ng.optimizers.ParametrizedCMA()
        optimizer = my_opt(parametrization=instrum, budget=budget)

    return optimizer


def lakas_bayessian_opt(instrum, name, input_data_file,
                        initialization='Hammersley',
                        init_budget=None, middle_point=False,
                        utility_kind='ucb', utility_kappa=2.576,
                        utility_xi=0.0, budget=100, gp_param_alpha=0.001):
    """
    Ref.: https://facebookresearch.github.io/nevergrad/optimizers_ref.html?highlight=logger#nevergrad.optimization.optimizerlib.ParametrizedBO
    """
    if input_data_file is not None:
        loaded_optimizer = ng.optimizers.ParametrizedBO()
        optimizer = loaded_optimizer.load(input_data_file)
        logger.info(f'optimizer: {name}, previous budget: {optimizer.num_ask}\n')
    else:
        gp_param = {'alpha': gp_param_alpha, 'normalize_y': True,
                    'n_restarts_optimizer': 5, 'random_state': None}

        logger.info(f'optimizer: {name},'
                    f' initialization: {initialization},'
                    f' init_budget: {init_budget},'
                    f' middle_point: {middle_point},'
                    f' utility_kind: {utility_kind},'
                    f' utility_kappa: {utility_kappa},'
                    f' utility_xi: {utility_xi},'
                    f' gp_parameters: {gp_param}\n')

        my_opt = ng.optimizers.ParametrizedBO(
            initialization=initialization, init_budget=init_budget,
            middle_point=middle_point,
            utility_kind=utility_kind, utility_kappa=utility_kappa,
            utility_xi=utility_xi,
            gp_parameters=gp_param)

        optimizer = my_opt(parametrization=instrum, budget=budget)

    return optimizer


def lakas_ngopt(instrum, name, input_data_file, budget=100):
    """
    References:
        https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.NGOpt
        https://arxiv.org/pdf/2004.14014.pdf
    """
    # Continue from previous session by loading the previous data.
    if input_data_file is not None:
        loaded_optimizer = ng.optimizers.NGOpt(instrum, budget=budget)
        optimizer = loaded_optimizer.load(input_data_file)
        logger.info(f'optimizer: {name}, previous budget: {optimizer.num_ask}\n')
    else:
        logger.info(f'optimizer: {name}\n')
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=budget)

    return optimizer


def main():
    main_pid = os.getpid()
    logger2.info('starting main()')

    process_name, proc_list = 'python', []
    process_objects = find_process_id_by_name(process_name)
    if len(process_objects) > 0:
        for elem in process_objects:
            processID = elem['pid']
            if processID != main_pid:
                continue
            proc = psutil.Process(processID)
            proc_list.append((proc, processID, process_name))
    else:
        logger2.warning('No Running Process found with given text')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description='Parameter optimizer using nevergrad library.',
        epilog='%(prog)s')
    parser.add_argument('--engine', required=True,
                        help='Engine filename or engine path and filename.')
    parser.add_argument('--hash', required=False, type=int,
                        help='Engine memory in MB, default=64.', default=64)
    parser.add_argument('--base-time-sec', required=False, type=int,
                        help='Base time in sec for time control, default=5.',
                        default=5)
    parser.add_argument('--inc-time-sec', required=False, type=float,
                        help='Increment time in sec for time control, default=0.05.',
                        default=0.05)
    parser.add_argument('--depth', required=False, type=int,
                        help='The maximum search depth that the engine is'
                             ' allowed, default=1000.\n'
                             'Example:\n'
                             '--depth 6 ...\n'
                             'If depth is high say 24 and you want this depth\n'
                             'to be always respected increase the base time'
                             ' control.\n'
                             'tuner.py --depth 24 --base-time-sec 300 ...',
                        default=1000)
    parser.add_argument('--optimizer', required=False, type=str,
                        help='Type of optimizer to use, can be oneplusone or'
                             ' tbpsa or bayesopt, or spsa, or cmaes, or ngopt, default=oneplusone.',
                        default='oneplusone')
    parser.add_argument('--oneplusone-noise-handling', required=False, type=str,
                        help='Parameter for oneplusone optimizer, can be optimistic or random,\n'
                             'or a tuple, default=optimistic.\n'
                             'Example:\n'
                             '--oneplusone-noise-handling random ...\n'
                             '--oneplusone-noise-handling optimistic ...\n'
                             '--oneplusone-noise-handling "(\'optimistic\', 0.01)" ...\n'
                             'where:\n'
                             '  0.01 is the coefficient (the regularity of reevaluations),\n'
                             '  default coefficient is 0.05.',
                        default='optimistic')
    parser.add_argument('--oneplusone-mutation', required=False, type=str,
                        help='Parameter for oneplusone optimizer, can be gaussian or cauchy,\n'
                             'or discrete or discreteBSO or fastga or doublefastga or\n'
                             'portfolio, default=gaussian.',
                        default='gaussian')
    parser.add_argument('--oneplusone-crossover', required=False, type=str,
                        help='Parameter for oneplusone optimizer. Whether to add a genetic crossover step\n'
                             'every other iteration, default=false.',
                        default='false')
    parser.add_argument('--tbpsa-naive', required=False, type=str,
                        help='Parameter for tbpsa optimizer, set to false for'
                             ' noisy problem, so that the best points\n'
                             'will be an average of the final population, default=false.\n'
                             'Example:\n'
                             '--optimizer tbpsa --tbpsa-naive true ...',
                        default='false')
    parser.add_argument('--tbpsa-initial-popsize', required=False, type=int,
                        help='Parameter for tbpsa optimizer. Initial population size, default=4xdimension.\n'
                             'Example:\n'
                             '--optimizer tbpsa --tbpsa-initial-popsize 8 ...',
                        default=None)
    parser.add_argument('--bo-utility-kind', required=False, type=str,
                        help='Parameter for bo optimizer. Type of utility'
                             ' function to use among ucb, ei and poi,'
                             ' default=ucb.\n'
                             'Example:\n'
                             '--optimizer bayesopt --bo-utility-kind ei ...',
                        default='ucb')
    parser.add_argument('--bo-utility-kappa', required=False, type=float,
                        help='Parameter for bayesopt optimizer. Kappa parameter for'
                             ' the utility function, default=2.576.\n'
                             'Example:\n'
                             '--optimizer bayesopt --bo-utility-kappa 2.0 ...',
                        default=2.576)
    parser.add_argument('--bo-utility-xi', required=False, type=float,
                        help='Parameter for bayesopt optimizer. Xi parameter for'
                             ' the utility function, default=0.0.\n'
                             'Example:\n'
                             '--optimizer bayesopt --bo-utility-xi 0.01 ...',
                        default=0.0)
    parser.add_argument('--bo-initialization', required=False, type=str,
                        help='Parameter for bayesopt optimizer. Can be Hammersley or random or LHS, default=Hammersley.\n'
                             'Example:\n'
                             '--optimizer bayesopt --bo-initialization random ...',
                        default='Hammersley')
    parser.add_argument('--bo-gp-param-alpha', required=False, type=float,
                        help='Parameter for bayesopt optimizer on gaussian process regressor, default=0.001.\n'
                             'Example:\n'
                             '--optimizer bayesopt --bo-gp-param-alpha 0.05 ...',
                        default=0.001)
    parser.add_argument('--spsa-scale', required=False, type=int,
                        help='Parameter for spsa optimizer to increase/decrease param perturbation, default=500000.\n'
                             'Example:\n'
                             '--optimizer spsa --spsa-scale 600000 ...',
                        default=500000)
    parser.add_argument('--budget', required=False, type=int,
                        help='Iterations to execute, default=1000.',
                        default=1000)
    parser.add_argument('--concurrency', required=False, type=int,
                        help='Number of game matches to run concurrently, default=1.',
                        default=1)
    parser.add_argument('--games-per-budget', required=False, type=int,
                        help='Number of games per iteration, default=100.\n'
                             'This should be even number.', default=100)
    parser.add_argument('--match-manager', required=False, type=str,
                        help='Match manager name, can be cutechess or duel, default=cutechess.',
                        default='cutechess')
    parser.add_argument('--opening-file', required=True, type=str,
                        help='Start opening filename in pgn or epd format.')
    parser.add_argument('--opening-file-format', required=True, type=str,
                        help='Opening format can be epd or pgn.')
    parser.add_argument('--variant', required=False, type=str,
                        help='Game variant, default=normal',
                        default='normal')
    parser.add_argument('--input-data-file', required=False, type=str,
                        help='Load the saved data to continue the optimization.')
    parser.add_argument('--output-data-file', required=False, type=str,
                        help='Save optimization data to this file.')
    parser.add_argument('--optimizer-log-file', required=False, type=str,
                        help='The filename of the log of certain optimization'
                             ' session. This file can be used to create a'
                             ' plot. Default=log_nevergrad.txt, Mode=append.',
                        default='log_nevergrad.txt')
    parser.add_argument('--input-param', required=True, type=str,
                        help='The parameters that will be optimized.\n'
                             'Example 1 with 1 parameter:\n'
                             '--input-param \"{\'pawn\': {\'init\': 92,'
                             ' \'lower\': 90, \'upper\': 120}}\"\n'
                             'Example 2 with 2 parameters:\n'
                             '--input-param \"{\'pawn\': {\'init\': 92,'
                             ' \'lower\': 90, \'upper\': 120}},'
                             ' \'knight\': {\'init\': 300, \'lower\': 250,'
                             ' \'upper\': 350}}\"'
                        )
    parser.add_argument('--common-param', required=False, type=str,
                        help='The parameters that will be sent to both test and base engines.\n'
                             'Make sure that this param is not included in the input-param.\n'
                             'Example:\n'
                             '--common-param \"{\'RookOpenFile\': 92, \'KnightOutpost\': 300}\"')
    parser.add_argument('--deterministic-function', action='store_true',
                        help='A flag to consider the objective function as deterministic.')
    parser.add_argument('--use-best-param', action='store_true',
                        help='Use best param for the base engine. A param'
                             ' becomes best if it defeats the\n'
                             'current best by --best-result-threshold value.')
    parser.add_argument('--best-result-threshold', required=False, type=float,
                        help='When match result is greater than this, update'
                             ' the best param, default=0.5.\n'
                             'Only applied when the flag --use-best-param is enabled,'
                             ' the best param will be used by the\n'
                             'base engine against the test engine that'
                             ' uses the param from the optimizer.',
                        default=0.5)

    args = parser.parse_args()

    optimizer_name = args.optimizer.lower()
    oneplusone_crossover = True if args.oneplusone_crossover.lower() == 'true' else False
    tbpsa_naive = True if args.tbpsa_naive.lower() == 'true' else False
    optimizer_log_file = args.optimizer_log_file
    input_data_file = args.input_data_file
    output_data_file = args.output_data_file  # Overwrite
    common_param = args.common_param
    use_best_param = args.use_best_param
    best_result_threshold = args.best_result_threshold
    deterministic_function = args.deterministic_function
    spsa_scale = args.spsa_scale

    # Check the filename of the intended output data.
    if (output_data_file is not None and
            output_data_file.lower().endswith(
                ('.py', '.pgn', '.fen', '.epd'))):
        logger.exception('Invalid output data filename.')
        raise NameError('Invalid output data filename.')

    if common_param is not None:
        common_param = ast.literal_eval(common_param)

    # Convert the input param string to a dict of dict and sort by key.
    input_param = ast.literal_eval(args.input_param)
    input_param = OrderedDict(sorted(input_param.items()))

    logger.info(f'input param: {input_param}\n')
    init_param = set_param(input_param)

    logger.info(f'total budget: {args.budget}')
    logger.info(f'games_per_budget: {args.games_per_budget}')
    logger.info(f'tuning match move control: base_time_sec: {args.base_time_sec}, '
                f'inc_time_sec: {args.inc_time_sec}, depth={args.depth}')

    # Prepare parameters to be optimized.
    arg = {}
    for k, v in input_param.items():
        arg.update({k: ng.p.Scalar(init=v['init'], lower=v['lower'],
                                   upper=v['upper']).set_integer_casting()})
    instrum = ng.p.Instrumentation(**arg)

    # deterministic_function in Nevergrad default since
    # nevergrad==0.4.3 is true. Lakas by default is false.
    if not deterministic_function:
        instrum.descriptors.deterministic_function = False

    logger.info(f'parameter dimension: {instrum.dimension}')
    logger.info(f'deterministic function: {deterministic_function}')
    logger.info(f'use best param: {use_best_param}')
    if use_best_param:
        logger.info(f'best result threshold: {best_result_threshold}')

    if input_data_file is not None:
        path = Path(input_data_file)
        if not path.is_file():
            input_data_file = None

    # Define optimizer.
    if optimizer_name == 'oneplusone':
        optimizer = lakas_oneplusone(
            instrum, optimizer_name, input_data_file,
            args.oneplusone_noise_handling, args.oneplusone_mutation,
            oneplusone_crossover, args.budget)
    elif optimizer_name == 'tbpsa':
        optimizer = lakas_tbpsa(
            instrum, optimizer_name, input_data_file, tbpsa_naive,
            args.tbpsa_initial_popsize, args.budget)
    elif optimizer_name == 'bayesopt':
        bo_init_budget, bo_middle_point = None, False
        optimizer = lakas_bayessian_opt(
            instrum, optimizer_name, input_data_file, args.bo_initialization,
            bo_init_budget, bo_middle_point, args.bo_utility_kind,
            args.bo_utility_kappa, args.bo_utility_xi, args.budget,
            args.bo_gp_param_alpha)
    elif optimizer_name == 'spsa':
        optimizer = lakas_spsa(instrum, optimizer_name, input_data_file, args.budget)
    elif optimizer_name == 'cmaes':
        optimizer = lakas_cmaes(instrum, optimizer_name, input_data_file, args.budget)
    elif optimizer_name == 'ngopt':
        optimizer = lakas_ngopt(instrum, optimizer_name, input_data_file, args.budget)
    else:
        logger.exception(f'optimizer {optimizer_name} is not supported.')
        raise

    # Save optimization log to file, append mode.
    nevergrad_logger = ng.callbacks.ParametersLogger(optimizer_log_file)
    optimizer.register_callback("tell", nevergrad_logger)

    # Get a recommended param from loaded optimization data
    # from previous optimization.
    # Todo: Create own file and remember the best param and loss
    # from previous optimization, so that we can continue from
    # those data. Applicable only if --use-best-param flag
    # is set to ON.
    best_param = {}
    best_loss = None

    if use_best_param:
        if optimizer.num_ask < 1:
            best_loss = best_result_threshold - (1.0 - best_result_threshold) * 0.001
            optimizer.tell(instrum, best_loss)
            recommendation = optimizer.provide_recommendation()
            recommendation_value = recommendation.value
            best_param = recommendation_value[1]
            curr_best_loss = optimizer.current_bests
            best_loss = curr_best_loss["average"].mean

            if output_data_file is not None:
                optimizer.dump(output_data_file)

        elif input_data_file is not None:
            recommendation = optimizer.provide_recommendation()
            recommendation_value = recommendation.value
            best_param = recommendation_value[1]
            curr_best_loss = optimizer.current_bests
            best_loss = curr_best_loss["average"].mean

    objective = Objective(optimizer, args.engine, input_param, init_param,
                          args.opening_file, args.opening_file_format,
                          best_param, best_loss,
                          games_per_budget=args.games_per_budget,
                          depth=args.depth, concurrency=args.concurrency,
                          base_time_sec=args.base_time_sec,
                          inc_time_sec=args.inc_time_sec,
                          match_manager=args.match_manager,
                          variant=args.variant,
                          best_result_threshold=best_result_threshold,
                          use_best_param=use_best_param,
                          hashmb=args.hash, common_param=common_param,
                          deterministic_function=deterministic_function,
                          optimizer_name=optimizer_name, spsa_scale=spsa_scale,
                          proc_list=proc_list)

    # Start the optimization.
    for _ in range(optimizer.budget):
        x = optimizer.ask()

        loss = objective.run(**x.kwargs)

        # Scale up the loss for spsa optimizer to make
        # param value increment higher than 1,  default spsa_scale=500000.
        if optimizer_name == 'spsa':
            loss = loss * spsa_scale

        optimizer.tell(x, loss)

        # Save optimization data to continue in the next session.
        # --output-data-file opt_data.dat ...
        if output_data_file is not None:
            optimizer.dump(output_data_file)

    # Optimization done, get the best param.
    recommendation = optimizer.provide_recommendation()
    best_param = recommendation.value
    logger.info(f'best_param: {best_param[1]}')

    # Plot optimization data with hiplot, save it to html file.
    # Install the hiplot lib with "pip install hiplot".
    try:
        exp = nevergrad_logger.to_hiplot_experiment()
    except ImportError as msg:
        logger.warning(msg)
    except Exception:
        logger.exception('Unexpected exception.')
    else:
        exp.to_html(f'{optimizer_log_file}.html')

    # Output for match manager.
    option_output = ''
    for k, v in best_param[1].items():
        option_output += f'option.{k}={v} '
    logger.info(f'{option_output}\n')


if __name__ == "__main__":
    main()
