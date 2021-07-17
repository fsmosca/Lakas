#!/usr/bin/env python


"""Lakas

A game parameter optimizer using nevergrad framework"""


__author__ = 'fsmosca'
__script_name__ = 'Lakas'
__version__ = 'v0.42.0'
__credits__ = ['ChrisWhittington', 'Claes1981', 'joergoster', 'Matthies',
               'musketeerchess', 'teytaud', 'thehlopster',
               'tryingsomestuff']


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
import shlex

import nevergrad as ng
import psutil


os_name = platform.system()  # Linux, Windows or ''


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
    'lakas_logger', 'log_lakas.txt', log_formatter,
    level=logging.INFO, console=True, mode='a')


logger2 = setup_logger(
    'match_logger', 'lakas_match.txt', log_formatter2,
    level=logging.INFO, console=False)


def find_process_id_by_name(process_name):
    process_object = []

    # Iterate over all the running process.
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

    num_threads = psutil.cpu_count(logical=True)

    for (p, pid, name) in proc_list:
        mem_mbytes = p.memory_info()[0] / (1024 * 1024)  # rss, working set in windows resource monitor
        if os_name.lower() == 'windows':
            cpu_pct = p.cpu_percent(interval=None) / num_threads
        else:
            cpu_pct = p.cpu_percent(interval=None)
        logger2.info(f'{msg:43s},'
                     f' proc_id: {pid},'
                     f' cpu_usage%: {cpu_pct:0.0f},'
                     f' mem_mb: {mem_mbytes:0.0f},'
                     f' num_threads: {num_threads},'
                     f' proc_name: {name}')


class Objective:
    def __init__(self, optimizer, engine_file, input_param, init_param,
                 opening_file, opening_file_format, best_param, best_loss,
                 games_per_budget=100, depth=1000, concurrency=1,
                 base_time_sec=None, inc_time_sec=None,
                 move_time_ms=None, nodes=None,
                 match_manager='cutechess',
                 match_manager_path=None, variant='normal',
                 best_result_threshold=0.5, use_best_param=False,
                 common_param=None, deterministic_function=False,
                 optimizer_name=None, spsa_scale=500000, proc_list=[],
                 cutechess_debug=False, cutechess_wait=5000,
                 protocol='uci',
                 timemargin=50,
                 enhance=False,
                 enhance_hashmb=64,
                 enhance_threads=1,
                 enhance_limitvalue=15,
                 enhance_fenfile='default',
                 enhance_limittype='depth',
                 enhance_evaltype='mixed',
                 enhance_posperfile=50):
        self.optimizer = optimizer
        self.engine_file = engine_file
        self.input_param = input_param
        self.init_param = init_param
        self.games_per_budget = games_per_budget
        self.best_loss = best_loss
        self.concurrency = concurrency
        self.timemargin = timemargin

        self.depth = int(depth) if depth is not None else depth
        self.base_time_sec = int(base_time_sec) if base_time_sec is not None else base_time_sec
        self.inc_time_sec = float(inc_time_sec) if inc_time_sec is not None else inc_time_sec
        self.move_time_ms = move_time_ms
        self.nodes = nodes

        if self.move_time_ms is not None:
            self.move_time = int(self.move_time_ms)/1000  # cutechess uses st=N, N in sec
        else:
            self.move_time = self.move_time_ms

        # Raise error if there are no or unsupported move control.
        if not enhance:
            if self.move_time is None and self.nodes is None:
                if self.base_time_sec is None and self.depth is None:
                    raise Exception('Error, missing time and depth control!')
                elif self.base_time_sec is None and self.inc_time_sec is not None and self.depth is not None:
                    raise Exception('Error, not supported move control!')

        self.opening_file = opening_file
        self.opening_file_format = opening_file_format
        self.match_manager = match_manager
        self.match_manager_path = match_manager_path
        self.variant = variant
        self.best_result_threshold = best_result_threshold
        self.use_best_param = use_best_param
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
        self.cutechess_debug=cutechess_debug
        self.cutechess_wait=cutechess_wait
        self.protocol=protocol
        self.enhance = enhance
        self.enhance_hashmb = enhance_hashmb
        self.enhance_threads = enhance_threads
        self.enhance_limitvalue = enhance_limitvalue
        self.enhance_fenfile = enhance_fenfile
        self.enhance_limittype = enhance_limittype
        self.enhance_evaltype = enhance_evaltype
        self.enhance_posperfile = enhance_posperfile

    def bench(self, test_options):
        """
        Run the engine with bench command using enhance.py interface and
        return the total nodes searched.
        """
        total_nodes = None
        command = f' -engine cmd={self.engine_file} {test_options}'
        command += f' -hashmb {self.enhance_hashmb}'
        command += f' -threads {self.enhance_threads}'
        command += f' -limitvalue {self.enhance_limitvalue}'
        command += f' -fenfile {self.enhance_fenfile}'
        command += f' -limittype {self.enhance_limittype}'
        command += f' -evaltype {self.enhance_evaltype}'
        command += f' -posperfile {self.enhance_posperfile}'
        command += f' -concurrency {self.concurrency}'

        if os_name.lower() == 'windows':
            process = Popen(str(self.match_manager_path) + command, stdout=PIPE, text=True)
        else:
            process = Popen(shlex.split(str(self.match_manager_path) + command), stdout=PIPE, text=True)

        # Parse the bench output.
        for eline in iter(process.stdout.readline, ''):
            line = eline.strip()
            # total nodes searched from 4 workers: 894889
            if line.startswith('total nodes searched from '):
                total_nodes = int(line.split(': ')[1])
            elif 'bench done' in line:
                break

        if total_nodes is None:
            raise Exception('Error, there is something wrong with the bench command.')

        return total_nodes

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
        logger2.info(f'test engine options: {test_options}')

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
            logger2.info(f'base engine options: {base_options}')
        else:
            for k, v in self.init_param.items():
                base_options += f'option.{k}={v} '
            logger2.info(f'base engine options: {base_options}')

        # Add common param.
        if self.common_param is not None:
            for k, v in self.common_param.items():
                base_options += f'option.{k}={v} '

        base_options = base_options.rstrip()

        if self.optimizer_name != 'spsa' or self.optimizer.num_ask > 1:
            logger.info(f'best param: {opt_best_param[1]}')

        # Output for match manager.
        option_output = ''
        for k, v in opt_best_param[1].items():
            option_output += f'option.{k}={v} '
        logger.info(f'{option_output}')

        # optimistic for non-deterministic and average for deterministic.
        if not self.deterministic_function:
            curr_best_loss = opt_curr_best_value["pessimistic"].mean
        else:
            curr_best_loss = opt_curr_best_value["average"].mean

        # Scale down the spsa loss for display.
        if self.optimizer_name == 'spsa':
            curr_best_loss = curr_best_loss/self.spsa_scale

        if self.optimizer_name != 'spsa' or self.optimizer.num_ask > 1:
            logger.info(f'best loss: {curr_best_loss}')

        logger.info(f'init param: {self.init_param}')

        if self.common_param is not None:
            logger.info(f'common param: {self.common_param}')

        if not self.enhance:
            if self.use_best_param:
                logger.info(f'recommended vs best')
            else:
                logger.info(f'recommended vs init')

        log_cpu(self.proc_list, msg='before a match starts')

        if self.enhance:
            result = self.bench(test_options)
            logger.info(f'total nodes searched: {result}')
            return -result
        else:
            result = engine_match(self.engine_file, test_options, base_options,
                                  self.opening_file, self.opening_file_format,
                                  games=self.games_per_budget,
                                  depth=self.depth, concurrency=self.concurrency,
                                  base_time_sec=self.base_time_sec,
                                  inc_time_sec=self.inc_time_sec,
                                  match_manager=self.match_manager,
                                  match_manager_path=self.match_manager_path,
                                  variant=self.variant,
                                  cutechess_debug=self.cutechess_debug,
                                  cutechess_wait=self.cutechess_wait,
                                  move_time=self.move_time, nodes=self.nodes,
                                  protocol=self.protocol,
                                  timemargin=self.timemargin)

            min_res = 1.0 - result

            log_cpu(self.proc_list, msg='after the match')

            logger.info(f'actual result: {result:0.5f} @{self.games_per_budget} games,'
                        f' minimized result or loss: {min_res:0.5f},'
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
        if type(v) == list:
            new_param.update({k: v[0]})  # First value is default.
        else:
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
                       match_manager_path,
                       variant, cutechess_debug, cutechess_wait,
                       move_time, nodes, protocol, timemargin):
    if match_manager == 'cutechess':
        tour_manager = Path(match_manager_path)
    else:
        # match_manager_path = 'python c:/chess/tourney_manager/duel/duel.py'
        tour_manager = match_manager_path

    test_name = 'test'
    base_name = 'base'
    pgn_output = 'nevergrad_games.pgn'

    command = f' -concurrency {concurrency}'
    command += ' -tournament round-robin'

    if variant != 'normal':
        command += f' -variant {variant}'

    if match_manager == 'cutechess':
        command += f' -pgnout {pgn_output} fi'

        # Set the move control.
        if move_time is not None:
            command += f' -each st={move_time}'
        elif nodes is not None:
            command += f' -each tc=inf nodes={nodes}'
        else:
            if base_time_sec is not None and inc_time_sec is not None and depth is not None:
                command += f' -each tc=0/0:{base_time_sec}+{inc_time_sec} depth={depth}'
            elif base_time_sec is not None and inc_time_sec is not None:
                command += f' -each tc=0/0:{base_time_sec}+{inc_time_sec}'
            elif base_time_sec is not None:
                command += f' -each tc=0/0:{base_time_sec}'
            elif inc_time_sec is not None and depth is not None:
                command += f' -each tc=0/0:{0}+{inc_time_sec} depth={depth}'
            elif inc_time_sec is not None:
                command += f' -each tc=0/0:{0}+{inc_time_sec}'
            elif depth is not None:
                command += f' -each tc=inf depth={depth}'

        command += f' -engine cmd={engine_file} name={test_name} timemargin={timemargin} proto={protocol} {test_options}'
        command += f' -engine cmd={engine_file} name={base_name} timemargin={timemargin} proto={protocol} {base_options}'
        command += f' -rounds {games//2} -games 2 -repeat 2'
        command += ' -recover'
        command += f' -wait {cutechess_wait}'
        command += f' -openings file={opening_file} order=random format={opening_file_format}'
        command += ' -resign movecount=6 score=700 twosided=true'
        command += ' -draw movenumber=30 movecount=6 score=1'

        if cutechess_debug:
            command += ' -debug'
    # duel.py match manager
    else:
        command += f' -pgnout {pgn_output}'
        if depth is not None:
            command += f' -each tc=inf depth={depth}'
        else:
            command += f' -each tc=0/0:{base_time_sec}+{inc_time_sec}'
        command += f' -engine cmd={engine_file} name={test_name} {test_options}'
        command += f' -engine cmd={engine_file} name={base_name} {base_options}'
        command += f' -rounds {games} -repeat 2'
        command += f' -openings file={opening_file}'
        command += f' -draw movenumber=40 movecount=10 score=0'
        command += f' -resign movecount=6 score=900'

    return tour_manager, command


def engine_match(engine_file, test_options, base_options, opening_file,
                 opening_file_format, games=10, depth=None, concurrency=1,
                 base_time_sec=None, inc_time_sec=None,
                 match_manager='cutechess', match_manager_path=None,
                 variant='normal', cutechess_debug=False,
                 cutechess_wait=5000, move_time=None, nodes=None,
                 protocol='uci',
                 timemargin=50) -> float:
    result = ''

    tour_manager, command = get_match_commands(
        engine_file, test_options, base_options, opening_file,
        opening_file_format, games, depth, concurrency, base_time_sec,
        inc_time_sec, match_manager, match_manager_path, variant, cutechess_debug,
        cutechess_wait, move_time, nodes, protocol, timemargin)

    # Execute the command line to start the match.
    if os_name.lower() == 'windows':
        process = Popen(str(tour_manager) + command, stdout=PIPE, text=True)
    else:
        process = Popen(shlex.split(str(tour_manager) + command), stdout=PIPE, text=True)
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
    parser.add_argument('--protocol', required=False,
                        help='Engine filename or engine path and filename, default=uci',
                        default='uci')
    parser.add_argument('--base-time-sec', required=False,
                        help='Base time in sec for time control.')
    parser.add_argument('--inc-time-sec', required=False,
                        help='Increment time in sec for time control.')
    parser.add_argument('--depth', required=False,
                        help='The maximum search depth that the engine is'
                             ' allowed.\n'
                             'Example:\n'
                             '--depth 6 ...')
    parser.add_argument('--move-time-ms', required=False,
                        help='The maximum search time in milliseconds. This is only for cutechess. Example\n'
                             '--move-time-ms 1000\n'
                             'and engine is set to search at 1s. The cutechess\n'
                             'timemargin is set at 50ms.')
    parser.add_argument('--nodes', required=False,
                        help='The maximum nodes that the engine is'
                             ' allowed to search. This is only for cutechess. Do not use other\n'
                             'move control like --base-time-sec or'
                             ' --depth or --move-time-sec example:\n'
                             '--nodes 500 ...')
    parser.add_argument('--time-margin', required=False,
                        help='time margin in milliseconds for cutechess interface (not required), default=50.',
                        default=50)
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
    parser.add_argument('--match-manager-path', required=True,
                        help='Match manager path and/or filename. Example:\n'
                             'cutechess:\n'
                             '--match-manager-path c:/chess/tourney_manager/cutechess/cutechess-cli.exe\n'
                             'duel.py for xboard engines:\n'
                             '--match-manager-path python c:/chess/tourney_manager/duel/duel.py\n'
                             'or\n'
                              '--match-manager-path c:/python3/python c:/chess/tourney_manager/duel/duel.py\n'
                             'enhance.py for bench\n'
                             '--match-manager-path python c:/lakas/interface/enhance.py')
    parser.add_argument('--enhance', action='store_true',
                        help='a flag to run engine with bench command and return the nodes as objective value.')
    parser.add_argument('--enhance-hashmb', required=False, type=int,
                        help='hash size in mb, default=64',
                        default=64)
    parser.add_argument('--enhance-threads', required=False, type=int,
                        help='engine threads, default=1',
                        default=1)
    parser.add_argument('--enhance-limitvalue', required=False, type=int,
                        help='search limit value, default=15',
                        default=15)
    parser.add_argument('--enhance-fenfile', required=False, type=str,
                        help='position file in fen or epd format used for the bench command, default=default',
                        default='default')
    parser.add_argument('--enhance-limittype', required=False, type=str,
                        help='search limit type [depth, perft, nodes, movetime], default=depth',
                        default='depth')
    parser.add_argument('--enhance-evaltype', required=False, type=str,
                        help='eval type [mixed, classical, nnue], default=mixed',
                        default='mixed')
    parser.add_argument('--enhance-posperfile', required=False, type=int,
                        help='number of positions in the bench file, default=50',
                        default=50)
    parser.add_argument('--opening-file', required=False, type=str,
                        help='start opening filename in pgn or fen/epd format')
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
    parser.add_argument('--cutechess-debug', action='store_true',
                        help='Enable -debug flag of cutechess-cli, this will output engine logging.')
    parser.add_argument('--cutechess-wait', required=False, type=int,
                        help='Sets the -wait N option of cutechess-cli,\n'
                             'that is wait N milliseconds between games,\n'
                             'default=5000 or 5s.',
                        default=5000)

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

    opening_file_format = 'pgn'
    if not args.enhance:
        if args.opening_file is None:
            raise Exception('start opening file is missing!')
        else:
            opening_file_format = Path(args.opening_file).suffix[1:]
            if opening_file_format == 'fen' or opening_file_format == 'epd':
                opening_file_format = 'epd'

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

    logger.info(f'Lakas {__version__}')
    logger.info(f'nevegrad {ng.__version__}')

    logger.info(f'input param: {input_param}\n')
    init_param = set_param(input_param)

    logger.info(f'total budget: {args.budget}')
    logger.info(f'games/budget: {args.games_per_budget}')
    logger.info(f'move control: base_time_sec: {args.base_time_sec}, '
                f'inc_time_sec: {args.inc_time_sec}, depth={args.depth},'
                f' nodes={args.nodes}')

    # Prepare parameters to be optimized.
    arg = {}
    for k, v in input_param.items():
        if type(v) == list:
            arg.update({k: ng.p.Choice(v)})
        else:
            if isinstance(v["init"], int):
                arg.update({k: ng.p.Scalar(init=v['init'], lower=v['lower'],
                                           upper=v['upper']).set_integer_casting()})
            elif isinstance(v["init"], float):
                arg.update({k: ng.p.Scalar(init=v['init'], lower=v['lower'],
                                           upper=v['upper'])})

    instrum = ng.p.Instrumentation(**arg)

    # deterministic_function in Nevergrad default since
    # nevergrad==0.4.3 is true. Lakas by default is false.
    if not deterministic_function:
        instrum.descriptors.deterministic_function = False

    logger.info(f'parameter dimension: {instrum.dimension}')
    logger.info(f'deterministic function: {deterministic_function}')
    if use_best_param:
        logger.info(f'use best param: {use_best_param}, optimizer suggested param is against the best param found so far')
    else:
        logger.info(f'use best param: {use_best_param}, optimizer suggested param is always against the init param')
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

    best_param = {}
    best_loss = None

    # Get best loss.

    # If there is no existing optimization data we will tell
    # the optimizer (except spsa) the best param is the init param and
    # its loss is 0.5 or best result threshold.
    if optimizer.num_ask < 1:
        if optimizer_name != 'spsa':

            # Dynamic opp: optimizer opponent is the best param found so far.
            if use_best_param:
                best_loss = 1.0 - best_result_threshold

            # Fix opp: optimizer opponent is always the default or init param.
            else:
                best_loss = 0.5

            optimizer.tell(instrum, best_loss)

            recommendation = optimizer.provide_recommendation()
            recommendation_value = recommendation.value
            best_param = recommendation_value[1]
            curr_best_loss = optimizer.current_bests
            best_loss = curr_best_loss["average"].mean

            if output_data_file is not None:
                optimizer.dump(output_data_file)

    # If there is already existing optimization data.
    elif input_data_file is not None:
        recommendation = optimizer.provide_recommendation()
        recommendation_value = recommendation.value
        best_param = recommendation_value[1]
        curr_best_loss = optimizer.current_bests
        best_loss = curr_best_loss["average"].mean

    objective = Objective(optimizer, args.engine, input_param, init_param,
                          args.opening_file, opening_file_format,
                          best_param, best_loss,
                          games_per_budget=args.games_per_budget,
                          depth=args.depth, concurrency=args.concurrency,
                          base_time_sec=args.base_time_sec,
                          inc_time_sec=args.inc_time_sec,
                          move_time_ms=args.move_time_ms,
                          nodes=args.nodes,
                          match_manager=args.match_manager,
                          match_manager_path=args.match_manager_path,
                          variant=args.variant,
                          best_result_threshold=best_result_threshold,
                          use_best_param=use_best_param,
                          common_param=common_param,
                          deterministic_function=deterministic_function,
                          optimizer_name=optimizer_name, spsa_scale=spsa_scale,
                          proc_list=proc_list,
                          cutechess_debug=args.cutechess_debug,
                          cutechess_wait=args.cutechess_wait,
                          protocol=args.protocol,
                          timemargin = args.time_margin,
                          enhance=args.enhance,
                          enhance_hashmb=args.enhance_hashmb,
                          enhance_threads=args.enhance_threads,
                          enhance_limitvalue=args.enhance_limitvalue,
                          enhance_fenfile=args.enhance_fenfile,
                          enhance_limittype=args.enhance_limittype,
                          enhance_evaltype=args.enhance_evaltype,
                          enhance_posperfile=args.enhance_posperfile)

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

    # Optimization done, get the best param.
    recommendation = optimizer.provide_recommendation()
    best_param = recommendation.value
    logger.info(f'best_param: {best_param[1]}')

    # Output for match manager.
    option_output = ''
    for k, v in best_param[1].items():
        option_output += f'option.{k}={v} '
    logger.info(f'{option_output}\n')


if __name__ == "__main__":
    main()
