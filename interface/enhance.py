#!/usr/bin/env python


""""
Enhance.py

Run engine bench command and return nodes.

format:
    bench <hash> <threads> <limitvalue> <fenfile | default>
          <limittype [depth(default), perft, nodes, movetime]> <evaltype [mixed(default), classical, NNUE]>

bench 128 1 4 file.epd depth mixed
"""


__author__ = 'fsmosca'
__script_name__ = 'enhance'
__version__ = 'v0.3.1'
__credits__ = ['joergoster', 'musketeerchess']


from pathlib import Path
import subprocess
import argparse
import time
import random
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import logging
from statistics import mean


logging.basicConfig(
    filename='enhance_log.txt', filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - pid%(process)5d - %(levelname)5s - %(message)s')


class Enhance:
    def __init__(self, engineinfo, hashmb=64, threads=1, limitvalue=15,
                 fenfile='default', limittype='depth', evaltype='mixed',
                 concurrency=1, randomizefen=False, posperfile=50):
        self.engineinfo = engineinfo
        self.hashmb = hashmb
        self.threads = threads
        self.limitvalue = limitvalue
        self.fenfile=fenfile
        self.limittype=limittype
        self.evaltype=evaltype
        self.concurrency = concurrency
        self.randomizefen = randomizefen
        self.posperfile = posperfile

        self.nodes = None  # objective value

    def send(self, p, command):
        """ Send msg to engine """
        p.stdin.write('%s\n' % command)
        logging.debug('>> %s' % command)
        
    def read_engine_reply(self, p, command):
        """ Read reply from engine """
        self.nodes = None
        for eline in iter(p.stdout.readline, ''):
            line = eline.strip()
            logging.debug('<< %s' % line)
            if command == 'uci' and 'uciok' in line:
                break
            if command == 'isready' and 'readyok' in line:
                break
            # Nodes searched  : 3766422
            if command == 'bench' and 'nodes searched' in line.lower():
                self.nodes = int(line.split(': ')[1])
            elif command == 'bench' and 'Nodes/second' in line:
                break

    def bench(self, fenfn) -> int:
        """
        Run the engine, send a bench command and return the nodes searched.
        """
        folder = Path(self.engineinfo['cmd']).parent

        # Start the engine.
        proc = subprocess.Popen(self.engineinfo['cmd'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1, cwd=folder)

        self.send(proc, 'uci')
        self.read_engine_reply(proc, 'uci')

        # Set param values to be optimized.
        for k, v in self.engineinfo['opt'].items():
            self.send(proc, f'setoption name {k} value {v}')

        self.send(proc, 'isready')
        self.read_engine_reply(proc, 'isready')

        self.send(proc, f'bench {self.hashmb} {self.threads} {self.limitvalue} {fenfn} {self.limittype} {self.evaltype}')
        # self.send(proc, f'bench')
        self.read_engine_reply(proc, 'bench')

        # Quit the engine.
        self.send(proc, 'quit')

        return self.nodes

    def delete_file(self, fn):
        filepath = Path(fn)
        filepath.unlink(missing_ok=True)  # python 3.8 missing_ok is added
        logging.info(f'The file {fn} was deleted.')

    def generate_files(self):
        """
        Read the input fen file, convert it to a list, sort it randomly if required.
        If there are 2 concurrencies, 2 files will be created (file0.fen, file1.fen).
        The number of positions in each file is determined by the posperfile option value.
        These files will be used in the bench command as in:
            bench 64 1 12 file0.fen depth mixed
        """
        t0 = time.perf_counter()

        fenlist, filelist = [], []
        numpos = self.posperfile

        fen_file = Path(self.fenfile)
        if not fen_file.is_file():
            if self.fenfile != 'default':
                print(f'Warning, {self.fenfile} is missing, default will be used!')
                logging.warning(f'{self.fenfile} is missing, default will be used!')
            return filelist

        ext = fen_file.suffix

        with open(self.fenfile) as f:
            for lines in f:
                fen = lines.rstrip()
                fenlist.append(fen)

        # sort randomly this big fen list
        if self.randomizefen:
            random.shuffle(fenlist)

        # Extract the specified number of fens per bench run.
        # If concurrency is 2 we will create 2 files namely file0.fen and file1.fen
        for i in range(self.concurrency):
            # i starts at 0 and if numpos is 50:
            # start=0, end=50
            # fenlist[0:50] will extract the first 50 fens as a list.
            start = i * numpos
            end = (i+1) * numpos

            fens = fenlist[start:end]
            fn = f'file{i}{ext}'

            # Save the list of fens in the file.
            with open(fn, 'w') as f:
                for fen in fens:
                    f.write(f'{fen}\n')

            pathandfile = Path(Path().absolute(), fn).as_posix()
            filelist.append(pathandfile)

        logging.info(f'numfiles: {len(filelist)}')
        logging.info(f'file generation elapse {time.perf_counter() - t0:0.2f}s')

        return filelist

    def run(self):
        """
        Run the engine with bench command to get the nodes searched and
        return it as the objective value.
        """
        objectivelist, joblist = [], []

        fenfiles = self.generate_files()

        # Use Python 3.8 or higher.
        with ProcessPoolExecutor(max_workers=self.concurrency) as executor:
            if len(fenfiles) == 0:
                job = executor.submit(self.bench, 'default')
                joblist.append(job)
            else:
                for fn in fenfiles:
                    job = executor.submit(self.bench, fn)
                    joblist.append(job)

            for future in concurrent.futures.as_completed(joblist):
                try:
                    nodes_searched = future.result()
                    objectivelist.append(nodes_searched)
                except concurrent.futures.process.BrokenProcessPool as ex:
                    logging.exception(f'exception: {ex}')
                    raise

        # Delete fen files.
        for fn in fenfiles:
            pass  # self.delete_file(fn)

        logging.debug(f'bench nodes: {objectivelist}')
        logging.debug(f'mean nodes: {int(mean(objectivelist))}')
        logging.debug(f'sum nodes: {sum(objectivelist)}')

        # This is used by optimizer to signal that the job is done.
        print(f'total nodes searched from {self.concurrency} workers: {sum(objectivelist)}')
        print('bench done')


def define_engine(engine_option_value):
    """
    Define engine files, and options.
    """
    optdict = {}
    engineinfo = {'cmd': None, 'opt': optdict}

    for eng_opt_val in engine_option_value:
        for value in eng_opt_val:
            if 'cmd=' in value:
                engineinfo.update({'cmd': value.split('=')[1]})
            elif 'option.' in value:
                # option.QueenValueOpening=1000
                optn = value.split('option.')[1].split('=')[0]
                optv = value.split('option.')[1].split('=')[1]
                optdict.update({optn: optv})
                engineinfo.update({'opt': optdict})

    return engineinfo


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description='Run bench command.',
        epilog='%(prog)s')
    parser.add_argument('-engine', nargs='*', action='append', required=True,
                        metavar=('cmd=', 'option.<optionname>=value'),
                        help='Define engine filename and option, required=True. Example:\n'
                        '-engine cmd=eng.exe option.FutilityMargin=120 option.MoveCount=1000')
    parser.add_argument('-concurrency', required=False,
                        help='the number of benches to run in parallel, default=1',
                        type=int, default=1)
    parser.add_argument('-hashmb', required=False, help='memory value in mb, default=64',
                        type=int, default=64)
    parser.add_argument('-threads', required=False, help='number of threads, default=1',
                        type=int, default=1)
    parser.add_argument('-limitvalue', required=False,
                        help='a number that limits the engine search, default=15',
                        type=int, default=15)
    parser.add_argument('-fenfile', required=False,
                        help='Filename of FEN or EPD file, default=default',
                        default='default')
    parser.add_argument('-limittype', required=False,
                        help='the type of limit can be depth, perft, nodes and movetime, default=depth',
                        type=str, default='depth')
    parser.add_argument('-evaltype', required=False,
                        help='the type of eval to use can be mixed, classical or NNUE, default=mixed',
                        type=str, default='mixed')
    parser.add_argument('-randomizefen', action='store_true',
                        help='A flag to randomize position before using it in bench when position file is used.')
    parser.add_argument('-posperfile', required=False,
                        help='the number of positions in the file to be used in the bench, default=50',
                        type=int, default=50)
    parser.add_argument('-v', '--version', action='version', version=f'{__version__}')

    args = parser.parse_args()

    # Define engine files, name and options.
    engineinfo = define_engine(args.engine)

    # Exit if engine file is not defined.
    if engineinfo['cmd'] is None:
        print('Error, engines are not properly defined!')
        return

    tstart = time.perf_counter()

    duel = Enhance(engineinfo, hashmb=args.hashmb, threads=args.threads,
                   limitvalue=args.limitvalue, fenfile=args.fenfile,
                   limittype=args.limittype, evaltype=args.evaltype,
                   concurrency=args.concurrency,
                   randomizefen=args.randomizefen,
                   posperfile=args.posperfile)
    duel.run()

    logging.info(f'total elapse time: {time.perf_counter() - tstart:0.2f}s')


if __name__ == '__main__':
    main()
