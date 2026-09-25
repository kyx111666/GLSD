"""Server launcher for the fixed-structure SAMMLV residual screen.

The server must contain a local mirror of the sealed Drive tree and the
passed stage-0 audit directory. No model training or response generation is
performed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Local mirror of the Drive MyDrive tree')
    parser.add_argument('--metst-root', type=Path, required=True,
                        help='ME-TST checkout containing Utils and training_utils.py')
    parser.add_argument('--audit', type=Path, required=True,
                        help='Extracted passed stage-0 SAMMLV audit directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for path in (args.data_root, args.metst_root, args.audit):
        if not path.exists():
            raise SystemExit('missing required path: '+str(path))
    env = dict(os.environ)
    env['GLSD_DATA_ROOT'] = str(args.data_root.resolve())
    env['ME_TST_ROOT'] = str(args.metst_root.resolve())
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(args.metst_root.resolve()),
                                         env.get('PYTHONPATH', '')])
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONWARNINGS'] = 'ignore:The behavior of DataFrame concatenation with empty or all-NA entries is deprecated:FutureWarning'
    test = [sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT),
            '-p', 'test_reference_residual.py', '-v']
    print('Running local critical-path tests...', flush=True)
    subprocess.run(test, cwd=args.metst_root, env=env, check=True)
    command = [sys.executable, '-u', str(ROOT/'colab_reference_residual_entry.py'),
               '--mode', 'screen', '--setting', 'sammlv', '--output', str(args.output),
               '--audit', str(args.audit)]
    if args.resume:
        command.append('--resume')
    print('GLSD_DATA_ROOT =', env['GLSD_DATA_ROOT'], flush=True)
    print('ME_TST_ROOT =', env['ME_TST_ROOT'], flush=True)
    print('OUTPUT =', args.output, flush=True)
    subprocess.run(command, cwd=args.metst_root, env=env, check=True)


if __name__ == '__main__':
    main()
