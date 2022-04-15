#!/bin/bash

RUNS=1
TARGET='triangle' # targets = ['triangle', 'circle', 'biped', 'xenobot']

# CTRL 1 (bi-loss)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='error' --objective2='None' --target=${TARGET}
done

# CTRL 2 (tri-loss)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='error_phase1' --objective2='error_phase2' --target=${TARGET}
done

# CTRL 3 (bi-empowerment)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='MI' --objective2='None' --target=${TARGET}
done

# TX (tri-loss-empowerment)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='error' --objective2='MI' --target=${TARGET}
done
