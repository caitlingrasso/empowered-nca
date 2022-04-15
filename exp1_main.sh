#!/bin/bash

RUNS=1

# CTRL 1 (bi-loss)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='error' --objective2='None' --target=square
done

# CTRL 2 (tri-loss)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='error_phase1' --objective2='error_phase2' --target=square
done

# CTRL 3 (bi-empowerment)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='MI' --objective2='None' --target=square
done

# TX (tri-loss-empowerment)
for r in `seq 1 $RUNS`
do
    python3 main.py --run=${r} --objective1='error' --objective2='MI' --target=square
done
