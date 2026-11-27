#!/usr/bin/env bash
# Detached launcher: runs the chain with its output in run_all.out.
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"   # a shell started from Windows has no Unix PATH
cd "$(dirname "$0")/../.."
exec bash outputs/p16_budget_pilot/run_all.sh > outputs/p16_budget_pilot/run_all.out 2>&1
