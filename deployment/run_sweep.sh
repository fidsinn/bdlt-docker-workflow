SWEEP_SERVER="sweep_server.py" \
SWEEP_AGENT="sweep_agent.py" \
WANDB_ENTITY="YOUR_WANDB_USERNAME" \
WANDB_PROJECT="YOUR_WANDB_USERNAME" \
SWEEPID=$( ./sweep_server.sh ) \
NWORKERS=2 \
./sweep_loop.sh