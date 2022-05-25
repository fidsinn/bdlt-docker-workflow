srun \
--export=ALL \
--container-image=./20ng.sqsh \
--container-mounts=/mnt/ceph:/mnt/ceph \
bash -c " cd ~ && PYTHONPATH=. python3 $SWEEP_SERVER" \
| tee >&2 >(tail -1)