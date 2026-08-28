# Interface contract (CONTRACT-TIER: changes require freeze process + coordinator sign-off)

- editable files: models/AirFM/unified_model.py (lambda assembly in unified_pretrain_forward only), physics_distance.py, masked.py;
- soft_dtw_cuda.py is locked: change its invocation/weights, not the kernel;
- pretraining data manifest and mask ratio protocol are frozen.
