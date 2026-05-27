# S600 BPU runtime

SAM_s600 uses S600 BPU through the D-Robotics runtime and converted HBM model files.

Useful runtime checks on the board:

```bash
hb_version
hrt_ucp_monitor -b -d 1000 -e bpu
hrt_model_exec model_info --model_file <model.hbm>
```

`hobot-smi` is not used as a project dependency because it targets PAC/PCIe management rather than local SAM3 BPU utilization.
