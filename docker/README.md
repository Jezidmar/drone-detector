Along with Dockerfile, this directory contains other necessary files to run the detector module.
This is a first release, and it supports *only* CPU inference.
The int4 quantized model is saved on the cloud, it shall automatically be downloaded by dockerfile script.

**Step 1. Build the Docker image:**
```bash
docker build -t drone-detector .
```
**Step 2. Run the Docker image:**
```bash
docker run --rm -it drone-detector
```

If you want to adjust the hyperparameters, you can use something like: 
```bash
docker run -it  --device /dev/snd   --group-add $(getent group audio | cut -d: -f3)   -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native   -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native   -v ~/.config/pulse/cookie:/root/.config/pulse/cookie   --name trial_1   drone-detector:latest   bash python3 run_app.py"
```