# LeRobot

## Find the USB Port
First, connect the PCB to the laptop using USB-C.
Then run inside the workspace env,
```bash
lerobot-find-port
```

## Setting Motor ID
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM0 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```
