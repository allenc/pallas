# Pallas C++ Components

## Code Organization

```
pallas/ 
    core-cpp/                 [common lib]
        microservice.h
        shm_spsc.h
        timer.h
    model-cpp/                [model interface lib]
    
    starburstd/               [camera interface daemon]
    starforged/               [camera visualizer developer tools: camera viewer, camera calibrator]
    psystreamd/               [ml vision pipeline executor daemon]
    psyforged/                [ml vision pipeline developer tools: data viewer, model evals]

    joyd/                     [health monitoring daemon]
```

## Building

```bash
cmake -B build && cmake --build build
```

## Using the PS3 Eye Camera

### Prerequisites

Make sure you have the necessary dependencies installed:
- libusb-1.0
- opencv

### Starting the PS3 Eye Camera Pipeline

1. **Launch the camera daemon**:
   ```bash
   ./build/starburstd
   ```
   This starts the PS3 Eye camera service which captures frames and makes them available to other services.

2. **Launch the streaming service**:
   ```bash
   ./build/streamd
   ```
   This starts the HTTP streaming server that provides camera streams to the frontend and processes vision tasks.

3. **Access the web interface**:
   Open a web browser and navigate to:
   ```
   http://localhost:8000
   ```
   The frontend allows you to view the camera feed and interact with the vision pipeline.

### Troubleshooting

- If you encounter permission issues with the USB device, you may need to run with sudo or add udev rules
- To check if your PS3 Eye camera is detected, run `lsusb` and look for a device with ID 1415:2000
- If the camera freezes after a short time:
  - This may be caused by buffer issues or camera disconnection
  - Try running starburstd with sudo: `sudo ./build/starburstd`
  - Make sure the PS3 Eye camera is connected directly to your computer, not through a hub
  - Check the camera's power supply - some PS3 Eye cameras need more power than standard USB ports provide
  - Try using a different USB port, preferably USB 2.0 instead of USB 3.0
- If the camera doesn't display in the frontend, check the logs from starburstd and streamd for error messages

## Core Components

Implementation of a single-producer, single-consumer, lock-free queue