# Pallet Detection & Alignment System

A real-time computer vision system that detects industrial pallets and guides a robotic forklift to align correctly with the pallet fork openings — no hardware required.

## Demo
> Run `python align.py` and point your camera at a pallet.

## Problem
Manual alignment of robotic forklifts with pallets causes:
- Load instability and pallet damage
- Safety risks for workers
- Operational inefficiency in warehouses

## Solution
A real-time CV pipeline that:
1. Detects pallets using a fine-tuned YOLOv8 model
2. Estimates offset and distance relative to the robot camera
3. Outputs directional movement commands in real time
4. Visualizes everything on a live dashboard

## Tech Stack
- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

## Setup
```bash
pip install ultralytics opencv-python numpy
python align.py
```

## Model
Fine-tuned YOLOv8n on the [Pallet Detection Dataset](https://universe.roboflow.com/sundharesan-kumaresan/pallet-detection-ith6b) by sundharesan-kumaresan — 3,300+ images, mAP 99.5%, CC BY 4.0 license.

## Project Structure
```
pallet-alignment/
├── align.py          # Main live alignment script
├── detect.py         # Single image detection script
├── train.py          # Model fine-tuning script
├── data.yaml         # Dataset config
└── runs/             # Training outputs and weights
```
