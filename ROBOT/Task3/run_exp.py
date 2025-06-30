import os
import time
import random
import csv
import ast
import panda_py
import sys
import numpy as np
import sounddevice as sd

CORRECT_PROB = 0.3
RESET_STATE = (3, 1)
GOAL_STATE = (1, 3)
NUM_TRIALS = 30
SPEED_FACTOR = 0.35
BUCKET_FILE = "/home/sandor/frankaR/ROBOT/Task3/buckets.txt"
BASE_DIR = "/home/sandor/frankaR/ROBOT/Task3/participants"

grid_positions = {
    (1, 1): [-0.0254, 0.4809, -0.2196, -2.1889, 0.1165, 2.6415, -1.0980],
    (1, 2): [-0.0694, 0.2762, -0.2195, -2.5629, 0.1164, 2.7972, -1.1763],
    (1, 3): [-0.2126, 0.0934, -0.2360, -2.8753, 0.1142, 2.9090, -1.3537],
    (2, 1): [0.0994, 0.4540, -0.1100, -2.2462, 0.1142, 2.6705, -0.8718],
    (2, 2): [0.0983, 0.2443, -0.1109, -2.6422, 0.1142, 2.8527, -0.8829],
    (2, 3): [0.0177, 0.0247, -0.0738, -2.9824, 0.1143, 2.9487, -0.8801],
    (3, 1): [0.2215, 0.4560, -0.0254, -2.2216, 0.1143, 2.6733, -0.6812],
    (3, 2): [0.2780, 0.2445, -0.0280, -2.6075, 0.1143, 2.8233, -0.6189],
    (3, 3): [0.3919, 0.0479, -0.0201, -2.9442, 0.1142, 2.9408, -0.4931]
}

def log_movement(trial, current, target, correct, punished, path):
    with open(path, "a", newline="") as log:
        csv.writer(log).writerow([
            time.time(), trial, current, target,
            "Correct" if correct else "Incorrect",
            "Punishment" if punished else "No Punishment"
        ])

def beep(freq=440, duration=0.2, volume=0.5):
    t = np.linspace(0, duration, int(44100 * duration), False)
    sd.play(volume * np.sin(2 * np.pi * freq * t), 44100)
    sd.wait()

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def closer_to_goal(prev, new):
    return manhattan(new, GOAL_STATE) < manhattan(prev, GOAL_STATE)

def load_paths(filepath):
    safe_paths, punishment_paths = [], []
    in_safe_section = True  # Start in safe_paths section

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if line == "safe_paths:":
                in_safe_section = True
            elif line == "punishment_paths:":
                in_safe_section = False
            elif line:
                path = ast.literal_eval(line)
                (safe_paths if in_safe_section else punishment_paths).append(path)

    return safe_paths, punishment_paths

def ensure_log(participant_id):
    path = os.path.join(BASE_DIR, participant_id)
    os.makedirs(path, exist_ok=True)
    logfile = os.path.join(path, f"log_{participant_id}.csv")
    with open(logfile, "w", newline="") as log:
        csv.writer(log).writerow([
            "Timestamp", "Trial", "Current Position", "Intended Position", "Correctness", "Punishment"
        ])
    return logfile

def run_trial(panda, trial, safe_paths, punishment_paths, logfile):
    panda.move_to_joint_position(grid_positions[(1, 1)], speed_factor=SPEED_FACTOR)
    time.sleep(2)
    beep(700, 0.15, 0.3)

    path = random.choice(safe_paths if random.random() < 0.7 else punishment_paths)
    print(f"\nTrial {trial} | Path: {path}")

    for i in range(1, len(path)):
        prev, step = path[i - 1], path[i]
        speed = 0.2 if (prev, step) in [((1, 1), (2, 1)), ((2, 1), (1, 1)), ((2, 1), (3, 1)), ((3, 1), (2, 1))] else SPEED_FACTOR
        panda.move_to_joint_position(grid_positions[step], speed_factor=speed)
        time.sleep(0.6)
        correct = closer_to_goal(prev, step)
        log_movement(trial, prev, step, correct, step == RESET_STATE, logfile)

def main():
    participant_id = input("Enter the participant ID: ").strip()
    safe_paths, punishment_paths = load_paths(BUCKET_FILE)
    logfile = ensure_log(participant_id)
    panda = panda_py.Panda(sys.argv[1])

    start_time = time.time()
    for trial in range(1, NUM_TRIALS + 1):
        run_trial(panda, trial, safe_paths, punishment_paths, logfile)
        print("Trial done\n")

    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")
    panda.move_to_start()

if __name__ == "__main__":
    main()
