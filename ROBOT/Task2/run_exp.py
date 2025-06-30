import numpy as np
import time
import random
import panda_py
import sys
import csv
import sounddevice as sd

NUM_TRIALS = 50
LOG_FILE = "/home/sandor/frankaR/ROBOT/Task2/robot_experiment_log.csv"
positions = np.loadtxt('/home/sandor/frankaR/ROBOT/Task2/saved_positions.txt')

if len(positions) < 5:
    raise ValueError("The saved positions file should contain at least 5 positions.")

LEFT_POSITION, LEFT_UP_POSITION, MIDDLE_POSITION, RIGHT_UP_POSITION, RIGHT_POSITION = positions[:5]
SPEED_FACTOR = 0.3

panda = panda_py.Panda(sys.argv[1])
gripper = panda_py.libfranka.Gripper('172.16.0.2')

def log_movement(trial, direction, action):
    with open(LOG_FILE, "a", newline='') as log:
        csv.writer(log).writerow([time.time(), trial, direction, action])

def beep(freq=440, duration=0.2, volume=0.5):
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = volume * np.sin(2 * np.pi * freq * t)
    sd.play(tone, fs)
    sd.wait()

if __name__ == '__main__':
    with open(LOG_FILE, "w", newline='') as log:
        csv.writer(log).writerow(["Timestamp", "Trial", "Direction", "Action"])

    start_time = time.time()

    current_side = "left"
    gripper.move(0.08, 0.1)
    panda.move_to_joint_position(LEFT_POSITION, speed_factor=SPEED_FACTOR)
    time.sleep(1.5)
    gripper.grasp(0.03, 0.02, 0.0001)

    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n==== Trial {trial}/{NUM_TRIALS} ====")
        beep(700, 0.15, 0.3)

        if current_side == "left":
            trajectory = [LEFT_UP_POSITION, MIDDLE_POSITION, RIGHT_UP_POSITION, RIGHT_POSITION]
            direction = "left_to_right"
            end_position = RIGHT_POSITION
        else:
            trajectory = [RIGHT_UP_POSITION, MIDDLE_POSITION, LEFT_UP_POSITION, LEFT_POSITION]
            direction = "right_to_left"
            end_position = LEFT_POSITION

        drop = random.random() < 0.3
        action = "drop" if drop else "correct"
        print(f"Moving {direction} | Drop block? {drop}")

        if drop:
            panda.move_to_joint_position(trajectory[0:2], speed_factor=SPEED_FACTOR)
            gripper.move(0.08, 0.1)
            print(">> Dropped block at middle.")
            panda.move_to_joint_position(trajectory[2:], speed_factor=SPEED_FACTOR)
        else:
            panda.move_to_joint_position(trajectory, speed_factor=SPEED_FACTOR)

        if action == "drop":
            print("Waiting for user to place block at end position...")
            time.sleep(6)
            gripper.move(0.03, 0.1)

        log_movement(trial, direction, action)
        print(f"Trial {trial} completed.")
        current_side = "right" if current_side == "left" else "left"
        time.sleep(1)

    duration = time.time() - start_time
    print(f"Experiment completed successfully in {duration:.2f} seconds.")
