import numpy as np
import time
import random
import panda_py
import sys
import csv
import pygame
import sounddevice as sd

NUM_TRIALS = 50
SPEED_FACTOR = 0.35
LOG_FILE = "/home/sandor/frankaR/ROBOT/Task1/robot_experiment_log.csv"

positions = np.loadtxt('/home/sandor/frankaR/ROBOT/Task1/saved_positions.txt')
if len(positions) < 3:
    raise ValueError("The saved positions file should contain at least 3 positions")

START_POSITION = positions[0]
YELLOW_POSITION = positions[1]
BLUE_POSITION = positions[2]

panda = panda_py.Panda(sys.argv[1])

pygame.init()
screen = pygame.display.set_mode((pygame.display.Info().current_w, pygame.display.Info().current_h))
pygame.display.set_caption("Color Display")
font = pygame.font.SysFont(None, 55)

def beep(freq=700, duration=0.15, volume=0.3):
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = volume * np.sin(2 * np.pi * freq * t)
    sd.play(tone, fs)
    sd.wait()

def log_movement(trial, current_pos, intended_pos, correct_move):
    with open(LOG_FILE, "a", newline='') as log:
        csv.writer(log).writerow([time.time(), trial, current_pos, intended_pos, correct_move])

def show_color(color):
    color_map = {"yellow": (255, 255, 0), "blue": (0, 0, 255), "black": (0, 0, 0)}
    screen.fill(color_map[color])
    pygame.display.flip()

with open(LOG_FILE, "w", newline='') as log:
    csv.writer(log).writerow(["Timestamp", "Trial", "From Position", "To Position", "Correctness"])

experiment_start = time.time()

for trial in range(1, NUM_TRIALS + 1):
    print(f"\n==== Trial {trial}/{NUM_TRIALS} ====")
    panda.move_to_joint_position(START_POSITION, speed_factor=SPEED_FACTOR)
    time.sleep(1)

    color = random.choice(["yellow", "blue"])
    show_color(color)
    time.sleep(1)

    if random.random() < 0.3:
        error = True
        target_color = "blue" if color == "yellow" else "yellow"
    else:
        error = False
        target_color = color

    target_position = YELLOW_POSITION if target_color == "yellow" else BLUE_POSITION

    print(f"Moving to {target_color.capitalize()} Position")
    beep()
    panda.move_to_joint_position(target_position, speed_factor=SPEED_FACTOR)
    time.sleep(1)

    log_movement(trial, color, target_color, error)
    print("Trial completed.")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    show_color("black")

experiment_end = time.time()
total_duration = experiment_end - experiment_start
print(f"Experiment completed successfully in {total_duration:.2f} seconds.")
pygame.quit()
