import os
import sys
import time
import random
import csv
import json
import ast
from datetime import datetime
import panda_py
import pygame
import sounddevice as sd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ============= Configuration =============
CONFIG = {
    "num_trials": {
        "task1": 50,
        "task2": 50,
        "task3": 30
    },
    "speed_factors": {
        "task1": 0.35,
        "task2": 0.3,
        "task3": 0.35
    },
    "data_dir": "experiment_data",
    "demo_duration": 1,  # seconds for each demo action
    "pause_duration": 1,  # seconds to wait after pause
    "robot_ip": "172.16.0.2",  # Robot IP address
    "gripper_ip": "172.16.0.2",  # Gripper IP address
    "error_probabilities": {
        "task1": 0.3,  # 30% error probability for task1
        "task2": 0.3,  # 20% error probability for task2; willen we deze verhogen naar 30%? Nu hebben we ~10 error trials per keer.
        "task3": 0.3   # 30% error probability for task3; 
    },
    "break_interval": 4.5,  # Number of minutes before a break
    "task1_positions_file": "/home/sandor/frankaR/ROBOT/Task1/saved_positions.txt",
    "task2_positions_file": "/home/sandor/frankaR/ROBOT/Task2/saved_positions.txt",
    "task3_buckets_file": "/home/sandor/frankaR/ROBOT/Task3/buckets.txt"
}

# Validate configuration
def validate_config():
    """Validate configuration values."""
    required_keys = ["num_trials", "speed_factors", "data_dir", "robot_ip", "gripper_ip", "error_probabilities", "break_interval"]
    for key in required_keys:
        if key not in CONFIG:
            raise ValueError(f"Missing required configuration key: {key}")
    
    for task in ["task1", "task2", "task3"]:
        if task not in CONFIG["num_trials"] or CONFIG["num_trials"][task] <= 0:
            raise ValueError(f"Invalid number of trials for {task}")
        if task not in CONFIG["speed_factors"] or CONFIG["speed_factors"][task] <= 0:
            raise ValueError(f"Invalid speed factor for {task}")
        if task not in CONFIG["error_probabilities"] or CONFIG["error_probabilities"][task] < 0 or CONFIG["error_probabilities"][task] > 1:
            raise ValueError(f"Invalid error probability for {task}")
    
    if CONFIG["break_interval"] <= 0:
        raise ValueError("Break interval must be greater than 0")

# Validate configuration at startup
validate_config()

# ============= Task 1 Constants =============
# Load Task 1 positions
try:
    TASK1_POSITIONS = np.loadtxt(CONFIG["task1_positions_file"])
    if len(TASK1_POSITIONS) < 3:
        raise ValueError("Task 1 positions file should contain at least 3 positions")

    TASK1_START_POSITION = TASK1_POSITIONS[0]
    TASK1_YELLOW_POSITION = TASK1_POSITIONS[1]
    TASK1_BLUE_POSITION = TASK1_POSITIONS[2]
except Exception as e:
    print(f"Error loading Task 1 positions: {str(e)}")
    sys.exit(1)

# ============= Task 2 Constants =============
# Load Task 2 positions
try:
    TASK2_POSITIONS = np.loadtxt(CONFIG["task2_positions_file"])
    if len(TASK2_POSITIONS) < 5:
        raise ValueError("Task 2 positions file should contain at least 5 positions")

    TASK2_LEFT_POSITION = TASK2_POSITIONS[0]
    TASK2_LEFT_UP_POSITION = TASK2_POSITIONS[1]
    TASK2_MIDDLE_POSITION = TASK2_POSITIONS[2]
    TASK2_RIGHT_UP_POSITION = TASK2_POSITIONS[3]
    TASK2_RIGHT_POSITION = TASK2_POSITIONS[4]
except Exception as e:
    print(f"Error loading Task 2 positions: {str(e)}")
    sys.exit(1)

# ============= Task 3 Constants =============
TASK3_GRID_POSITIONS = {
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

TASK3_RESET_STATE = (3, 1)
TASK3_GOAL_STATE = (1, 3)

# ============= Experiment Manager =============
class ExperimentManager:
    def __init__(self, participant_id: str):
        # Add "participant_" prefix if not present
        if not participant_id.startswith("participant_"):
            participant_id = f"participant_{participant_id}"
        
        # Find the next available run number
        base_dir = Path(CONFIG["data_dir"])
        run_number = 1
        while True:
            self.participant_id = f"{participant_id}_run_{run_number}"
            self.base_dir = base_dir / self.participant_id
            if not self.base_dir.exists():
                break
            run_number += 1
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize task order first
        self.tasks = ["task2", "task1", "task3"]
        random.shuffle(self.tasks)
        
        # Initialize logging
        self.log_file = self.base_dir / "experiment_log.csv"
        self.task_logs = {}
        self._setup_logging()
        
        # Initialize robot (lazy initialization for gripper)
        self.panda = None
        self.gripper = None
        self._init_robot()
        
        # Initialize display (will be initialized when needed for Task 1)
        self.screen = None
        self.font = None
        self.pygame_initialized = False
        
        # Load Task 3 paths
        self.safe_paths, self.punishment_paths = self._load_task3_paths()
    
    def _setup_logging(self):
        """Setup all logging files."""
        # Main experiment log
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Event Type", "Details"])
        
        # Task-specific logs
        for task in self.tasks:
            task_log = self.base_dir / f"{task}_log.csv"
            self.task_logs[task] = task_log
            with open(task_log, 'w', newline='') as f:
                writer = csv.writer(f)
                if task == "task1":
                    writer.writerow(["Timestamp", "Trial", "From Position", "To Position", "Correctness"])
                elif task == "task2":
                    writer.writerow(["Timestamp", "Trial", "Direction", "Action"])
                elif task == "task3":
                    writer.writerow(["Timestamp", "Trial", "Current Position", "Intended Position", "Correctness", "Punishment"])
    
    def _init_robot(self):
        """Initialize robot connection."""
        try:
            self.panda = panda_py.Panda(CONFIG["robot_ip"])
            # Gripper will be initialized when needed for Task 2
        except Exception as e:
            self.log_event("error", f"Robot initialization failed: {str(e)}")
            raise
    
    def _init_gripper(self):
        """Initialize gripper if not already initialized."""
        if self.gripper is None:
            try:
                self.gripper = panda_py.libfranka.Gripper(CONFIG["gripper_ip"])
                self.log_event("info", "Gripper initialized")
            except Exception as e:
                self.log_event("error", f"Gripper initialization failed: {str(e)}")
                raise
    
    def _init_display(self):
        """Initialize pygame display if not already initialized."""
        if not self.pygame_initialized:
            try:
                pygame.init()
                # Use the primary display
                self.screen = pygame.display.set_mode((pygame.display.Info().current_w, pygame.display.Info().current_h))
                pygame.display.set_caption("Experiment Display")
                self.font = pygame.font.SysFont(None, 55)
                self.pygame_initialized = True
                self.log_event("info", "Display initialized")
            except Exception as e:
                self.log_event("error", f"Display initialization failed: {str(e)}")
                raise
    
    def _cleanup_display(self):
        """Clean up pygame resources."""
        if self.pygame_initialized:
            try:
                pygame.quit()
                self.pygame_initialized = False
                self.screen = None
                self.font = None
                self.log_event("info", "Display cleaned up")
            except Exception as e:
                self.log_event("error", f"Display cleanup failed: {str(e)}")
    
    def _cleanup_gripper(self):
        """Clean up gripper resources."""
        if self.gripper is not None:
            try:
                # Release gripper if it's holding something
                self.gripper.move(0.08, 0.1)
                self.gripper = None
                self.log_event("info", "Gripper cleaned up")
            except Exception as e:
                self.log_event("error", f"Gripper cleanup failed: {str(e)}")
    
    def _cleanup_robot(self):
        """Clean up robot resources."""
        if self.panda is not None:
            try:
                self.panda.move_to_start()
                self.log_event("info", "Robot moved to start position")
            except Exception as e:
                self.log_event("error", f"Robot cleanup failed: {str(e)}")
    
    def _load_task3_paths(self):
        """Load Task 3 paths from file."""
        safe_paths, punishment_paths = [], []
        in_safe_section = True
        
        try:
            with open(CONFIG["task3_buckets_file"], "r") as file:
                for line in file:
                    line = line.strip()
                    if line == "safe_paths:":
                        in_safe_section = True
                    elif line == "punishment_paths:":
                        in_safe_section = False
                    elif line:
                        path = ast.literal_eval(line)
                        (safe_paths if in_safe_section else punishment_paths).append(path)
            
            if not safe_paths or not punishment_paths:
                raise ValueError("No paths loaded from buckets file")
            
            return safe_paths, punishment_paths
        except Exception as e:
            self.log_event("error", f"Failed to load Task 3 paths: {str(e)}")
            raise
    
    def log_event(self, event_type: str, details: str, task: Optional[str] = None, trial: Optional[int] = None,
                 from_pos: Optional[str] = None, to_pos: Optional[str] = None, 
                 correctness: Optional[str] = None, direction: Optional[str] = None,
                 action: Optional[str] = None, punishment: Optional[str] = None):
        """Log an event with timestamp."""
        timestamp = time.time()
        
        # Log to main experiment log
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, details])
        
        # Log to task-specific log if applicable
        if task and task in self.task_logs:
            with open(self.task_logs[task], 'a', newline='') as f:
                writer = csv.writer(f)
                if task == "task1":
                    if event_type == "trial":
                        writer.writerow([timestamp, trial, from_pos, to_pos, correctness])
                    else:
                        writer.writerow([timestamp, trial, event_type, details])
                elif task == "task2":
                    if event_type == "trial":
                        writer.writerow([timestamp, trial, direction, action])
                    else:
                        writer.writerow([timestamp, trial, event_type, details])
                elif task == "task3":
                    if event_type == "trial":
                        writer.writerow([timestamp, trial, from_pos, to_pos, correctness, punishment])
                    else:
                        writer.writerow([timestamp, trial, event_type, details])
    
    def beep(self, freq: int = 700, duration: float = 0.15, volume: float = 0.3):
        """Play a beep sound."""
        fs = 44100
        t = np.linspace(0, duration, int(fs * duration), False)
        tone = volume * np.sin(2 * np.pi * freq * t)
        # Save original stderr
        original_stderr = sys.stderr
        # Redirect stderr to /dev/null
        sys.stderr = open('/dev/null', 'w')
        try:
            sd.play(tone, fs)
            sd.wait()
        finally:
            # Restore original stderr
            sys.stderr.close()
            sys.stderr = original_stderr
    
    def goal_beep(self):
        """Play a special beep sound for reaching the goal state."""
        # Play a single higher frequency beep
        # Save original stderr
        original_stderr = sys.stderr
        # Redirect stderr to /dev/null
        sys.stderr = open('/dev/null', 'w')
        try:
            self.beep(freq=1200, duration=0.3, volume=0.4)
        finally:
            # Restore original stderr
            sys.stderr.close()
            sys.stderr = original_stderr
    
    def wait_for_enter(self, message: str = "Press ENTER to continue..."):
        """Wait for user to press ENTER."""
        input(message)
        self.log_event("pause", message)
    
    def show_color(self, color: str, trial: Optional[int] = None, total_trials: Optional[int] = None):
        """Display a color on the screen with optional trial counter."""
        if not self.pygame_initialized:
            self._init_display()
            
        color_map = {"yellow": (255, 255, 0), "blue": (0, 0, 255), "black": (0, 0, 0)}
        if color not in color_map:
            self.log_event("error", f"Invalid color: {color}")
            return
        
        self.screen.fill(color_map[color])
        pygame.display.flip()
    
    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def closer_to_goal(self, prev: Tuple[int, int], new: Tuple[int, int]) -> bool:
        """Check if a move brings the robot closer to the goal."""
        return self.manhattan(new, TASK3_GOAL_STATE) < self.manhattan(prev, TASK3_GOAL_STATE)
    
    def demo_task1(self):
        """Demonstrate Task 1 actions."""
        self.log_event("demo_start", "Starting Task 1 demonstration")
        print("\nDemonstrating Task 1:")
        
        print("1. Moving to start position")
        self.panda.move_to_joint_position(TASK1_START_POSITION, speed_factor=CONFIG["speed_factors"]["task1"])
        time.sleep(1)
        
        print("\n2. Demonstrating yellow target movement")
        self.show_color("yellow")
        time.sleep(CONFIG["demo_duration"])
        print("Moving to yellow position...")
        self.beep()  # Beep only at start of trial
        time.sleep(0.6)
        self.panda.move_to_joint_position(TASK1_YELLOW_POSITION, speed_factor=CONFIG["speed_factors"]["task1"])
        time.sleep(1)
        self.show_color("black")
        
        print("\n3. Returning to start position")
        self.panda.move_to_joint_position(TASK1_START_POSITION, speed_factor=CONFIG["speed_factors"]["task1"])
        time.sleep(1)
        
        print("\n4. Demonstrating blue target movement")
        self.show_color("blue")
        time.sleep(CONFIG["demo_duration"])
        print("Moving to blue position...")
        self.beep()  # Beep only at start of trial
        time.sleep(0.6)
        self.panda.move_to_joint_position(TASK1_BLUE_POSITION, speed_factor=CONFIG["speed_factors"]["task1"])
        time.sleep(1)
        self.show_color("black")
        
        print("\n5. Returning to start position")
        self.panda.move_to_joint_position(TASK1_START_POSITION, speed_factor=CONFIG["speed_factors"]["task1"])
        
        # Clean up display after demo
        self._cleanup_display()
        
        self.wait_for_enter("\nDemo complete. Press ENTER to start Task 1...")
        self.log_event("demo_complete", "Task 1 demonstration completed")
    
    def show_countdown(self, seconds: int):
        """Show a countdown on the screen."""
        if not self.pygame_initialized:
            self._init_display()
            
        for i in range(seconds, 0, -1):
            self.screen.fill((0, 0, 0))  # Black background
            text = self.font.render(f"Task 1 starting in {i}...", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.screen.get_width()/2, self.screen.get_height()/2))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            time.sleep(1)
        
        # Clear screen after countdown
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
        time.sleep(0.5)
    
    def run_task1(self):
        """Run Task 1."""
        self.log_event("task_start", "Starting Task 1", task="task1")
        
        # Initialize display for Task 1
        self._init_display()
        
        # Show countdown before starting
        self.show_countdown(3)
        
        # Calculate halfway point
        halfway_point = CONFIG["num_trials"]["task1"] // 2
        
        for trial in range(1, CONFIG["num_trials"]["task1"] + 1):
            print(f"\rProgress: {trial}/{CONFIG['num_trials']['task1']} trials", end="")
            self.panda.move_to_joint_position(TASK1_START_POSITION, speed_factor=CONFIG["speed_factors"]["task1"])
            time.sleep(1)
            
            color = random.choice(["yellow", "blue"])
            self.show_color(color, trial, CONFIG["num_trials"]["task1"])
            time.sleep(1)
            
            if random.random() < CONFIG["error_probabilities"]["task1"]:
                error = True
                target_color = "blue" if color == "yellow" else "yellow"
            else:
                error = False
                target_color = color
            
            target_position = TASK1_YELLOW_POSITION if target_color == "yellow" else TASK1_BLUE_POSITION
            
            self.beep()
            time.sleep(0.6)
            self.log_event("trial", f"Trial {trial}", task="task1", trial=trial,
                          from_pos=color, to_pos=target_color, 
                          correctness="error" if error else "correct")
            self.panda.move_to_joint_position(target_position, speed_factor=CONFIG["speed_factors"]["task1"])
            time.sleep(1)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._cleanup_display()
                    sys.exit()
            
            self.show_color("black", trial, CONFIG["num_trials"]["task1"])
            
            # Check if we need a break at halfway point
            if trial == halfway_point and trial < CONFIG["num_trials"]["task1"]:
                print(f"\nTaking a break at the halfway point ({trial} trials completed)...")
                # Clean up display before break
                self._cleanup_display()
                self.wait_for_enter("Press ENTER to continue with the next set of trials...")
                # Reinitialize display after break
                self._init_display()
                # Show countdown before resuming
                self.show_countdown(3)
        
        print()  # New line after progress
        # Clean up display after Task 1
        self._cleanup_display()
        
        self.log_event("task_complete", "Completed Task 1", task="task1")
    
    def demo_task2(self):
        """Demonstrate Task 2 actions."""
        self.log_event("demo_start", "Starting Task 2 demonstration")
        print("\nDemonstrating Task 2:")
        
        # Initialize gripper for demonstration
        self._init_gripper()
        
        try:
            print("1. Moving to left position and grasping block")
            self.gripper.move(0.08, 0.1)
            self.panda.move_to_joint_position(TASK2_LEFT_POSITION, speed_factor=CONFIG["speed_factors"]["task2"])
            time.sleep(1.5)
            input("Press ENTER when ready to grasp the block...")
            self.gripper.grasp(0.03, 0.02, 0.0001)
            time.sleep(1)
            
            # Demonstrate left-to-right movement without dropping
            print("\n2. Demonstrating left-to-right movement without dropping")
            self.beep()  # Beep at start of trial
            time.sleep(0.6)
            trajectory = [TASK2_LEFT_UP_POSITION, TASK2_MIDDLE_POSITION, TASK2_RIGHT_UP_POSITION, TASK2_RIGHT_POSITION]
            self.panda.move_to_joint_position(trajectory, speed_factor=CONFIG["speed_factors"]["task2"])
            time.sleep(1)
            
            # Demonstrate right-to-left movement with dropping
            print("\n3. Demonstrating right-to-left movement with dropping")
            self.beep()  # Beep at start of trial
            time.sleep(0.6)
            trajectory = [TASK2_RIGHT_UP_POSITION, TASK2_MIDDLE_POSITION, TASK2_LEFT_UP_POSITION, TASK2_LEFT_POSITION]
            self.panda.move_to_joint_position(trajectory[0:2], speed_factor=CONFIG["speed_factors"]["task2"])
            self.gripper.move(0.08, 0.1)
            print(">> Dropped block at middle position")
            self.panda.move_to_joint_position(trajectory[2:], speed_factor=CONFIG["speed_factors"]["task2"])
            time.sleep(1)
            
            print("\n4. Waiting for block placement demonstration")
            print("In the actual task, you would need to place the block here...")
            input("Press ENTER when ready to grasp the block...")
            self.gripper.grasp(0.03, 0.02, 0.0001)
            
            self.wait_for_enter("\nDemo complete. Press ENTER to start Task 2...")
            self.log_event("demo_complete", "Task 2 demonstration completed")
            
        finally:
            # Clean up gripper after demonstration
            pass
    
    def demo_task3(self):
        """Demonstrate Task 3 actions."""
        self.log_event("demo_start", "Starting Task 3 demonstration")
        print("\nDemonstrating Task 3:")
        
        # Initialize gripper for demonstration
        self._init_gripper()
        self.gripper.move(0.03, 0.1)  # Set gripper to 0.03
        
        try:
            print("1. Moving to start position (1,1)")
            self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[(1, 1)], speed_factor=CONFIG["speed_factors"]["task3"])
            time.sleep(1)
            
            # Demonstrate a safe path
            print("\n2. Demonstrating a safe path to goal")
            safe_path = [(1, 1), (2, 1), (2,2), (2, 3), (1, 3)]
            print(f"Following path: {safe_path}")
            
            self.beep()  # Beep only at start of trial
            time.sleep(0.6)
            for i in range(1, len(safe_path)):
                prev, step = safe_path[i - 1], safe_path[i]
                speed = 0.2 if (prev, step) in [((1, 1), (2, 1)), ((2, 1), (1, 1)), ((2, 1), (3, 1)), ((3, 1), (2, 1))] else CONFIG["speed_factors"]["task3"]
                self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[step], speed_factor=speed)
                time.sleep(0.6)
                if step == TASK3_GOAL_STATE:
                    self.goal_beep()
            
            time.sleep(1)
            
            # Return to start for punishment path demonstration
            print("\n3. Returning to start position")
            self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[(1, 1)], speed_factor=CONFIG["speed_factors"]["task3"])
            time.sleep(1)
            
            # Demonstrate a punishment path
            print("\n4. Demonstrating a punishment path")
            punishment_path = [(1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 3)]
            print(f"Following path: {punishment_path}")
            
            self.beep()  # Beep only at start of trial
            time.sleep(0.6)
            for i in range(1, len(punishment_path)):
                prev, step = punishment_path[i - 1], punishment_path[i]
                speed = 0.2 if (prev, step) in [((1, 1), (2, 1)), ((2, 1), (1, 1)), ((2, 1), (3, 1)), ((3, 1), (2, 1))] else CONFIG["speed_factors"]["task3"]
                self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[step], speed_factor=speed)
                time.sleep(0.6)
                if step == TASK3_GOAL_STATE:
                    self.goal_beep()
            
            # Return to start position after demo
            print("\n5. Returning to start position")
            self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[(1, 1)], speed_factor=CONFIG["speed_factors"]["task3"])
            
            self.wait_for_enter("\nDemo complete. Press ENTER to start Task 3...")
            self.log_event("demo_complete", "Task 3 demonstration completed")
            
        finally:
            # Clean up gripper after demonstration
            pass
    
    def run_task2(self):
        """Run Task 2."""
        self.log_event("task_start", "Starting Task 2", task="task2")
        
        try:
            current_side = "left"

            self.panda.move_to_joint_position(TASK2_LEFT_POSITION, speed_factor=CONFIG["speed_factors"]["task2"])
            time.sleep(1.5)
            action = False
            
            # Calculate halfway point
            halfway_point = CONFIG["num_trials"]["task2"] // 2
            
            trial = 1
            while trial <= CONFIG["num_trials"]["task2"]:
                try:
                    if action == "drop":
                        input("Press ENTER when ready to grasp the block...")
                        self.gripper.grasp(0.03, 0.02, 0.0001)

                    print(f"\rProgress: {trial}/{CONFIG['num_trials']['task2']} trials", end="")
                    self.beep(700, 0.15, 0.3)
                    time.sleep(0.6)
                    
                    if current_side == "left":
                        trajectory = [TASK2_LEFT_UP_POSITION, TASK2_MIDDLE_POSITION, TASK2_RIGHT_UP_POSITION, TASK2_RIGHT_POSITION]
                        direction = "left_to_right"
                        end_position = TASK2_RIGHT_POSITION
                    else:
                        trajectory = [TASK2_RIGHT_UP_POSITION, TASK2_MIDDLE_POSITION, TASK2_LEFT_UP_POSITION, TASK2_LEFT_POSITION]
                        direction = "right_to_left"
                        end_position = TASK2_LEFT_POSITION
                    
                    drop = random.random() < CONFIG["error_probabilities"]["task2"]
                    action = "drop" if drop else "correct"
                    
                    if drop:
                        self.panda.move_to_joint_position(trajectory[0:2], speed_factor=CONFIG["speed_factors"]["task2"])
                        self.log_event("trial", f"Trial {trial}", task="task2", trial=trial,
                                  direction=direction, action=action)
                        self.gripper.move(0.08, 0.1)
                        self.panda.move_to_joint_position(trajectory[2:], speed_factor=CONFIG["speed_factors"]["task2"])
                    else:
                        self.panda.move_to_joint_position(trajectory[0:2], speed_factor=CONFIG["speed_factors"]["task2"])
                        self.log_event("trial", f"Trial {trial}", task="task2", trial=trial,
                                  direction=direction, action=action)
                        self.panda.move_to_joint_position(trajectory[2:], speed_factor=CONFIG["speed_factors"]["task2"])
                    
                    current_side = "right" if current_side == "left" else "left"
                    
                    # Check if we need a break at halfway point
                    if trial == halfway_point and trial < CONFIG["num_trials"]["task2"]:
                        print(f"\nTaking a break at the halfway point ({trial} trials completed)...")
                        self.wait_for_enter("Press ENTER to continue with the next set of trials...")
                    
                    trial += 1
                    time.sleep(1)
                    
                except Exception as e:
                    self.log_event("error", f"Error during Trial {trial}: {str(e)}", task="task2")
                    print(f"\nError during Trial {trial}: {str(e)}")
                    print("Would you like to:")
                    print("1. Retry the current trial")
                    print("2. Exit the task")
                    
                    choice = input("Enter your choice (1-3): ").strip()
                    if choice == "1":
                        print("\nRetrying the current trial...")
                        self.gripper.move(0.08, 0.1)
                        continue
                    else:
                        print("\nExiting task...")
                        raise
                        
        except Exception as e:
            self.log_event("error", f"Error during Task 2: {str(e)}", task="task2")
            raise
        finally:
            # Clean up gripper after Task 2
            self._cleanup_gripper()
        
        print()  # New line after progress
        self.log_event("task_complete", "Completed Task 2", task="task2")
    
    def run_task3(self):
        """Run Task 3."""
        self.log_event("task_start", "Starting Task 3", task="task3")
        
        try:
            # Calculate halfway point
            halfway_point = CONFIG["num_trials"]["task3"] // 2
            
            for trial in range(1, CONFIG["num_trials"]["task3"] + 1):
                print(f"\rProgress: {trial}/{CONFIG['num_trials']['task3']} trials", end="")
                self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[(1, 1)], speed_factor=CONFIG["speed_factors"]["task3"])
                time.sleep(1)
                self.beep()
                time.sleep(0.6)
                
                path = random.choice(self.safe_paths if random.random() < 0.7 else self.punishment_paths)
                
                for i in range(1, len(path)):
                    prev, step = path[i - 1], path[i]
                    speed = 0.2 if (prev, step) in [((1, 1), (2, 1)), ((2, 1), (1, 1)), ((2, 1), (3, 1)), ((3, 1), (2, 1))] else CONFIG["speed_factors"]["task3"]
                    correct = self.closer_to_goal(prev, step)
                    self.log_event("trial", f"Trial {trial}", task="task3", trial=trial,
                                  from_pos=str(prev), to_pos=str(step),
                                  correctness="correct" if correct else "incorrect",
                                  punishment="punishment" if step == TASK3_RESET_STATE else "no punishment")
                    self.panda.move_to_joint_position(TASK3_GRID_POSITIONS[step], speed_factor=speed)
                    time.sleep(1.2)
                    if step == TASK3_GOAL_STATE:
                        self.goal_beep()
                
                time.sleep(2)
                
                # Check if we need a break at halfway point
                if trial == halfway_point and trial < CONFIG["num_trials"]["task3"]:
                    print(f"\nTaking a break at the halfway point ({trial} trials completed)...")
                    self.wait_for_enter("Press ENTER to continue with the next set of trials...")
            
            print()  # New line after progress
            self.log_event("task_complete", "Completed Task 3", task="task3")
            
        finally:
            # Clean up gripper after Task 3
            self._cleanup_gripper()
    
    def run_task(self, task: str) -> bool:
        """Run a single task with error handling and logging."""
        try:
            # Run demonstration
            if task == "task1":
                self.demo_task1()
                self.run_task1()
            elif task == "task2":
                self.demo_task2()
                self.run_task2()
            elif task == "task3":
                self.demo_task3()
                self.run_task3()
            
            return True
            
        except Exception as e:
            self.log_event("error", f"Error in {task}: {str(e)}", task=task)
            print(f"\nError in {task}: {str(e)}")
            print("Would you like to:")
            print("1. Retry the current task")
            print("2. Skip to the next task")
            print("3. Exit the experiment")
            
            choice = input("Enter your choice (1-3): ").strip()
            if choice == "1":
                print("\nRetrying the current task...")
                return self.run_task(task)
            elif choice == "2":
                print("\nSkipping to the next task...")
                return False
            else:
                print("\nExiting experiment...")
                raise
    
    def run_experiment(self):
        """Run the full experiment."""
        try:
            # Log experiment start
            self.log_event("experiment_start", f"Participant {self.participant_id}")
            print(f"\nWelcome to the experiment, Participant {self.participant_id}!")
            print("Tasks will be presented in the following order:")
            for i, task in enumerate(self.tasks, 1):
                print(f"{i}. {task.upper()}")
            self.wait_for_enter("\nPress ENTER to begin the experiment...")
            
            # Run each task
            for i, task in enumerate(self.tasks):
                next_task = self.tasks[i + 1] if i + 1 < len(self.tasks) else None
                print(f"\n{'='*20} Starting {task.upper()} {'='*20}")
                try:
                    success = self.run_task(task)
                    if not success:
                        print(f"\nWarning: {task} was skipped due to errors.")
                except Exception as e:
                    print(f"\nError in {task}: {str(e)}")
                    print("Would you like to:")
                    print("1. Retry the current task")
                    print("2. Skip to the next task")
                    print("3. Exit the experiment")
                    
                    choice = input("Enter your choice (1-3): ").strip()
                    if choice == "1":
                        print("\nRetrying the current task...")
                        continue
                    elif choice == "2":
                        print("\nSkipping to the next task...")
                        continue
                    else:
                        print("\nExiting experiment...")
                        raise
                
                if next_task:
                    self.wait_for_enter(f"\nPress ENTER to continue to {next_task.upper()}...")
                else:
                    self.wait_for_enter("\nPress ENTER to finish the experiment...")
            
            # Log experiment completion
            self.log_event("experiment_complete", "Experiment finished successfully")
            print("\nExperiment completed! Thank you for participating.")
            
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user.")
            self.log_event("experiment_interrupted", "Experiment interrupted by user")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            self.log_event("experiment_error", f"Experiment error: {str(e)}")
        finally:
            # Cleanup
            self._cleanup_display()
            self._cleanup_gripper()
            self._cleanup_robot()

def main():
    try:
        # Get participant ID
        participant_id = input("Please enter participant ID: ").strip()
        while not participant_id:
            participant_id = input("Participant ID cannot be empty. Please enter participant ID: ").strip()
        
        # Create and run experiment
        experiment = ExperimentManager(participant_id)
        experiment.run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()