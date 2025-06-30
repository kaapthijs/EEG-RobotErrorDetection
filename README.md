# EEG-RobotErrorDetection

This repository contains the code and data for the EEG-Robot Error Detection project, focusing on the integration of robot experiment execution, data collection, and subsequent classification based on electroencephalography (EEG) data.

## Project Structure

```
.
├── DATA/
│   └── (Contains pre-merged data used for classification)
├── ROBOT/
│   ├── Bucket generation/
│   │   ├── bucket_gen.py
│   │   └── paths_output.txt
│   ├── experiment_data/
│   ├── Task1/ (Reaching)
│   │   ├── robot_experiment_log.csv
│   │   ├── run_exp.py
│   │   └── saved_positions.txt
│   ├── Task2/ (Block)
│   │   ├── robot_experiment_log.csv
│   │   ├── run_exp.py
│   │   └── saved_positions.txt
│   ├── Task3/ (Grid)
│   │   ├── participants/
│   │   ├── buckets.txt
│   │   └── run_exp.py
│   ├── run_full_experiment.py
│   ├── saved_positions.txt
│   ├── send_continious_trigger.py
│   └── simulate.py
├── README.md
├── requirements.txt
└── run_classification.py
```

## Description of Files and Directories

* **`DATA/`**: This directory holds the already merged data used for the classification tasks.

* **`ROBOT/`**: This directory contains all the Python scripts and related files necessary for executing the robot experiments.

  * **`Bucket generation/`**: Scripts and data related to generating experiment paths (buckets) for Task 3 (Grid).

  * **`experiment_data/`**: Output folder for data.

  * **`Task1/`**, **`Task2/`**, **`Task3/`**: Each of these subdirectories corresponds to a specific experimental task. (Reaching, Block, Grid, respectively)

    * `robot_experiment_log.csv`: Logs from the robot experiment for the respective task.

    * `run_exp.py`: Script to run an individual task experiment.

    * `saved_positions.txt`: Files containing the robot's saved position coordinates for each task.

    * `buckets.txt`: Files containing the robot's saved position coordinates for Grid task.

  * **`run_full_experiment.py`** (in `ROBOT/Task3/`): The main script to run all predefined robot experiments sequentially.

  * **`send_continious_trigger.py`** (in `ROBOT/Task3/`): A script designed to send a continuous timestamp trigger. This is crucial for merging data from the robot's movements with data acquired from the Unicorn EEG software.

  * **`simulate.py`**: This script allows you to determine and test the robot's position coordinates. These coordinates can then be used to populate the `saved_positions.txt` files for each task.

* **`run_classification.py`**: This script takes the data from the `DATA/` directory and performs the classification analysis.

## How to Use

### Running Experiments

1. **Determine Robot Positions**: Use `simulate.py` to find and verify the necessary robot position coordinates for each task.

2. **Configure Saved Positions**: Place the determined coordinates into the respective `saved_positions.txt` files within the `ROBOT/Task1/`, `ROBOT/Task2/`, and `ROBOT/Task3/` directories.

3. **Run Individual Tasks (Optional)**: To test individual tasks, you can run `ROBOT/TaskX/run_exp.py` for `Task1`, `Task2`, or `Task3`.

4. **Run Full Experiment**: To execute the entire set of experiments, run `ROBOT/Task3/run_full_experiment.py`.

### Data Synchronization

* **`send_continious_trigger.py`**: Ensure this script is running while running the full experiment to provide continuous timestamp triggers to the Unicorn EEG software. This is essential for accurate synchronization and merging of robot and EEG data.

### Classification

* **`run_classification.py`**: Execute this script to perform the classification analysis.

## Requirements

The `requirements` file (or section) contains all the necessary Python libraries that need to be installed before running any of the scripts. Please refer to this file for a complete list of dependencies.
