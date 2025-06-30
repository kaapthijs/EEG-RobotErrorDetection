import sys
import time
import panda_py
from panda_py import controllers

def save_positions(positions_save, filename):
  """Saves the joint positions to a file."""
  with open(filename, 'w') as file:
    for pos_save in positions_save:
      file.write(f"{pos_save}\n")
  print(f"Positions saved to {filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError(f'Usage: python {sys.argv[0]} <robot-hostname>')

    panda = panda_py.Panda('172.16.0.2')
    gripper = panda_py.libfranka.Gripper('172.16.0.2')

    panda.move_to_start()
    gripper.move(0.08, 0.1)
    #gripper.grasp(0.04, 0.02, 0.01)

    # Ask user for the number of joint positions to teach
    num_positions = 3






    positions = []

    print('Please teach the joint positions to the robot.')
    panda.teaching_mode(True)

    for i in range(num_positions):
        print(f'Move the robot into pose {i+1} and press enter to continue.')
        input()
        positions.append(panda.q)

    panda.teaching_mode(False)

    save_positions(positions, 'ROBOT/saved_positions.txt')

    # Replay the positions
    print("Replaying the saved positions...")
    for pos in positions:
        panda.move_to_joint_position(pos)
        time.sleep(2)  # Allow some time for the robot to move to each position

    print("Replay completed.")
