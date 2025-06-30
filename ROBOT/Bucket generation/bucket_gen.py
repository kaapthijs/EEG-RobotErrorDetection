import random
import json

start_state = (1, 1)
goal_state = (1, 3)
punishment_state = (3, 1)


def get_adjacent_cells(grid_pos):
    x, y = grid_pos
    adjacent_cells = []

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_pos = (x + dx, y + dy)
        if 1 <= new_pos[0] <= 3 and 1 <= new_pos[1] <= 3:
            adjacent_cells.append(new_pos)

    return adjacent_cells

def generate_path_with_error(include_punishment=False, error_ratio=0.2):
    current_position = start_state
    path = [current_position]
    punishment_crossed = False

    while current_position != goal_state:
        adjacent_cells = get_adjacent_cells(current_position)
        correct_move = random.random() < (1 - error_ratio)

        if correct_move:
            correct_moves = [pos for pos in adjacent_cells if pos[1] > current_position[1] or pos[0] < current_position[0]]
            intended_position = random.choice(correct_moves)
        else:
            wrong_moves = [pos for pos in adjacent_cells if pos[1] < current_position[1] or pos[0] > current_position[0]]
            if wrong_moves:
                intended_position = random.choice(wrong_moves)
            else:
                correct_moves = [pos for pos in adjacent_cells if pos[1] > current_position[1] or pos[0] < current_position[0]]
                intended_position = random.choice(correct_moves)

        path.append(intended_position)
        current_position = intended_position

        if current_position == punishment_state and include_punishment:
            punishment_crossed = True
        if len(path) >= 11:
            return None

    if include_punishment and punishment_crossed:
        return path
    elif not include_punishment and punishment_state not in path:
        return path
    else:
        return None

safe_paths = []
punishment_paths = []
seen_paths = set()

while len(safe_paths) < 15:
    path = generate_path_with_error(include_punishment=False, error_ratio=0.2)
    if path:
        path_tuple = tuple(path)
        if path_tuple not in seen_paths:
            safe_paths.append(path)
            seen_paths.add(path_tuple)

while len(punishment_paths) < 15:
    path = generate_path_with_error(include_punishment=True, error_ratio=0.2)
    if path:
        path_tuple = tuple(path)
        if path_tuple not in seen_paths:
            punishment_paths.append(path)
            seen_paths.add(path_tuple)


all_paths = {"safe_paths": safe_paths, "punishment_paths": punishment_paths}

output_file = 'paths_output.txt'
with open(output_file, 'w') as file:
    for category, paths in all_paths.items():
        file.write(f"{category}:\n")
        for path in paths:
            file.write(f"{path}\n")

print(f"Paths saved to {output_file}")
