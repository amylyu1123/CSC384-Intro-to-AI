#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
import math  # for infinity

from search import *  # for search engines
from sokoban import sokoban_goal_state, SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems


# SOKOBAN HEURISTICS
def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    # EXPLAIN YOUR HEURISTIC IN THE COMMENTS. Please leave this function (and your explanation) at the top of your solution file, to facilitate marking.

    # Firstly, call available_storage_box to get all storages that have not been
    # occupied and all boxes that have not gotten to any storage because we want
    # to match each box to one storage without repeatedly assigned to one storage.
    # Secondly, for each box that is not at the storage,

    # (a) check if it is at a location that causes dead state (cannot be solved)
    # which happens when under any of the following situations:
    #   (1) at each corner of the wall
    #   (2) along the wall and at the corner of an obstacle beside the wall
    #   (3) at each corner of obstacles
    #   (4) two boxes along the wall
    #   (5) two boxes along the obstacles
    # If this box is at a location that causes dead state, penalize it by return
    # a large distance (infinity)

    # (b) call get_distance to get the alternative heuristic distance for box,
    # and the storage this box would be assigned to.
    # add the distance to result and remove the storage index from available storages
    # to avoid duplicate assigned later

    # after finish iterating all available boxes, return the result

    result = 0
    storages, boxes = available_storage_box(state)

    for box in boxes:
        if dead_state(box, state) == float("inf"):
            return float("inf")
        curr_distance, storage_index = get_distance(box, state, storages)
        if curr_distance == float("inf"):
            return curr_distance
        else:
            result += curr_distance
            # remove the storage this box would be assigned to from index of available storages
            storages.remove(storage_index)

    return result

# return lists of all boxes that are not at any storage and all storages which do not
# have any box on it.
def available_storage_box(state):
    boxes = []
    storages = []
    for box in state.boxes:
        if box not in state.storage:
            boxes.append(box)
    for storage in state.storage:
        if storage not in state.boxes:
            storages.append(storage)
    return storages, boxes

# Return float("inf") if box(not at any storage) is at any of the below situation:
# (1) at each corner of the wall
# (2) along the wall and at the corner of an obstacle beside the wall
# (3) at each corner of obstacles
# (4) two boxes along the wall
# (5) two boxes along the obstacles
# Otherwise, return None.
def dead_state(box, state):
    (x, y) = box
    height = state.height
    width = state.width
    obstacles = state.obstacles
    boxes = state.boxes

    # at least one side along the wall
    if x == 0 or x == width - 1:
        # corner cases of wall
        if y == 0 or y == height - 1:
            return float("inf")
        # along the wall, at corner of obstacle
        if (x, y + 1) in obstacles or (x, y - 1) in obstacles:
            return float("inf")
        # two boxes
        if (x, y + 1) in boxes or (x, y - 1) in boxes:
            return float("inf")

    # along top wall or bottom without corner
    if y == 0 or y == height - 1:
        if (x - 1, y) in obstacles or (x + 1, y) in obstacles:
            return float("inf")
        # two boxes
        if (x - 1, y) in boxes or (x + 1, y) in boxes:
            return float("inf")

    # --------------corner cases of obstacles + two boxes---------------------
    # left obstacle
    if (x - 1, y) in obstacles:
        # top left and bottom left
        if (x, y + 1) in obstacles or (x, y - 1) in obstacles:
            return float("inf")
        if (x, y - 1) in boxes and (x - 1, y - 1) in obstacles:
            return float("inf")
        if (x, y + 1) in boxes and (x - 1, y + 1) in obstacles:
            return float("inf")

    # right obstacle
    if (x + 1, y) in obstacles:
        # top right and bottom right
        if (x, y + 1) in obstacles or (x, y - 1) in obstacles:
            return float("inf")
        if (x, y - 1) in boxes and (x + 1, y - 1) in obstacles:
            return float("inf")
        if (x, y + 1) in boxes and (x + 1, y + 1) in obstacles:
            return float("inf")

    # bottom obstacle
    if (x, y - 1) in obstacles:
        if (x - 1, y) in boxes and (x - 1, y - 1) in obstacles:
            return float("inf")
        if (x + 1, y) in boxes and (x + 1, y - 1) in obstacles:
            return float("inf")

    # top obstacle
    if (x, y + 1) in obstacles:
        if (x - 1, y) in boxes and (x - 1, y + 1) in obstacles:
            return float("inf")
        if (x + 1, y) in boxes and (x + 1, y + 1) in obstacles:
            return float("inf")
    return None

# Return the alternative heuristic distance for box, and the storage this box
# would be assigned to.
# If there is no storage that this box could get to, return float("inf"), None.
# Parameter: box is not at any storage, storages are all available storages with no box
# assigned to it
def get_distance(box, state, storages):
    (x, y) = box
    height = state.height
    width = state.width

    # the index of the storage this box should be matched to
    storage_index = None

    # initialize the nearest distance between box and the nearest storage
    distance = float("inf")

    # if box is along the wall(not at any corner)
    # since it can only be pushing along the wall, we would only consider
    # storage along the same wall
    # find the nearest one and return the Manhattan distance and storage index
    if (x == 0 or x == width - 1) or (y == 0 or y == height - 1):
        # left or right wall
        if x == 0 or x == width - 1:
            for storage in storages:
                # when this storage is at the same column of the box
                if storage[0] == x:
                    # if there is no obstacle between box and storage
                    if not check_wall_obstacles(box, storage, state):
                        curr = abs(y - storage[1])
                        if curr < distance:
                            distance = curr
                            storage_index = storage
        # top or bottom wall
        else:
            for storage in storages:
                # when this storage is at the same row of the box
                if storage[1] == y:
                    # if there is no obstacle between box and storage
                    if not check_wall_obstacles(box, storage, state):
                        curr = abs(x - storage[0])
                        if curr < distance:
                            distance = curr
                            storage_index = storage

        # return the nearest Manhattan distance and storage index for this box.
        # return float("inf"), None if there is no storage the box could get to.
        return distance, storage_index

    # for other boxes, consider Manhattan distance between its nearest robot and
    # itself and the Manhattan distance between itself and nearest storage that
    # has not been assigned
    else:
        # find the nearest storage that has not been assigned for this box and
        # record the Manhattan distance and storage index
        for storage in storages:
            (x_1, y_1) = storage
            curr = abs(x - x_1) + abs(y - y_1)
            if curr < distance:
                distance = curr
                storage_index = storage
        # add the number of obstacles that may encounter between box and the
        # storage to distance
        distance += check_obstacles(box, storage_index, state)

    r_to_b = None
    robot_index = None
    # find the nearest robot for this box and
    # record the Manhattan distance and robot index
    for robot in state.robots:
        (x_r, y_r) = robot
        curr_distance = abs(x - x_r) + abs(y - y_r)
        if r_to_b is None or curr_distance < r_to_b:
            r_to_b = curr_distance
            robot_index = robot
    # add the Manhattan distance to distance
    distance += r_to_b
    # add the number of obstacles that may encounter between robot and the box
    distance += check_obstacles(robot_index, box, state)

    # # return the nearest Manhattan distance and storage index for this box.
    # return float("inf"), None, if there is no storage the box could be assigned to.
    return distance, storage_index

# Return True if there is an obstacle between box and storage, where storage
# and box are either on the same row or column along any wall. box is not
# at any storage
def check_wall_obstacles(box, storage, state):
    (x, y) = box
    (s_x, s_y) = storage
    for obstacle in state.obstacles:
        # box and storage at the same column
        if x == s_x and x == obstacle[0]:
            if y < obstacle[1] < s_y or y > obstacle[1] > s_y:
                return True
        # box and storage at the same row
        elif y == s_y and y == obstacle[1]:
            if x < obstacle[0] < s_x or x > obstacle[0] > s_x:
                return True
    return False

# check the number of non-passable things(obstacles, robots, boxes)
# that may encounter if moving from source to destination
# there are two cases for source and destination: from robot to box, and from box to storage
# when a robot is moving toward a box, if there is another robot or box or any
# obstacle inside the space of possible location between source and destination,
# we add 1 to indicate the existence of any of the object. Same for the other case.
def check_obstacles(source, destination, state):
    (x, y) = source
    (x_1, y_1) = destination
    count = 0

    left = min(x, x_1)
    right = max(x, x_1)
    top = max(y, y_1)
    bottom = min(y, y_1)

    for obstacle in state.obstacles:
        if left <= obstacle[0] <= right and bottom <= obstacle[1] <= top:
            count += 1
    for other_box in state.boxes:
        if other_box[0] != x and other_box[1] != y and \
                other_box[0] != x_1 and other_box[1] != y_1:
            if left <= other_box[0] <= right and bottom <= other_box[1] <= top:
                count += 1
    for robot in state.robots:
        if robot[0] != x and robot[1] != y and \
                robot[0] != x_1 and robot[1] != y_1:
            if left <= robot[0] <= right and bottom <= robot[1] <= top:
                count += 1
    return count

def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def heur_manhattan_distance(state):
    # IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.

    result = 0
    for box in state.boxes:
        if box not in state.storage:
            (x_0, y_0) = box
            nearest = None
            for storage in state.storage:
                (x_1, y_1) = storage
                curr_distance = abs(x_0 - x_1) + abs(y_0 - y_1)
                if nearest is None or curr_distance < nearest:
                    nearest = curr_distance
            result += nearest
    return result

def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
    return sN.gval + weight * sN.hval

# helper of weighted_astar and iterative_astar
def _astar_helper(initial_state, heur_fn, weight, timebound, costbound=None):
    se = SearchEngine('custom', 'full')
    se.init_search(initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn,
                   fval_function=(lambda sN: fval_function(sN, weight)))
    return se.search(timebound, costbound)

# SEARCH ALGORITHMS
def weighted_astar(initial_state, heur_fn, weight, timebound):
    # IMPLEMENT
    '''Provides an implementation of weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of weighted astar algorithm'''

    return _astar_helper(initial_state, heur_fn, weight, timebound)

def iterative_astar(initial_state, heur_fn, weight=1, timebound=5):  # uses f(n), see how autograder initializes a search line 88
    # IMPLEMENT
    '''Provides an implementation of realtime a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of iterative astar algorithm'''

    final, stats = None, None
    costbound = None

    while timebound > 0:
        time = os.times()[0]  # record the start time for this iteration

        curr_final, curr_stats = _astar_helper(initial_state, heur_fn, weight,
                                               timebound, costbound)
        if final is None:  # first iteration
            final, stats = curr_final, curr_stats
        if curr_final:
            curr_cost = (float("inf"), float("inf"), curr_final.gval)
            # if this result has been the optimal
            if costbound is None or curr_cost[2] < costbound[2]:
                costbound = curr_cost
                final, stats = curr_final, curr_stats
        weight = weight * 0.5

        difference = os.times()[0] - time  # record time taken for search for this iteration
        timebound -= difference  # decreases the timebound by time time taken for this search

    return final, stats

def iterative_gbfs(initial_state, heur_fn, timebound=5):  # only use h(n)
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of iterative gbfs algorithm'''
    final, stats = None, None
    costbound = None

    se = SearchEngine('best_first', 'full')
    se.init_search(initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn)

    while timebound > 0:
        time = os.times()[0]
        curr_final, curr_stats = se.search(timebound, costbound)
        if final is None:
            final, stats = curr_final, curr_stats
        if curr_final:
            curr_cost = (curr_final.gval, float("inf"), float("inf"))
            if costbound is None or curr_cost[0] < costbound[0]:
                costbound = curr_cost
                final, stats = curr_final, curr_stats

        difference = os.times()[0] - time
        timebound -= difference

    return final, stats




