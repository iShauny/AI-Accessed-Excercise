import numpy as np
from uofgsocsai import LochLomondEnv
import sys
import time
# import matplotlib.pyplot as plt
# from matplotlib import lines
AIMA_TOOLBOX_ROOT = "./amia"
sys.path.append(AIMA_TOOLBOX_ROOT)
import search


def my_best_first_graph_search_for_vis(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    # we use these two variables at the time of visualisations
    iterations = 0

    f = search.memoize(f, 'f')
    node = search.Node(problem.initial)

    iterations += 1

    if problem.goal_test(node.state):
        iterations += 1
        return (iterations, node)

    frontier = search.PriorityQueue('min', f)
    frontier.append(node)

    iterations += 1

    explored = set()
    while frontier:
        node = frontier.pop()

        iterations += 1
        if problem.goal_test(node.state):
            iterations += 1
            return (iterations, node)

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                iterations += 1
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    iterations += 1

        iterations += 1
    return None


def my_astar_search_graph(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = search.memoize(h or problem.h, 'h')
    iterations, node = my_best_first_graph_search_for_vis(
        problem, lambda n: n.path_cost + h(n))

    return (iterations, node)


def env2statespace(env):
    """
    This simple parser demonstrates how you can extract the state space from
    the Open AI env

    We *assume* full observability, i.e., we can directly ignore Hole states.
    Alternatively, we could place a very high step cost to a Hole state or
    use a directed representation (i.e., you can go to a Hole state but
    never return). Feel free to experiment with both if time permits.

    Input:
        env: an Open AI Env follwing the std in the FrozenLake-v0 env

    Output:
        state_space_locations : a dict with the available states
        state_space_actions   : a dict of dict with available actions in each
                                state
        state_start_id        : the start state
        state_goal_id         : the goal state

        These objects are enough to define a Graph problem using the AIMA
        toolbox, e.g., using UndirectedGraph, GraphProblem and astar_search
        (as in AI (H) Lab 3)

    Notice: the implementation is very explicit to demonstarte all the steps
    (it could be made more elegant!)

    bjorn.jensen@glasgow.ac.uk

    """
    state_space_locations = {}  # create a dict
    for i in range(env.desc.shape[0]):
        for j in range(env.desc.shape[1]):
            if not (b'H' in env.desc[i, j]):
                state_id = "S_" + str(int(i)) + "_" + str(int(j))
                state_space_locations[state_id] = (int(i), int(j))
                if env.desc[i, j] == b'S':
                    state_initial_id = state_id
                elif env.desc[i, j] == b'G':
                    state_goal_id = state_id

                # -- Generate state / action list --#
                # First define the set of actions in the defined coordinate
                # system
                actions = {
                    "west": [-1, 0],
                    "east": [+1, 0],
                    "north": [0, +1],
                    "south": [0, -1]
                }
                state_space_actions = {}
                for state_id in state_space_locations:
                    possible_states = {}
                    for action in actions:
                        # -- Check if a specific action is possible --#
                        delta = actions.get(action)
                        state_loc = state_space_locations.get(state_id)
                        state_loc_post_action = [
                            state_loc[0] + delta[0], state_loc[1] + delta[1]
                        ]

                        # -- Check if the new possible state is in the
                        # state_space, i.e., is accessible --#
                        state_id_post_action = "S_" + str(
                            state_loc_post_action[0]) + "_" + str(
                                state_loc_post_action[1])
                        if state_space_locations.get(
                                state_id_post_action) is not None:
                            possible_states[state_id_post_action] = 1

                    # -- Add the possible actions for this state to the global
                    # dict --#
                    state_space_actions[state_id] = possible_states

    return state_space_locations, state_space_actions, state_initial_id, \
        state_goal_id


def main(problemID, mapID):
    problem = int(problemID)
    reward_hole = -1.0
    stochastic = False
    episodes = 100
    mapBase = mapID
    stats = {}

    # start from a known seed
    np.random.seed(12)

    # set up the environment
    env = LochLomondEnv(problem_id=problem,
                        is_stochastic=stochastic,
                        map_name_base=mapBase,
                        reward_hole=reward_hole)

    state_space_locations, state_space_actions, state_initial_id, \
        state_goal_id = env2statespace(env)

    # Insert the solution here to find and output the solution using A-star
    # define the states and actions in a table
    maze_map = search.UndirectedGraph(state_space_actions)
    maze_map.locations = state_space_locations

    maze_problem = search.GraphProblem(state_initial_id, state_goal_id,
                                       maze_map)

    for episode in range(episodes):  # iterate over episodes
        env.reset()  # reset the state of the env to the starting state
        iterations, node = my_astar_search_graph(problem=maze_problem, h=None)
        # -- Trace the solution --#
        solution_path = [node]
        cnode = node.parent
        solution_path.append(cnode)
        while cnode.state != state_initial_id:
            cnode = cnode.parent
            solution_path.append(cnode)

        print("----------------------------------------")
        print("Identified goal state:" + str(solution_path[0]))
        print("Solution trace:" + str(solution_path))
        print("Iterations:" + str(iterations))
        print("----------------------------------------")
        # log stats
    stats["solutiontrace"] = str(solution_path)
    stats["numberofiterations"] = str(iterations)

    return stats


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
