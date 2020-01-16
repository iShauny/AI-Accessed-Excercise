import numpy as np
from uofgsocsai import LochLomondEnv
import sys
import random
AIMA_TOOLBOX_ROOT = "./amia"
sys.path.append(AIMA_TOOLBOX_ROOT)


def generate_q(environment, episodes, maxSteps):
    # https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
    # https://github.com/magnarch/AI-Exercise/blob/master/run_rl.py
    learningRate = 0.1  # learning rate
    gamma = 0.99  # discount factor
    epsilon = 1  # exploration rate, do we explore or stick to what we know?
    # first action is chosen at random as we have no knowledge
    # start from a known seed

    actionSize = environment.action_space.n
    stateSize = environment.observation_space.n

    Q = np.zeros((stateSize, actionSize))

    print("___________________________________________")
    print("TRAINING IN PROGRESS (THIS MAY TAKE SOME TIME)...")

    for episode in range(episodes):
        state = environment.reset()
        done = False
        if episode % 100 == 0:
            # reduces likelihood of bad policy
            epsilon = (1 - (episode / episodes))

        for step in range(maxSteps):

            # choose an action
            rand = random.uniform(0, 1)

            # if this number is greater than our exploration rate
            # then we exploit this, otherwise we take a random
            # action
            if (rand > epsilon):
                chosenAction = np.argmax(Q[state, :])
            else:
                chosenAction = environment.action_space.sample()

            # carry out the action
            nextState, reward, done, info = environment.step(chosenAction)

            # update our q table using the formula below
            Q[state, chosenAction] = Q[state, chosenAction] + learningRate * (
                reward +
                gamma * np.max(Q[nextState, :] - Q[state, chosenAction]))

            # update the state
            state = nextState

            # reduce the learning rate
            learningRate = 1 / (1 + step)
            if done:
                if reward != 0:
                    break

            epsilon = 0.015  # use our knowledge and explore less
    return Q


def main(problemID, mapID):
    problem = int(problemID)
    rewardHole = -0.02
    stochastic = True
    trainingEpisodes = 35000
    episodes = 1000
    iterPerEpisode = 2000
    mapBase = mapID
    np.random.seed(12)
    successes = 0  # records the number of successes
    totalReward = 0
    stats = {"episodes": {}}

    # set up the environment
    env = LochLomondEnv(problem_id=problem,
                        is_stochastic=stochastic,
                        map_name_base=mapBase,
                        reward_hole=rewardHole)

    qTable = generate_q(env, trainingEpisodes, iterPerEpisode)

    print("___________________________________")
    print("Training Finished")
    print("Attempting to find solution...")

    for episode in range(episodes):
        # initial params
        state = env.reset()
        step = 0
        done = False
        reward = 0
        for step in range(iterPerEpisode):
            action = np.argmax(qTable[state, :])  # take the best action
            nextState, reward, done, info = env.step(action)
            if done:
                stats["episodes"][episode] = {"steps": step, "reward": reward}
                if (reward == 1.0):
                    successes += 1
                totalReward += reward
                break
        state = nextState

    successRate = ((successes / episodes) * 100)
    print("___________________________________")
    print("Finished")
    print("Success Rate: " + str(successRate) + "%")
    print("Total Reward: " + str(totalReward))
    # log stats
    stats["successrate"] = successRate
    stats["totalreward"] = totalReward
    stats["qtable"] = qTable
    return stats, qTable


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
