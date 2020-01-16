import numpy as np
from uofgsocsai import LochLomondEnv
import sys


def main(problemID, mapID):
    problem = int(problemID)
    reward_hole = -1.0
    stochastic = True
    episodes = 1000
    iterPerEpisode = 2000
    mapBase = mapID
    successes = 0  # records the number of successes
    stats = {"episodes": {}}
    totalReward = 0  # reward per episode

    # set up the environment
    env = LochLomondEnv(problem_id=problem,
                        is_stochastic=stochastic,
                        map_name_base=mapBase,
                        reward_hole=reward_hole)

    np.random.seed(12)

    for episode in range(episodes):  # iterate over episodes
        print("___________________________________")
        print("EPISODE: " + str(episode))
        observation = env.reset(
        )  # reset the state of the env to the starting state

        reward = 0

        for step in range(iterPerEpisode):
            action = env.action_space.sample(
            )  # your agent goes here (the current agent takes random actions)
            observation, reward, done, info = env.step(
                action)  # observe what happends when you take the action
            # Check if we are done and monitor rewards etc...

            if done:
                stats["episodes"][episode] = {"steps": step, "reward": reward}
                totalReward += reward
                break

    successRate = ((successes / episodes) * 100)
    print("Finished")
    print("Success Rate: " + str(successRate) + "%")
    print("Total Reward: " + str(totalReward))
    # log stats
    stats["successrate"] = successRate
    stats["totalreward"] = totalReward
    return stats


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
