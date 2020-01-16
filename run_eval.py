import run_simple
import run_rl
import run_random
import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

simpleStats = []
rlStats = []
randomStats = []


def process_stats(mapID, numberOfProblems):
    randomTable = PrettyTable()
    randomTable.field_names = ["Problem", "Success Rate", "Total Reward"]
    randomSummary = {}  # for returning a summary of the agent
    print("___________________________________________________________")
    print("Random Agent Running...")
    time.sleep(3)
    # handle random
    for problem in range(numberOfProblems):
        randomSuccessRates = []
        totalRandomRewards = 0
        print("___________________________________________________________")
        print("Problem " + str(problem) + "...")
        time.sleep(2)
        statsRandom = run_random.main(problem, mapID)
        randomStats.append(statsRandom)
        # for summary
        randomSuccessRate = statsRandom["successrate"]
        randomReward = statsRandom["totalreward"]
        # for calculating averages
        randomSuccessRates.append(statsRandom["successrate"])
        totalRandomRewards += statsRandom["totalreward"]
        # append data to table
        randomTable.add_row([problem, randomSuccessRate, randomReward])
        # write a summary to return to compare agents
        randomSummary[problem] = {
            "successrate": randomSuccessRate,
            "totalreward": randomReward
        }
    print("___________________________________________________________")
    print("Average Success Rate: " + str(np.average(randomSuccessRates)))
    print("Total Reward: " + str(totalRandomRewards))
    print(randomTable)
    plot_random(randomStats)
    os.system("cls")

    simpleTable = PrettyTable()
    simpleTable.field_names = ["Problem", "Solution Trace", "Iterations"]
    simpleSummary = {}  # for returning a summary of the agent
    print("___________________________________________________________")
    print("Simple Agent Running...")
    time.sleep(3)
    # handle simple
    for problem in range(numberOfProblems):
        print("___________________________________________________________")
        print("Problem " + str(problem) + "...")
        time.sleep(2)
        statsSimple = run_simple.main(problem, mapID)
        simpleStats.append(statsSimple)
        solutionTrace = statsSimple["solutiontrace"]
        numberOfIterations = statsSimple["numberofiterations"]
        simpleTable.add_row([problem, solutionTrace, numberOfIterations])
        # write a summary to return to compare agents
        simpleSummary[problem] = {"successrate": 100, "totalreward": 1}
    print("___________________________________________________________")
    print(simpleTable)
    plot_simple(simpleStats)
    os.system("cls")

    rlTable = PrettyTable()
    rlTable.field_names = ["Problem", "Success Rate", "Total Reward"]
    rlSummary = {}  # for returning a summary of the agent
    print("___________________________________________________________")
    print("Reinforced Learning Agent Running...")
    time.sleep(3)
    # handle rl
    for problem in range(numberOfProblems):
        print("___________________________________________________________")
        print("Problem " + str(problem) + "...")
        time.sleep(2)
        statsRl, Q = run_rl.main(problem, mapID)
        rlStats.append(statsRl)
        rlSuccessRate = statsRl["successrate"]
        rlTotalReward = statsRl["totalreward"]
        rlTable.add_row([problem, rlSuccessRate, rlTotalReward])
        # write a summary to return to compare agents
        rlSummary[problem] = {
            "successrate": rlSuccessRate,
            "totalreward": rlTotalReward
        }
    print("___________________________________________________________")
    print(rlTable)
    plot_rl(rlStats)
    return randomSummary, simpleSummary, rlSummary


def plot_random(stats):
    for i in range(len(stats)):
        plt.figure("Episode VS Reward, Problem: " + str(i))
        plt.title("Episode VS Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        x = []
        y = []
        for e in range(len(stats[i]["episodes"])):
            x.append(e)
            y.append(stats[i]["episodes"][e]["reward"])
        plt.scatter(x, y)
    print("Please Close All Plots To Continue...")
    plt.show()

    for i in range(len(stats)):
        plt.figure("Steps VS Reward, Problem: " + str(i))
        plt.title("Steps VS Reward")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        x = []
        y = []
        for e in range(len(stats[i]["episodes"])):
            x.append(stats[i]["episodes"][e]["steps"])
            y.append(stats[i]["episodes"][e]["reward"])
        plt.scatter(x, y)
    print("Please Close All Plots To Continue...")
    plt.show()


def plot_simple(stats):
    plt.figure("Problem VS Iteration")
    plt.title("Problem VS Iteration")
    plt.xlabel("Problem Number")
    plt.ylabel("Number of Iterations")
    x = []
    y = []
    for i in range(len(stats)):
        x.append(i)
        y.append(int(stats[i]["numberofiterations"]))

    plt.scatter(x, y)
    print("Please Close All Plots To Continue...")
    plt.show()


def plot_rl(stats):
    for i in range(len(stats)):
        plt.figure("Episode VS Reward, Problem: " + str(i))
        plt.title("Episode VS Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        x = []
        y = []
        for e in range(len(stats[i]["episodes"])):
            x.append(e)
            y.append(stats[i]["episodes"][e]["reward"])
        plt.scatter(x, y)
    print("Please Close All Plots To Continue...")
    plt.show()

    for i in range(len(stats)):
        plt.figure("Steps VS Reward, Problem: " + str(i))
        plt.title("Steps VS Reward")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        x = []
        y = []
        for e in range(len(stats[i]["episodes"])):
            x.append(stats[i]["episodes"][e]["steps"])
            y.append(stats[i]["episodes"][e]["reward"])
        plt.scatter(x, y)
    print("Please Close All Plots To Continue...")
    plt.show()


def main(mapID):
    if (mapID == "4x4-base"):
        numberOfProblems = 4
    elif (mapID == "8x8-base"):
        numberOfProblems = 8
    elif (mapID == "16x16-base"):
        numberOfProblems = 16
    else:
        print("Invalid Map ID Selected, aborting...")
        exit()
    randomSummary, simpleSummary, \
        rlSummary = process_stats(mapID, numberOfProblems)

    summaryTable = PrettyTable()
    summaryTable.field_names = [
        "Agent", "Problem", "Success Rate", "Total Reward"
    ]
    # after evaluating all agents we must compare them
    for problem in range(numberOfProblems):
        summaryTable.add_row([
            "Random Agent", problem, randomSummary[problem]["successrate"],
            randomSummary[problem]["totalreward"]
        ])

    for problem in range(numberOfProblems):
        summaryTable.add_row([
            "Simple Agent", problem, simpleSummary[problem]["successrate"],
            simpleSummary[problem]["totalreward"]
        ])

    for problem in range(numberOfProblems):
        summaryTable.add_row([
            "RL Agent", problem, rlSummary[problem]["successrate"],
            rlSummary[problem]["totalreward"]
        ])

    print(summaryTable)


if __name__ == "__main__":
    main(sys.argv[1])
