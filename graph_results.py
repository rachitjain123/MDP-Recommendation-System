from mdp import MDP
import matplotlib.pyplot as plt
# import numpy as np


def graph_recommendation_score(scale=4, m=10, with_comparison=False):
    """
    Function to generate a graph for the recommendation score over a range of m for a set of k
    :param scale: the limit to which k should vary
    :param m: a parameter in recommendation score computation
    :param with_comparison: plot a random policy's graph
    :return: None
    """

    fig = plt.figure()
    k = [i+1 for i in range(1, scale)]
    x = [i + 1 for i in range(m)]
    for j in k:
        y_recommendation = []
        y_recommendation_rand = []
        rs = MDP(path='data-mini', k=j)
        rs.load('mdp-model_k=' + str(j) + '.pkl')
        for i in x:
            if with_comparison:
                rs.initialise_mdp()
                y_recommendation_rand.append(rs.evaluate_recommendation_score(m=i))
            y_recommendation.append(rs.evaluate_recommendation_score(m=i))

        plt.plot(x, y_recommendation, color=(0.2 + (j-2)*0.4, 0.4, 0.6, 0.6), label="MC model " + str(j))
        plt.scatter(x, y_recommendation, color=(0.2 + (j-2)*0.4, 0.4, 0.6, 0.6))

        if with_comparison:
            plt.plot(x, y_recommendation_rand, color=(0.2, 0.8, 0.6, 0.6), label="Random model, For m=" + str(m))
            plt.scatter(x, y_recommendation_rand)

        plt.xticks(x)
        plt.yticks([i for i in range(20, 100, 10)])

        for x1, y in zip(x, y_recommendation):
            text = '%.2f' % y
            plt.text(x1, y, text)

        if with_comparison:
            for x1, y in zip(x, y_recommendation_rand):
                text = '%.2f' % y
                plt.text(x1, y, text)

    fig.suptitle('Recommendation Score vs Prediction List size')
    plt.xlabel('Prediction List size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def graph_decay_score(scale, rand=False):
    """
    Function to generate a graph for the exponential decay score over a range of k
    :param scale: the limit to which k should vary
    :param rand: to use a random policy or not
    :return: None
    """

    fig = plt.figure()
    x = [i + 1 for i in range(scale)]
    y_decay = []
    for i in x:
        rs = MDP(path='data-mini', k=i)

        if rand:
            rs.initialise_mdp()
            y_decay.append(rs.evaluate_decay_score())
            continue

        rs.load('mdp-model_k=' + str(i) + '.pkl')
        y_decay.append(rs.evaluate_decay_score())

    plt.bar(x, y_decay, width=0.5, color=(0.2, 0.4, 0.6, 0.6))
    xlocs = [i + 1 for i in range(0, 10)]
    for i, v in enumerate(y_decay):
        plt.text(xlocs[i] - 0.46, v + 0.9, '%.2f' % v)

    plt.xticks(x)
    plt.yticks([i for i in range(0, 100, 10)])

    fig.suptitle('Avg Exponential Decay Score vs Number of items in each state')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.show()


# if __name__ == "__main__":
    # graph_decay_score(10, rand=False)
    # graph_decay_score(10, rand=True)
    # graph_recommendation_score(scale=4, m=16, with_comparison=False)

    # model1 = 67.88
    # model2 = 65.49
    # model3 = 49.88
    # model4 = 68.01
    #
    # fig, ax = plt.subplots()
    # index = np.arange(1)
    # bar_width = 0.001
    # opacity = 0.5
    #
    # rects1 = plt.bar(index, model1, bar_width, alpha=opacity, color=['#000099'], label='MC Model 2')
    # plt.text(index, model1+0.9, '%.2f' % model1)
    # rects2 = plt.bar(index+bar_width, model2, bar_width, alpha=opacity, color=['#0000FF'], label='MC Model 3')
    # plt.text(index+bar_width, model2+0.9, '%.2f' % model2)
    # rects3 = plt.bar(index+2*bar_width, model3, bar_width, alpha=opacity, color=['#3366FF'], label='Predictor NS')
    # plt.text(index+2*bar_width, model3+0.9, '%.2f' % model3)
    # rects4 = plt.bar(index+3*bar_width, model4, bar_width, alpha=opacity, color=['#9966FF'],
    #                  label='MC Mixture Model 4')
    # plt.text(index+3*bar_width, model4+0.9, '%.2f' % model4)
    #
    # plt.xlabel('Models')
    # plt.ylabel('Score')
    # plt.xticks(index-bar_width, (''))
    # plt.yticks([i for i in range(0, 100, 10)])
    # plt.legend()
    # plt.title('Exponential Decay Score')
    #
    # plt.tight_layout()
    # plt.show()
