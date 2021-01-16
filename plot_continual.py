import matplotlib.pyplot as plt
import json
import numpy as np

import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--models-name', help="The name of the model", default='ranger_buff*')

args = parser.parse_args()

plots = args.models_name.split(";")

legends = []
for (i,plot) in enumerate(plots):

    plot_title = ''

    if 'mult' in plot:
        plot_title = ' - Multiplicative'
    if 'add' in plot:
        plot_title = ' - Mean'
    if 'fine' in plot:
        plot_title = ' - Fine Tuning'
    if 'entr' in plot:
        plot_title = ' - Entropy'

    plt.figure(i)
    plt.title("Mean Rewards{}".format(plot_title))
    #plt.title("Ranger (trained with DeepCrawl) + Archer (trained with AIRL)")

    rewards = []
    filenames = []

    models_name = plot
    models_name = models_name.replace(' ', '')
    models_name = models_name.replace('.json', '')
    models_name = models_name.split(",")

    for model_name in models_name:
        path = glob.glob("reward_experiments/" + model_name + ".json")
        path.sort()
        for filename in path:
            with open(filename, 'r') as f:
                filenames.append(filename)
                rewards.append(json.load(f))

    keys = rewards[0].keys()

    min_dict = dict()
    max_dict = dict()
    for k in keys:
        min_dict[k] = 9999999
        max_dict[k] = -9999999

    for k in keys:
        for r_dict in rewards:
            for r in r_dict[k]:
                if np.min(r) < min_dict[k]:
                    min_dict[k] = np.min(r)
                if np.max(r) > max_dict[k]:
                    max_dict[k] = np.max(r)


    min_dict['reward_0'] = 0
    max_dict['reward_0'] = 10.

    for k in keys:
        all_rews = []
        data = []
        for r_dict in rewards:
            length = 0
            episode_rewards = []
            print(np.sum(r_dict[k]))
            for r in r_dict[k]:
                length += len(r)
                current_rews = [(v - min_dict[k]) / (max_dict[k] - min_dict[k]) for v in r]
                #current_rews = r
                episode_rewards.append(np.sum(current_rews))

            data.append(np.mean(episode_rewards))
            all_rews.extend(episode_rewards)

        data = np.array(data)
        print(data)
        data = (data - np.min(all_rews)) / (np.max(all_rews) - np.min(all_rews))
        print(np.max(all_rews))
        print(np.min(all_rews))
        print(data)

        plt.plot(range(len(data)), data, '-o')
        legends.append(k)

    #plt.xticks(np.arange(4), ['$\pi(r1)$', '$\pi(r1+r2)$', '$\pi(r1+r2+r3)$', '$\pi(r1+r3)$'])
    plt.xticks(np.arange(5), ['$t=0.0$', '$t=1.0$', '$t=0.5$', '$t=0.2$', '$t=0.1$'])
    plt.legend(legends)
plt.show()