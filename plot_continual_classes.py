import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="dark")
sns.set(font="Times New Roman", font_scale=2)

import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--models-name', help="The name of the model", default='*warrior*')

args = parser.parse_args()

plots = args.models_name.split(";")

legends = []
f, (x1, x2) = plt.subplots(2,1, figsize=(10,6))
f.tight_layout(pad=0.5)
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

    #plt.title("Mean Rewards{}".format(plot_title))
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

    try:
        qual_metrics = []
        for filename in filenames:
            dictionary = dict()
            filename = '{}.txt'.format(filename.replace('.json',''))
            with open(os.path.join(filename), 'r') as f:
                for line in f:
                    if line.split(',')[0] in dictionary:
                        dictionary[line.split(',')[0]].extend([float(line.split(',')[1])])
                    else:
                        dictionary[line.split(',')[0]] = [float(line.split(',')[1])]
                qual_metrics.append(dictionary)
    except Exception as e:
        pass

    # Create legends
    if any('warrior' in f for f in filenames):
        legends.append('$R_0$')
        legends.append('$R_w$')
    elif any('archer' in f for f in filenames):
        legends.append('$R_0$')
        legends.append('$R_a$')
    else:
        legends.append('$R_0$')
        legends.append('$R_1$')
        legends.append('$R_2$')


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
    percentages = []
    all_data = []
    i = 0
    for k in keys:
        all_rews = []
        data = []
        for r_dict in rewards:
            length = 0
            episode_rewards = []

            for r in r_dict[k]:
                length += len(r)
                #current_rews = [(v - min_dict[k]) / (max_dict[k] - min_dict[k]) for v in r]
                current_rews = r
                episode_rewards.append(np.sum(current_rews))
            print(length)
            data.append(np.mean(episode_rewards))
            all_rews.extend(episode_rewards)

            if k == 'reward_0':
                percentages.append(np.sum(np.asarray(episode_rewards) > 0))

        data = np.array(data)
        #data = (data - np.min(all_rews)) / (np.max(all_rews) - np.min(all_rews))
        print(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        all_data.append(data)

        x1.plot(range(len(data)), data, '-o', ms=14, linewidth=5)

        #legends.append("$R_{}$".format(i))
        i += 1

    x1.set_xticks([])
    x1.legend(legends)

width = 0.35
try:
    # Percentages
    melees = []
    ranges = []
    for m in qual_metrics:
        melees.append(m['melee'][0]/m['episodes'][0])
        ranges.append(m['range'][0]/m['episodes'][0])

    melees = np.asarray(melees)
    ranges = np.asarray(ranges)

    #melees = (melees - np.min(melees)) / (np.max(melees) - np.min(melees))
    #ranges = (ranges - np.min(ranges)) / (np.max(ranges) - np.min(ranges))

    labels = ['Main\nPolicy', 'MP', 'PP', 'ET', 'EW', 'Real\nClass']
    x = np.arange(len(filenames))
    rect1 = x2.bar(x - width / 2, melees, width=width, label='Melee', color='seagreen')
    rect2 = x2.bar(x + width / 2, ranges, width=width, label='Ranged', color='khaki')
    x2.set_xticks(x)
    x2.set_xticklabels(labels)
    x2.legend()

    x2.set_xticklabels(labels)
    for p in x2.patches:
        x2.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                       va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.setp(x2.patches, linewidth=1.5, edgecolor='black')
except Exception as e:
    print(e)
    pass

x1.set_title('Normalized Rewards', pad=20)
x2.set_title('Average Number of Attacks', pad=20)
y = input('Do you want to save it? ')
if y == 'y':
    plt.savefig('imgs/results_warrior.eps', bbox_inches='tight', pad_inches=0, format='eps')
sns.despine()
plt.show()