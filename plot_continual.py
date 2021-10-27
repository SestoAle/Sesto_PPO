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

parser.add_argument('-mn', '--models-name', help="The name of the model", default='*fountain**')

args = parser.parse_args()

plots = args.models_name.split(";")

legends = []
colors = []
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
    colors = []
    if any('def_entr' in f for f in filenames):
        legends.append('$R_0$')
        legends.append('$R_1$')
        colors.append('b')
        colors.append('chocolate')
        ylim_max = 72
        ylim_min = 50
    elif any('buff_entr' in f for f in filenames):
        legends.append('$R_0$')
        legends.append('$R_2$')
        colors.append('b')
        colors.append('g')
        ylim_max = 72
        ylim_min = 50
    elif any('def_buff' in f for f in filenames):
        legends.append('$R_0$')
        legends.append('$R_1$')
        legends.append('$R_2$')
        colors.append('b')
        colors.append('chocolate')
        colors.append('g')
        ylim_max = 72
        ylim_min = 50
    else:
        legends.append('$R_0$')
        legends.append('$R_1$')
        legends.append('$R_2$')
        colors.append('b')
        colors.append('chocolate')
        colors.append('g')
        ylim_max = 77.5
        ylim_min = 48


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
            #print(length)
            data.append(np.mean(episode_rewards))
            all_rews.extend(episode_rewards)

            if k == 'reward_0':
                percentages.append(np.sum(np.asarray(episode_rewards) > 0))

        print(filenames)
        data = np.array(data)
        #data = (data - np.min(all_rews)) / (np.max(all_rews) - np.min(all_rews))
        #print(data)
        print(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        all_data.append(data)

        x1.show_data(range(len(data)), data, '-o'.format(colors[i]), ms=13, linewidth=5, color=colors[i])

        i += 1

    x1.set_xticks([])
    x1.legend(legends)


try:
    # Percentages
    percentages = []
    for m in qual_metrics:
        if 'items' in filename:
            percentages.append(m['loot'][0])
        else:
            percentages.append(round(m['win_rate'][0],1))


    if 'items' in filename:
        pal = sns.random_color("summer_r", len(percentages))
    else:
        pal = sns.random_color("Reds_d", len(percentages))
    rank = data.argsort().argsort()
    x = np.array(range(len(percentages)))
    x2.legend('win rate')
    sns.barplot(x=x, y=(np.array(percentages)), palette=np.array(pal[::-1])[rank], ax=x2)
    x2.set_ylim(ylim_min, ylim_max)
    labels = ['Main\nPolicy', 'MP', 'PP', 'ET', 'EW', 'From\nScratch', 'Fine\nTuning']
    x2.set_xticklabels(labels)
    for p in x2.patches:
        x2.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                       va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.setp(x2.patches, linewidth=1.5, edgecolor='black')
except Exception as e:
    print(e)
    labels = ['None', 'Mean', 'Mult', 'Entr Thresh', 'Entr Weight', 'From Scratch', 'Fine Tunining']
    x1.set_xticklabels(labels)
    pass

x1.set_title('Normalized Rewards', pad=20)
if 'items' in filename:
    x2.set_title('Number of Collected Loot', pad=20)
else:
    x2.set_title('Win Rates', pad=20)

y = input('Do you want to save it? ')
if y == 'y':
    plt.savefig('imgs/results_fountain.eps', bbox_inches='tight', pad_inches=0, format='eps')
sns.despine()
plt.show()
