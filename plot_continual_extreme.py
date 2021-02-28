import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="dark")
sns.set(font="Times New Roman", font_scale=1.5)

import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--models-name', help="The name of the model", default='*shit_*')

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

    rewards = []
    filenames = []

    models_name = plot
    models_name = models_name.replace(' ', '')
    models_name = models_name.replace('.json', '')
    models_name = models_name.split(",")


    qual_metrics = []
    for model_name in models_name:
        path = glob.glob("reward_experiments/" + model_name + ".txt")
        path.sort()
        for filename in path:
            dictionary = dict()
            with open(os.path.join(filename), 'r') as f:
                print(f)
                for line in f:
                    if line.split(',')[0] in dictionary:
                        dictionary[line.split(',')[0]].extend([float(line.split(',')[1])])
                    else:
                        dictionary[line.split(',')[0]] = [float(line.split(',')[1])]
                qual_metrics.append(dictionary)

    win_rates = []
    for m in qual_metrics:
        win_rates.append(m['win_rate'][0])

    win_rates = np.asarray(win_rates)

    x1.plot(range(len(win_rates)), win_rates, '-o', ms=12, linewidth=4)

    x1.set_xticks([])
    x1.legend(legends)

width = 0.35
try:
    # Percentages
    win_rates = []
    for m in qual_metrics:
        win_rates.append(m['win_rate'][0])

    win_rates = np.asarray(win_rates)

    #melees = (melees - np.min(melees)) / (np.max(melees) - np.min(melees))
    #ranges = (ranges - np.min(ranges)) / (np.max(ranges) - np.min(ranges))

    pal = sns.color_palette("Reds_d", len(win_rates))
    rank = win_rates.argsort().argsort()
    labels = ["$\pi_0$", "$\pi_0 + \pi_1$", "$\pi_0 + \pi_1 + \pi_2$", "$\pi_0 + \pi_1 + \pi_2 + \pi_3$", "From\nScratch"]
    x = np.arange(len(win_rates))
    sns.barplot(x=x, y=(np.array(win_rates)), palette=np.array(pal[::-1])[rank], ax=x2)
    #x2.set_ylabel('Number of attacks')
    x2.set_xticks(x)
    x2.set_xticklabels(labels)
    x2.legend()

    x2.set_xticklabels(labels)
    for p in x2.patches:
        x2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                       va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.setp(x2.patches, linewidth=1.5, edgecolor='black')
except Exception as e:
    print(e)
    pass

x1.set_title('Win Rates', pad=20)
x2.set_title('Win Rates', pad=20)
y = input('Do you want to save it? ')
if y == 'y':
    plt.savefig('imgs/results_extreme.eps', bbox_inches='tight', pad_inches=0, format='eps')
sns.despine()
plt.show()