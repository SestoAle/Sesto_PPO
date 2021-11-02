import matplotlib.pyplot as plt
import numpy as np


lines = []
for i in range(5):
    line = np.arange(20) * np.random.randint(5, 20)
    lines.append(line)

lines.append(np.arange(24) * np.random.randint(5, 20))

plt.figure()
for l in lines:
    plt.plot(range(len(l)), l)

plt.figure()
#lines = np.asarray(lines)
means = np.mean(lines, axis=0)
stds = np.std(lines, axis=0)
plt.plot(range(len(means)), means)
plt.fill_between(range(len(stds)), means-stds, means+stds, alpha=0.5)

plt.show()