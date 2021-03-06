import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
mushrooms = pd.read_csv("./mushrooms.csv")
# print(mushrooms["habitat"].unique())
# print(mushrooms.groupby(['habitat', 'class']
#                         ).size().reset_index(name='counts'))
#
labels = np.array(['woods', 'grasses', 'leaves', 'meadows', 'paths', 'urban',
                   'waste'])
edible = [1880, 1408, 240, 256, 136, 96, 192]
poisonous = [1268, 740, 592, 36, 1008, 272, 0]
counts = [x + y for x, y in zip(edible, poisonous)]
edible = np.array([x / y for x, y in zip(edible, counts)])
poisonous = np.array([x / y for x, y in zip(poisonous, counts)])
print(edible)
# stats = df.loc[386, labels].values
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
# # close the plot
# stats = np.concatenate((stats, [stats[0]]))
edible = np.concatenate((edible, [edible[0]]))
poisonous = np.concatenate((poisonous, [poisonous[0]]))
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, edible, 'o-', linewidth=2)
ax.fill(angles, edible, alpha=0.25)
ax.plot(angles, poisonous, 'o-', linewidth=2)
ax.fill(angles, poisonous, alpha=0.25)
ax.set_thetagrids(angles * 180 / np.pi, labels)
ax.set_title("Percentage of poisonous and edible mushrooms")
ax.grid(True)
ax.legend(labels=['edible', 'poisonous'], loc='lower right')
# plt.show()

edible = np.array([1880, 1408, 240, 256, 136, 96, 192])
poisonous = np.array([1268, 740, 592, 36, 1008, 272, 0])

# stats = df.loc[386, labels].values
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
# # close the plot
# stats = np.concatenate((stats, [stats[0]]))
edible = np.concatenate((edible, [edible[0]]))
poisonous = np.concatenate((poisonous, [poisonous[0]]))
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, edible, 'o-', linewidth=2)
ax.fill(angles, edible, alpha=0.25)
ax.plot(angles, poisonous, 'o-', linewidth=2)
ax.fill(angles, poisonous, alpha=0.25)
ax.set_thetagrids(angles * 180 / np.pi, labels)
ax.set_title("Number of poisonous and edible mushrooms")
ax.grid(True)
ax.legend(labels=['edible', 'poisonous'], loc='lower right')
# plt.show()


# Create a dataframe
labels = ['woods', 'grasses', 'leaves', 'meadows', 'paths', 'urban',
                   'waste']
edible = [1880, 1408, 240, 256, 136, 96, 192]
poisonous = [1268, 740, 592, 36, 1008, 272, 0]
both = list()
both_labeles = list()
my_color = list()
for i in range(len(edible)):
    both.append(edible[i])
    both.append(poisonous[i])
    both_labeles.append('   ' + labels[i])
    both_labeles.append(labels[i] + '   ')
    my_color.append('skyblue')
    my_color.append('orange')
print(both_labeles)

df = pd.DataFrame({'group': both_labeles,
                   'values': both})

# Reorder it following the values:
# ordered_df = df.sort_values(by='values')
my_range = range(1, len(df.index) + 1)

# Create a color if the group is "B"
# my_color = np.where(df['group'] == 'woods', 'orange', 'skyblue')
my_size = np.where(df['group'] == 'woods', 100, 100)
# The vertival plot is made using the hline function
# I load the seaborn library only to benefit the nice looking feature
plt.vlines(x=my_range, ymin=0,
           ymax=df['values'], color=my_color, alpha=0.4)
plt.scatter(my_range, df['values'], color=my_color, marker='^',
            s=my_size, alpha=1, linewidth=10)
# Add title and exis names
mod_ran = ([(a + b) / 2 for a, b in zip(my_range[::2], my_range[1::2])])
print(type(df['group']))
plt.xticks(mod_ran, labels)
plt.title("Mushrooms and it's habitat", loc='left')
plt.xlabel('habitat')
plt.show()
