import matplotlib.pyplot as plt

labels = ['Python', 'C++', 'Ruby', 'Java']
values = [215, 130, 245, 210]

bars = plt.bar(labels, values, align='center')
plt.xticks(labels)

# Patterning can be done by set_hatch()
patterns = ['/', '+', 'x', '\\', '*', 'o', 'O', '.']

for bar in bars:
    bar.set_hatch(patterns.pop(0))

plt.show()


plt.show()
