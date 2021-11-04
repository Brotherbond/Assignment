## Matplotlib Part 9

# Histograms

import matplotlib.pyplot as plt

ages=[22,55,62,45,21,22,99,34,42,4,102,110,27,48,99,84]
bins=[0,20,40,60,80,100,120]

plt.hist(ages,bins)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Current ages of people')
plt.savefig('hist.png',dpi=500)
plt.show()