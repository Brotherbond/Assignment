## Matplotlib Part 2

import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7] # week days
y=[964,287,42,390,64,800,523] # views

plt.plot(x,y,label='Youtube views',
         color='r',
         marker='o',
         markerfacecolor='g',
         linestyle='--',
         linewidth=3)
plt.xlabel('Week days')
plt.ylabel('Views')
plt.title('Youtube views on daily basis')
plt.legend(loc='upper left')

plt.show()