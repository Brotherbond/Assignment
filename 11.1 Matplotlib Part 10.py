## Matplotlib Part 10

# Pie Charts

import matplotlib.pyplot as plt

labels_1=['Facebook','Instagram','Youtube','linked-in']
views=[300,350,400,450]
explode_1=[0,0,0,0.2]

plt.pie(views,labels=labels_1,
        autopct='%1.1f%%',
        explode=explode_1,
        shadow=True)
plt.savefig('pie.png',dpi=500)
plt.show()