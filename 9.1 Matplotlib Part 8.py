## Matplotlib Part 8

# Bar Charts

import matplotlib.pyplot as plt

y_views=[200,50,250,300,100,150,350]
fb_views=[160,200,40,240,80,120,280]

days=[1,2,3,4,5,6,7]

plt.bar(days,y_views,label='Youtube views')
plt.bar(days,fb_views,label='Facebook views')

plt.xlabel('Days')
plt.ylabel('Views')
plt.title('Social media views for a channel')
plt.legend()
plt.savefig('bar_2.png',dpi=500)
plt.show()