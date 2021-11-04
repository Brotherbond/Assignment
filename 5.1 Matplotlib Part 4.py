## Matplotlib Part 4

# Setting of X and Y axis limits

import matplotlib.pyplot as plt

y_views=[534,690,258,402,724,689,352]
f_views=[123,342,700,305,406,648,325]
t_views=[202,209,176,415,824,389,550]
days=[1,2,3,4,5,6,7]

plt.plot(days,y_views,label='Youtube views',marker='o')
plt.plot(days,f_views,label='Facebook views',marker='o')
plt.plot(days,t_views,label='Twitter views',marker='o')
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('Social media views for a channel')
plt.legend()

plt.xlim(1,5)
plt.ylim(100,900)
plt.show()