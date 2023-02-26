import numpy as np
a=np.load('episode_rewards_0.npy')
episode_rewards=a[()]
print(len(episode_rewards))
a=np.load('win_rates_0.npy')
win_rates=a[()]
print(len(win_rates))


import matplotlib.pyplot as plt;
plt.figure()
plt.ylim([0, 105])
plt.cla()
plt.subplot(2, 1, 1)
plt.plot(range(len(win_rates)), win_rates)
plt.xlabel('step*{}'.format(5000))
plt.ylabel('win_rates')

plt.subplot(2, 1, 2)
plt.plot(range(len(episode_rewards)),episode_rewards)
plt.xlabel('step*{}'.format(5000))
plt.ylabel('episode_rewards')

plt.savefig( './plt_{}.png'.format(0), format='png')