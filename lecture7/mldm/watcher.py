import matplotlib.pyplot as plt
import numpy as np

class Watcher(object):
    def __init__(self, epoches):

        self.fig = plt.figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(111)
        
        self.ax.set_xlim([0.0, epoches - 1])
        self.ax.set_ylim([0.0, 1.0])

        self.mean_loss, = self.ax.plot([], [], label='mean loss', color='blue')
        self.loss, = self.ax.plot([], [], alpha=0.5, color='blue')
        
        self.ax.legend()
        
    def draw(self, ls):
        s = np.std(ls)
        self.ax.set_ylim(np.percentile(ls, q=2) - s, np.percentile(ls, q=98) + s)
        
        self.mean_loss.set_xdata(np.arange(ls.shape[0]))
        self.mean_loss.set_ydata(np.mean(ls, axis=1))
        
        self.loss.set_xdata(np.linspace(0, ls.shape[0] - 1, num=np.prod(ls.shape)))
        self.loss.set_ydata(ls.ravel())
        
        self.fig.canvas.draw()