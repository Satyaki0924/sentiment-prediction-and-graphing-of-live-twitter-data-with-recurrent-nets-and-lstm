import os
import matplotlib.pyplot as plt
from matplotlib import style

from pylab import rcParams

rcParams['figure.figsize'] = 10, 2
style.use("ggplot")


class Plot(object):
    @staticmethod
    def plot():
        path = os.path.dirname(os.path.realpath(__file__))
        while True:
            kyword = str(input('Enter keyword for plot:\n>> ')).lower()
            if os.path.exists(os.path.dirname(os.path.realpath(path)) + '/points/' + kyword + '-plot.txt'):
                break
            else:
                print('There is no points to plot in this name... Try again')
        pullData = open(os.path.dirname(os.path.realpath(path)) + '/points/' + kyword + '-plot.txt', "r").read()
        lines = pullData.split('\n')
        lines = [float(l) for l in lines if l]
        neg = [float(l) for l in lines if l < 0.5]
        pos = [float(l) for l in lines if l > 0.5]
        per_pos = int((len(pos) / len(lines)) * 100)
        per_neg = int((len(neg) / len(lines)) * 100)
        plt.figure(figsize=(20, 10))
        plt.subplot(221)
        all1 = plt.plot([0.50 for i in range(len(lines))])
        all2 = plt.plot(lines, c='b')
        plt.legend((all1[0], all2[0]), ('Horizon', 'Sentiment graph'))
        plt.title(kyword + "'s twitter sentiments")
        plt.subplot(222)
        pos1 = plt.plot([0.50 for _ in range(len(pos))], c='b')
        pos2 = plt.plot(pos, c='g')
        plt.legend((pos1[0], pos2[0]), ('Horizon', 'Positive sentiment graph'))
        plt.title(kyword + "'s positive sentiments")
        plt.subplot(223)
        neg1 = plt.plot([0.50 for _ in range(len(neg))], c='b')
        neg2 = plt.plot(neg, c='r')
        plt.legend((neg1[0], neg2[0]), ('Horizon', 'Negative sentiment graph'))
        plt.title(kyword + "'s negative sentiments")
        plt.subplot(224)
        per1 = plt.bar([1], [per_pos], width=0.1, color='g')
        per2 = plt.bar([10], [per_neg], width=0.1, color='r')
        plt.legend((per2[0], per1[0]), ('Negative = {:}% '.format(per_neg), 'Positive = {:}%'.format(per_pos)))
        plt.title(kyword + "'s positive to negative percentage")
        plt.savefig(os.path.dirname(os.path.realpath(path)) + '/plots/' + kyword + '-plot.png')
        print('*** Image of plot saved successfully ***')