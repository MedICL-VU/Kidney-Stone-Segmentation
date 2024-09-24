import os
import pandas as pd


class History:
    def __init__(self, score='Dice', path=False, model_name='unet'):
        self.hist = []
        self.score = score
        self.model_name = model_name
        if path:
            with open(path, 'r') as f:
                self.hist = [float(l) for l in f]

    def __iter__(self):
        for val in self.hist:
            yield val

    def write(self):
        f = open(os.path.join('summaries/hist', self.model_name + '_hist.txt'), 'w')
        for val in self.hist:
            f.write(str(val) + '\n')
        f.close()

    def log(self, val):
        self.hist += [float(val)]

    def plot(self, save=False):
        try:
            import seaborn as sns
        except BaseException:
            print('Could not import seaborn in History.py')
            return

        summary = pd.DataFrame(self.hist).reset_index()
        summary.columns = ['epoch', self.score]

        sns.set_theme(style="darkgrid")
        try:
            p = sns.lineplot(x='epoch', y=self.score, data=summary)
        except BaseException as e:
            print(e)
            return

        if save:
            p.figure.savefig(os.path.join('summaries', self.model_name + '.png'))

        return p
