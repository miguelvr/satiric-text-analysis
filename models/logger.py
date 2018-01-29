from __future__ import division
import time
import numpy as np
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from data.text_utils import flatten_list


def color(x, color_select):
    colors = {
        'green': '\033[1;32m',
        'yellow': '\033[1;33m',
        'red': '\033[1;31m',
        'end': '\033[0;0m'
    }

    if isinstance(x, float):
        return '%s%2.3f%s' % (colors[color_select], x, colors['end'])
    elif isinstance(x, int):
        return '%s%d%s' % (colors[color_select], x, colors['end'])
    else:
        return '%s%s%s' % (colors[color_select], x, colors['end'])


class ClassificationLogger(object):
    def __init__(self, monitoring_metric, nr_samples, batch_size):
        self.state = None

        self.monitoring_metric = monitoring_metric
        self.metrics = {
            'accuracy': [],
            'f1_product': []
        }

        self.best_monitoring_metric = 0.
        self.best_epoch = 0
        self.n_samples = nr_samples
        self.b_size = batch_size

        if self.b_size:
            self.n_batches = int(np.ceil(self.n_samples / self.b_size))
        else:
            self.n_batches = 1

        self.batch_idx = 0
        self.pbar = None
        self.epoch = 0
        self.init_time = time.time()

    def update_on_batch(self, objective):

        if objective:
            self.batch_idx += 1

            if self.batch_idx == 1:
                self.pbar = tqdm(total=self.n_samples, ncols=100, leave=False)

            self.pbar.set_description(
                'Epoch %i | Batch %i/%i' %
                (self.epoch + 1, self.batch_idx, self.n_batches)
            )
            self.pbar.set_postfix(loss=objective)
            self.pbar.update(self.b_size)

    def update_on_epoch(self, predictions, gold):

        # Reset save state
        self.state = None

        if self.pbar:
            self.pbar.close()
            self.batch_idx = 0

        gold_tags = np.array(flatten_list(gold))
        pred_tags = np.where(np.array(flatten_list(predictions)) >= 0.5, 1., 0.)

        if 'f1_product' in self.metrics:
            f1 = f1_score(gold_tags, pred_tags, average=None)
            metric_score = f1[0] * f1[1]
            self.metrics['f1_product'].append(metric_score)

        if 'accuracy' in self.metrics:
            metric_score = accuracy_score(gold_tags, pred_tags)
            self.metrics['accuracy'].append(metric_score)

        # Update state
        self.state = None
        self.epoch += 1

        color_select = 'red'
        if self.metrics[self.monitoring_metric][-1] > self.best_monitoring_metric:
            self.state = 'save'
            self.best_monitoring_metric = self.metrics[self.monitoring_metric][-1]
            self.best_epoch = self.epoch
            color_select = 'green'

        epoch_time = (time.time() - self.init_time)

        # Inform user
        print("Epoch %d |" % self.epoch),
        print(
                "%s %s accuracy %2.3f |" %
                (
                    self.monitoring_metric,
                    color(self.metrics[self.monitoring_metric][-1], color_select),
                    self.metrics['accuracy'][-1]
                )
        ),
        print("Best %s: %s at epoch %d |" %
              (self.monitoring_metric, color(self.best_monitoring_metric, 'yellow'), self.best_epoch)),
        print("Time elapsed: %d seconds" % epoch_time)

        self.init_time = time.time()
