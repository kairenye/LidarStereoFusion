"""
This file is a wrapper for using tensorboard in PyTorch. I just found it easier
to use than TensorboardX.
Credit: https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e
"""

import io
import numpy as np
from PIL import Image
import tensorflow as tf

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_image(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

# if __name__ == '__main__':
#     tensorboard = Tensorboard('logs')
#     x = np.arange(1,101)
#     y = 20 + 3 * x + np.random.random(100) * 100

#     # Log simple values
#     for i in range(0,100):
#         tensorboard.log_scalar('value', y[i], i)

#     # Log images
#     img = skimage.io.imread(r'C:\Users\212551241\Downloads\example_img.jpg')
#     tensorboard.log_image('example_image', img, 0)

#     # Log plots
#     fig = plt.figure()
#     plt.plot(x, y, 'o')
#     plt.close()
#     tensorboard.log_plot('example_plot', fig, 0)

#     # Log histograms
#     rng = np.random.RandomState(10)
#     a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
#     tensorboard.log_histogram('example_hist', a, 0, 'auto')

#     tensorboard.close()