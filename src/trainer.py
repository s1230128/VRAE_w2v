import pickle
import numpy as np
import chainer.functions as F
import chainer.iterators as I
import chainer.optimizers as O
import chainer.training  as T
import chainer.training.extensions as E
import datetime
from VRAE import *


''' Setting '''
# Network Parameter
size_batch = 2
size_hidden = 32
size_middle = 1
rate_dropout = 0.1

# Trainer propertyç
out_dir = '../result/' + datetime.datetime.now().isoformat()
#   log
log_trigger = (10, 'epoch')
print_entries = ['epoch', 'main/loss', 'main/accu']
#   plot
plot_entries = ['main/loss']
plot_trigger = (1, 'epoch')
#   save
train_fname = 'train_epoch-{.updater.epoch}'
model_fname = 'model_epoch-{.updater.epoch}'
save_trigger = (100, 'epoch')


''' Loading '''
with open('../pickle/data.pickle' , 'rb') as f:  data = pickle.load(f)
with open('../pickle/vocab.pickle', 'rb') as f:  word_id, id_word = pickle.load(f)
n_data = len(data)
size_vocab = len(id_word)
print('data  :', n_data)
print('vocab :', size_vocab)


''' Training '''
# Network & Model
net = VRAE(size_vocab, size_hidden, size_middle, rate_dropout)
model = VRAELM(net)
# Optimizer
optimizer = O.Adam()
optimizer.setup(model)
# Data
iter_train = I.SerialIterator(data, size_batch, shuffle=True)
# Training
updater = T.StandardUpdater(iter_train, optimizer, converter=model.converter)
trainer = T.Trainer(updater, out=out_dir)
trainer.extend(E.LogReport(trigger=log_trigger))
trainer.extend(E.PrintReport(print_entries))
trainer.extend(E.PlotReport(plot_entries))
trainer.extend(E.snapshot(              filename=train_fname), trigger=save_trigger)
trainer.extend(E.snapshot_object(model, filename=model_fname), trigger=save_trigger)
trainer.run()
