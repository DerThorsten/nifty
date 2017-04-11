import nifty
import nifty.deep_learning

import keras

from nifty.deep_learning.targets import seg_batch_to_affinity_edges,cremi_z5_edges
from nifty.deep_learning.data_loader import MultiresCremiLoader
from nifty.deep_learning.models.homer_net import HomerNet
from nifty.deep_learning.loss.pixelwise_weighted_loss import LiftedEdgeLoss

import numpy
import random

# numpy.random.seed(7)
# random.seed(7)

params = {
    'cremi_data_folder' : 
        #  raw folder
        '/media/tbeier/4D11469249B96057/work/cremi/data/raw_padded',

    # data loader
    'data_loader' : {
        'neuron_ids' : True,
        'clefts' :     False,
        'patch_sizes' : [200,200,200],
        'n_z' : 5,
        'z_slice_ranges' : [[0,15 ], [50,55], [110,120]],
        'bad_slices'     : [[],[],[14]]
    },

    # the net
    'net' : {
        'n_resolutions' : 3,
        'input_patch_shapes' : [[200,200,5]]*3
    }

}


train_batch_size = 1

# the data source
loader = MultiresCremiLoader(root_folder=params['cremi_data_folder'],**params['data_loader'])

# the targets
edges = cremi_z5_edges(r=1, atrous_rate_xy=1, add_local_edges=False)
edge_priors = numpy.ones(len(edges))*0.2

# the training data gen
def gen():
    while True:
        raw_batch_list, neuron_ids_batch  = loader(batch_size=train_batch_size)
        gt_and_weights = seg_batch_to_affinity_edges(neuron_ids_batch, edges=edges, edge_priors=edge_priors)
        yield raw_batch_list, gt_and_weights


for x,y in gen():
    break
# make the loss function(s)
input_shape = [200,200,len(edges)*2]
loss = LiftedEdgeLoss(input_shape=input_shape, n_channels=len(edges))



# make the net
net_param = params['net']
homer_net = HomerNet(loss_function=loss, **params['net'])



class Trainer(object):
    def __init__(self, net, loss_functions):

        self.net = net
        self.loss_functions = loss_functions
        self.model = self.net.model
        opt = keras.optimizers.Adam()
        self.model.compile(optimizer=opt, loss=self.loss_functions)

    def train(self, train_gen, valid_gen,train_batch_size, n_val=30, epochs=100000, s_mult=50, callbacks=None):

        internal_callbacks = []

        self.model.fit_generator(generator=train_gen(), 
            steps_per_epoch=s_mult*train_batch_size,  
            epochs=epochs, 
            validation_data=valid_gen(),
            validation_steps=n_val,
            verbose=True,
            callbacks=internal_callbacks
        )


trainer = Trainer(net=homer_net, loss_functions=loss)
trainer.train(train_gen=gen, valid_gen=gen, train_batch_size=train_batch_size)



