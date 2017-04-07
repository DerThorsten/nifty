import numpy
import pylab
import math

from ... ground_truth import segToEdges2D,seg3dToCremiZ5Edges, thinSegFilter
from scipy import ndimage as ndi



def square_edges(r, half_size=False, atrous_rate=1, add_local_edges=False):
    s = r + 1
    offsets = [ ]
    if half_size:
        for x in range(s):
            y_start = -s +1
            if x == 0 :
                y_start = 0
            for y in range(y_start,s):
                offsets.append((x*atrous_rate,y*atrous_rate))
        offsets = offsets[1:]
    else:
        for x in range(-1*r,s):
            for y in range(-1*r,s):
                if x!=0 or y!=0:
                    offsets.append((x*atrous_rate,y*atrous_rate))

    if add_local_edges:
        offsets.append((0 , 1))
        offsets.append((0 ,-1))
        offsets.append((1 , 0))
        offsets.append((-1, 0))
    
    return offsets



def cremi_edges(r, rz, half_size=False, atrous_rate_xy=1, add_local_edges=False):

    s = r + 1
    ar = atrous_rate_xy
    offsets = [ ]
    if half_size:
        for x in range(s):
            y_start = -s +1
            if x == 0 :
                y_start = 0
            for y in range(y_start,s):
                for z in range(rz+1):
                    offsets.append((x*ar,y*ar, z))
        offsets = offsets[1:]
    else:
        for x in range(-1*r,s):
            for y in range(-1*r,s):
                for z in range(-rz, rz+1):
                    if x!=0 or y!=0 or z!=0:
                        offsets.append((x*ar,y*ar, z))

    if add_local_edges:
        offsets.append(( 0,  1,  0))
        offsets.append(( 0, -1,  0))
        offsets.append(( 1,  0,  0))
        offsets.append((-1,  0,  0))
        offsets.append(( 0,  0, -1))
        offsets.append(( 0,  0, +1))

    return offsets





def cremi_z5_edges(r, atrous_rate_xy=1, add_local_edges=True):
    
    edges = []
    # edges starting from the central slice
    edges_cetral = cremi_edges(r=r, rz=2, half_size=False, 
        atrous_rate_xy=atrous_rate_xy, add_local_edges=add_local_edges)



    # add a coordinate to indicate from where the edge is starting 
    edges_cetral = [ (0,)+ edge  for edge in edges_cetral]
    edges += edges_cetral

    # only 2d edges for other slices
    edges_non_central = square_edges(r=r, atrous_rate=atrous_rate_xy, add_local_edges=add_local_edges)
    z_starts = [2,1,-1,2]    

    for z_start in z_starts:
        edges += [ (z_start,) + edge + (0,)  for edge in edges_non_central]

    return edges
    


def seg_dt(seg,truncate=None):
    edges = segToEdges2D(seg)
    dt = ndi.distance_transform_edt(1 - edges)
    if truncate is not None:
        dt[dt>truncate] = truncate
    return dt



def truncated_dt(x, truncate=None, invert=True):
    #print("truncate=",truncate)
    if invert:
        x = 1-x


    dt = ndi.distance_transform_edt(x)
    if truncate is not None:
        dt[dt>truncate] = truncate
    return dt


def thin_segment_filter(seg, sigma=3.75):
    thin,dt = thinSegFilter(seg=seg,sigma=sigma)
    thin = numpy.exp(-0.9*thin)*3.0
    thin[dt==0] = 0
    thin[dt==1] = thin[dt==1]/2.0
    return thin 

def seg_batch_to_affinity_edges(neuron_ids_batch, edges, edge_priors,**kwargs):

    truncate_dt = kwargs.get('truncate_dt',None)
    false_positive_mult = kwargs.get('false_positive_mult',5.0)
    gamma = kwargs.get('gamma',0.06)

    n_batches = neuron_ids_batch.shape[0]
    n_z       = neuron_ids_batch.shape[3]
    shape_2d  = neuron_ids_batch.shape[1:3]




    n_edges =  len(edges)
    edges_array = numpy.array(edges, dtype='int32')


    out_shape = (n_batches,) + shape_2d + (2*n_edges,)

    batch_out =     numpy.zeros(out_shape ,dtype='float32')
    batch_gt      = batch_out[:,:,:,0:n_edges]
    batch_weights = batch_out[:,:,:,n_edges:2*n_edges]


    for bi in range(n_batches):


        gt = seg3dToCremiZ5Edges(
            segmentation=neuron_ids_batch[bi,:,:,:], 
            edges=edges_array).astype('float32')
        batch_gt[bi,...] = gt


        for ei in range(len(edges)):
            edge = edges[ei]
            edge_len_2d = int(math.sqrt(edge[1]**2 + edge[1]**2)+0.5)

            # has any cut edge?
            gt_ei= gt[:,:,ei]
            has_any = gt_ei.any()

            # make faster / more elegant
            w = numpy.ones(shape_2d,dtype='float32')*edge_priors[ei]
            w[gt_ei==1] = 1.0
            
            if has_any:
                dt = truncated_dt(gt[:,:,ei], truncate=truncate_dt).squeeze()
                gt[:,:,ei] = numpy.exp(-1.8*dt)
                w[dt==1] = 0.0
                mask = dt>2#+edge_len_2d
                w[mask] = (1.0-numpy.exp(-1.0*gamma*dt[mask].astype('float32')))*false_positive_mult*edge_priors[ei]
                #w += thin
            else:
                w[:] = 1.0*false_positive_mult

            batch_weights[bi,:,:,ei] = w


            if False:
                f = pylab.figure()
                f.add_subplot(1,2,1)
                s = pylab.imshow(gt[:,:,ei])
                cbar = f.colorbar(s, ticks=numpy.arange(0.0,1.0,0.1), orientation='vertical')
                f.add_subplot(1,2,2)
                s = pylab.imshow(w)
                cbar = f.colorbar(s, ticks=numpy.arange(0.0,false_positive_mult,0.1), orientation='vertical')
                pylab.show()


    return batch_out