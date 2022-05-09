import numpy as np
import os 

grads_path = '/data2/tangling/batch-norm/outs/tmp/0509-232126-seed0-sample_num100-input_dim10-itrs1000-max_orger4/exp/x_grads.npy'
max_order = 4
itrs = 1000
#(1000,5,100,1)

if __name__ == '__main__':
    grads = np.load(grads_path)

    grad_norm = np.linalg.norm(grads[:,0].reshape(itrs,-1),ord=2,axis=1)
    print(grad_norm.mean())

    for i in range(max_order):

        grad_norm = np.linalg.norm(grads[:,i+1].reshape(itrs,-1),ord=2,axis=1)
        print(grad_norm.mean())

        delta_grads = grads[:,i] - grads[:,i+1]
        norm = np.linalg.norm(delta_grads.reshape(itrs,-1),ord=2,axis=1)
        print(f'delta{i+1}_mean: {norm.mean()}')
        print(f'delta{i+1}_std: {norm.std()}')