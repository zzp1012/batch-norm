import numpy as np
import os 

grads_path = '/data2/zzp1012/batch-norm-exp3/outs/0511/exp3/0511-125633-seed0-sample_num100-input_dim10-itrs1000-max_orger4/exp/x_grads.npy'
max_order = 4
itrs = 1000


if __name__ == '__main__':
    grads = np.load(grads_path)
    print(grads.shape)
    for i in range(max_order):
        # absolute difference
        delta_grads = grads[:, i] - grads[:, i+1]
        delta_grads_norm = np.linalg.norm(delta_grads.reshape(itrs, -1), ord=2, axis=-1)
        print(f'delta{i+1}_mean: {delta_grads_norm.mean()}')
        print(f'delta{i+1}_std: {delta_grads_norm.std()}')
        # relatively
        # calculate the base loss grad's norm
        base_loss_grads_norm = np.linalg.norm(grads[:, i].reshape(itrs, -1), ord=2, axis=-1)
        relative_delta_grads_norm = delta_grads_norm / base_loss_grads_norm
        print(f'delta{i+1} / loss{i} - mean: {relative_delta_grads_norm.mean()}')
        print(f'delta{i+1} / loss{i} - std: {relative_delta_grads_norm.std()}')
