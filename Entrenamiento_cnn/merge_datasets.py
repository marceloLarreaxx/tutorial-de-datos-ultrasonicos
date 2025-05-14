import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')
# sys.path.append(r'C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV')

import numpy as np
import pickle
import tensorflow as tf
import imag3D.CNN_superficie.cnnsurf_funcs as fu

#
# def merge(cfg):
#     subdata = {}
#     for s in cfg['piezas']:
#         with open(cfg['data_path'] + s + '.pickle', 'rb') as f:
#             subdata[s] = pickle.load(f)
#
#     # concatenar los datos
#     train_fmc, test_fmc, train_t, test_t = [], [], [], []
#     n_data = {}
#     for i, s in enumerate(cfg['piezas']):
#         train_fmc.append(subdata[s]['train_fmc'])
#         train_t.append(subdata[s]['train_t'])
#         test_fmc.append(subdata[s]['test_fmc'])
#         test_t.append(subdata[s]['test_t'])
#         n_data[s] = [train_fmc[i].shape[0], test_fmc[i].shape[0]]
#
#     # hacer que todos tenga el mismo nro de samples
#     for i in range(len(cfg['piezas'])):
#         n = train_fmc[i].shape[-1]
#         if n > cfg['n_samples']:
#             # recortar
#             train_fmc[i] = np.delete(train_fmc[i], slice(cfg['n_samples'], n), axis=-1)
#             test_fmc[i] = np.delete(test_fmc[i], slice(cfg['n_samples'], n), axis=-1)
#         elif n < cfg['n_samples']:
#             # agregar ceros
#             temp = list(train_fmc[i].shape)
#             temp[-1] = cfg['n_samples']
#             temp2 = np.zeros(temp)
#             temp2[:, :, :, :n] = train_fmc[i]
#             train_fmc[i] = temp2
#             temp = list(test_fmc[i].shape)
#             temp[-1] = cfg['n_samples']
#             temp2 = np.zeros(temp)
#             temp2[:, :, :, :n] = test_fmc[i]
#             test_fmc[i] = temp2
#
#     train_fmc = np.concatenate(train_fmc)
#     test_fmc = np.concatenate(test_fmc)
#     train_t = np.concatenate(train_t)
#     test_t = np.concatenate(test_t)
#
#     print('creating labels')
#     train_mask = fu.crear_mask_labels(train_fmc, train_t, cfg)
#     test_mask = fu.crear_mask_labels(test_fmc, test_t, cfg)
#     print('done')
#
#     # data augmentation: flipear en las direcciones del array
#     if cfg['add_flip_data']:
#         print('flipping data')
#         train_fmc = fu.add_fliped_data(train_fmc)
#         train_mask = fu.add_fliped_data(train_mask)
#         test_fmc = fu.add_fliped_data(test_fmc)
#         test_mask = fu.add_fliped_data(test_mask)
#         train_t = fu.add_fliped_data(train_t)
#         test_t = fu.add_fliped_data(test_t)
#
#     if 'size_limit' in cfg.keys():
#         n1, n2 = cfg['size_limit']
#         np.random.seed(0)
#         print('reducing data size')
#         if train_fmc.shape[0] > n1:
#             idx = np.random.permutation(n1)
#             train_fmc = train_fmc[idx, ...]
#             train_mask = train_mask[idx, ...]
#             train_t = train_t[idx, ...]
#         if test_fmc.shape[0] > n2:
#             idx = np.random.permutation(n2)
#             test_fmc = test_fmc[idx, ...]
#             test_mask = test_mask[idx, ...]
#             test_t = test_t[idx, ...]
#
#     # agregar un dimension extra al final, que representa un solo canal
#     print('transformar numpy a tensor')
#     with tf.device('/cpu:0'):
#         train_fmc = tf.convert_to_tensor(np.expand_dims(train_fmc, axis=-1), dtype=np.float32)
#         test_fmc = tf.convert_to_tensor(np.expand_dims(test_fmc, axis=-1), dtype=np.float32)
#         train_mask = tf.convert_to_tensor(np.expand_dims(train_mask, axis=-1).astype(np.float32))
#         test_mask = tf.convert_to_tensor(np.expand_dims(test_mask, axis=-1).astype(np.float32))
#
#     return train_fmc, test_fmc, train_mask, test_mask, train_t, test_t


def merge(cfg, decimar=False, dims=[11,11]):
    subdata = {}
    for s in cfg['piezas']:
        with open(cfg['data_path'] + s + '.pickle', 'rb') as f:
        # with open(s + '.pickle', 'rb') as f:
            subdata[s] = pickle.load(f)

    # concatenar los datos
    fmc = {'train': [], 'val': [], 'test': []}
    tof = {'train': [], 'val': [], 'test': []}
    n_data = {}
    for i, s in enumerate(cfg['piezas']):
        n_data[s] = []
        for x in ['train', 'val', 'test']:
            if decimar:
                # quedarse con la mitad de las muestras
                fmc[x].append(subdata[s][x + '_fmc'][:, :, :, 0:-1:2])
                tof[x].append(np.round((subdata[s][x + '_t']/2)))
            else:
                fmc[x].append(subdata[s][x + '_fmc'])
                tof[x].append(subdata[s][x + '_t'])

            n_data[s].append(fmc[x][i].shape[0])

    # hacer que todos tenga el mismo nro de samples
    print('forzar nro de samples')
    for i in range(len(cfg['piezas'])):
        for x in ['train', 'val', 'test']:
            fmc[x][i] = fu.forzar_sample_number(fmc[x][i], cfg['n_samples'])

    print('concatenar y crear labels masks')
    mask = {}
    for x in ['train', 'val', 'test']:
        fmc[x] = np.concatenate(fmc[x])
        tof[x] = np.concatenate(tof[x])
        mask[x] = fu.crear_mask_labels(fmc[x], tof[x], cfg, dims=dims)
    print('done')

    # data augmentation: flipear en las direcciones del array
    if cfg['add_flip_data']:
        print('flipping data')
        for x in ['train', 'val', 'test']:
            fmc[x] = fu.add_fliped_data(fmc[x])
            tof[x] = fu.add_fliped_data(tof[x])
            mask[x] = fu.add_fliped_data(mask[x])

    if 'size_limit' in cfg.keys():
        nlim = cfg['size_limit']
        np.random.seed(0)
        print('reducing data size')
        for x in ['train', 'val', 'test']:
            if fmc[x].shape[0] > nlim[x]:
                idx = np.random.permutation(nlim[x])
                fmc[x] = fmc[x][idx, ...]
                mask[x] = mask[x][idx, ...]
                tof[x] = tof[x][idx, ...]

    # agregar un dimension extra al final, que representa un solo canal
    print('transformar numpy a tensor')
    with tf.device('/cpu:0'):
        for x in ['train', 'val', 'test']:
            fmc[x] = tf.convert_to_tensor(np.expand_dims(fmc[x], axis=-1), dtype=np.float32)
            mask[x] = tf.convert_to_tensor(np.expand_dims(mask[x], axis =-1), dtype=np.float32)

    # PRA DATOS DE CILINDRO 40
    # fmc['test'] = np.delete(fmc['test'], [4, 20, 36, 37, 43, 47, 50, 63, 82, 89], axis=0)
    # tof['test'] = np.delete(tof['test'], [4, 20, 36, 37, 43, 47, 50, 63, 82, 89], axis=0)
    return fmc, tof, mask



def merge_2(cfg, decimar=False):
    subdata = {}
    for s in cfg['piezas']:
        with open(cfg['data_path'] + s + '.pickle', 'rb') as f:
        # with open(s + '.pickle', 'rb') as f:
            subdata[s] = pickle.load(f)

    # concatenar los datos
    fmc = {'train': [], 'val': [], 'test': []}
    # tof = {'train': [], 'val': [], 'test': []}
    n_data = {}
    for i, s in enumerate(cfg['piezas']):
        n_data[s] = []
        for x in ['train', 'val', 'test']:
            if decimar:
                # quedarse con la mitad de las muestras
                fmc[x].append(subdata[s][x + '_fmc'][:, :, :, 0:-1:2])
                # tof[x].append(np.round((subdata[s][x + '_t']/2)))
            else:
                fmc[x].append(subdata[s][x + '_fmc'])
                # tof[x].append(subdata[s][x + '_t'])

            n_data[s].append(fmc[x][i].shape[0])

    # hacer que todos tenga el mismo nro de samples
    print('forzar nro de samples')
    for i in range(len(cfg['piezas'])):
        for x in ['train', 'val', 'test']:
            fmc[x][i] = fu.forzar_sample_number(fmc[x][i], cfg['n_samples'])

    print('concatenar y crear labels masks')
    mask = {}
    for x in ['train', 'val', 'test']:
        fmc[x] = np.concatenate(fmc[x])
        # tof[x] = np.concatenate(tof[x])
        # mask[x] = fu.crear_mask_labels(fmc[x], tof[x], cfg)
    print('done')

    # data augmentation: flipear en las direcciones del array
    if cfg['add_flip_data']:
        print('flipping data')
        for x in ['train', 'val', 'test']:
            fmc[x] = fu.add_fliped_data(fmc[x])
            # tof[x] = fu.add_fliped_data(tof[x])
            # mask[x] = fu.add_fliped_data(mask[x])

    if 'size_limit' in cfg.keys():
        nlim = cfg['size_limit']
        np.random.seed(0)
        print('reducing data size')
        for x in ['train', 'val', 'test']:
            if fmc[x].shape[0] > nlim[x]:
                idx = np.random.permutation(nlim[x])
                fmc[x] = fmc[x][idx, ...]
                # mask[x] = mask[x][idx, ...]
                # tof[x] = tof[x][idx, ...]

    # agregar un dimension extra al final, que representa un solo canal
    print('transformar numpy a tensor')
    with tf.device('/cpu:0'):
        for x in ['train', 'val', 'test']:
            fmc[x] = tf.convert_to_tensor(np.expand_dims(fmc[x], axis=-1), dtype=np.float32)
            # mask[x] = tf.convert_to_tensor(np.expand_dims(mask[x], axis =-1), dtype=np.float32)

    return fmc



def merge3(cfg, pz, decimar=False, dims=[11,11]):
    subdata = {}
    for s in cfg[pz]:
        with open(cfg['data_path'] + s + '.pickle', 'rb') as f:
        # with open(s + '.pickle', 'rb') as f:
            subdata[s] = pickle.load(f)

    # concatenar los datos
    fmc = {'train': [], 'val': [], 'test': []}
    tof = {'train': [], 'val': [], 'test': []}
    n_data = {}
    for i, s in enumerate(cfg[pz]):
        n_data[s] = []
        for x in ['train', 'val', 'test']:
            if decimar:
                # quedarse con la mitad de las muestras
                fmc[x].append(subdata[s][x + '_fmc'][:, :, :, 0:-1:2])
                tof[x].append(np.round((subdata[s][x + '_t']/2)))
            else:
                fmc[x].append(subdata[s][x + '_fmc'])
                tof[x].append(subdata[s][x + '_t'])

            n_data[s].append(fmc[x][i].shape[0])

    # hacer que todos tenga el mismo nro de samples
    print('forzar nro de samples')
    for i in range(len(cfg[pz])):
        for x in ['train', 'val', 'test']:
            fmc[x][i] = fu.forzar_sample_number(fmc[x][i], cfg['n_samples'])

    print('concatenar y crear labels masks')
    mask = {}
    for x in ['train', 'val', 'test']:
        fmc[x] = np.concatenate(fmc[x])
        tof[x] = np.concatenate(tof[x])
        mask[x] = fu.crear_mask_labels(fmc[x], tof[x], cfg, dims=dims)
    print('done')

    # data augmentation: flipear en las direcciones del array
    if cfg['add_flip_data']:
        print('flipping data')
        for x in ['train', 'val', 'test']:
            fmc[x] = fu.add_fliped_data(fmc[x])
            tof[x] = fu.add_fliped_data(tof[x])
            mask[x] = fu.add_fliped_data(mask[x])

    if 'size_limit' in cfg.keys():
        nlim = cfg['size_limit']
        np.random.seed(0)
        print('reducing data size')
        for x in ['train', 'val', 'test']:
            if fmc[x].shape[0] > nlim[x]:
                idx = np.random.permutation(nlim[x])
                fmc[x] = fmc[x][idx, ...]
                mask[x] = mask[x][idx, ...]
                tof[x] = tof[x][idx, ...]

    # agregar un dimension extra al final, que representa un solo canal
    print('transformar numpy a tensor')
    with tf.device('/cpu:0'):
        for x in ['train', 'val', 'test']:
            fmc[x] = tf.convert_to_tensor(np.expand_dims(fmc[x], axis=-1), dtype=np.float32)
            mask[x] = tf.convert_to_tensor(np.expand_dims(mask[x], axis =-1), dtype=np.float32)

    # PRA DATOS DE CILINDRO 40
    fmc['test'] = np.delete(fmc['test'], [4, 20, 36, 37, 43, 47, 50, 63, 82, 89], axis=0)
    tof['test'] = np.delete(tof['test'], [4, 20, 36, 37, 43, 47, 50, 63, 82, 89], axis=0)
    return fmc, tof, mask