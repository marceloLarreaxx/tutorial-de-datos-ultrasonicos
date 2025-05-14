import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import utils
import pickle
from imag2D.pintar import arcoiris_cmap
import imag3D.CNN_superficie.cnnsurf_funcs as fu
from keras.optimizers import Adam
import imag3D.CNN_superficie.keras_unet_mod.models.custom_vnet as cu
from imag3D.CNN_superficie.dataset.merge_datasets import merge
from imag3D.CNN_superficie.dataset.merge_datasets import merge_2
import imag3D.CNN_superficie.cnnsurf_plot as cnnplot
from imag3D.CNN_superficie.loss_funcs import load_segmentation_losses
import tensorflow as tf
from importlib import reload
import time
plt.ion()

reload(cu)
# reload(fu)

cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')
fmc, tof, mask = merge(cfg, decimar=cfg['decimar'], dims=[8,16])  # crear los datasets
# fmc, tof, mask = merge(cfg, decimar=cfg['decimar'], dims=[11,11])  # crear los datasets

n_out_1_metric = tf.keras.metrics.MeanMetricWrapper(fu.n_out_1, name='n_out_1')
n_out_2_metric = tf.keras.metrics.MeanMetricWrapper(fu.n_out_2, name='n_out_2')
# defina la metrica de error de indice
idx_error_metric = tf.keras.metrics.MeanMetricWrapper(fu.idx_error, u=0.5, name='idx_error')

print('\n\n Cargando Modelos FT:\n')
models_ft = {}
for m in  cfg['ft_models']:
    print(cfg['ft_model_path'] + m)
    model_ft = tf.keras.models.load_model(cfg['ft_model_path'] + m,
                custom_objects={'tversky_loss': tf.keras.utils.get_custom_objects()['Custom>tversky loss'],
                'idx_error': tf.keras.utils.get_custom_objects()['Custom>idx_error'],
                  'n_out_1': tf.keras.utils.get_custom_objects()['Custom>n_out_1'],
                  'n_out_2': tf.keras.utils.get_custom_objects()['Custom>n_out_2']})

    if m == "\Model_8x16.h5":
        model_ft.compile(optimizer=Adam(), loss=fu.tversky_loss,
                      # loss='binary_crossentropy', #loss=fu.WeightedBinaryCrossEntropy((1, 1)), #
                      metrics=[tf.keras.metrics.BinaryAccuracy(), idx_error_metric, n_out_1_metric, n_out_2_metric])
    models_ft[m] = model_ft


#---------------------- FUNCIONES -------------------------------#
def umbralizar(u, peco_interval, model_i,  window_num=10):
    """Calcula el indice de primer cruce de umbral sobre el test dataset"""
    n_test = fmc['test'].shape[0]
    idx = []
    # aplicar umbral a resultados del modelo
    for i in range(n_test):
        x = fmc['test'][i, :, :, :, 0]
        x = model_i(np.expand_dims(x, axis=0)).numpy()  # MAKING THE PREDICTION ON THE INPUT DATA
        x = x[0, :8, :16, :, 0].reshape((128, -1))  # here I have 128 A-scans
        # x = x[0, :11, :11, :, 0].reshape((121, -1))
        idx.append(utils.first_thr_cross(x, peco_interval, u, window_num))

    return np.array(idx)

def get_outlier(idx_list, e_max=40):
    print(idx_list.shape)
    n_out1 = 0
    n_out2 = 0
    errors = idx_list[:, :].flatten() - tof['test'].flatten()
    N = errors.shape[0]
    for i in range(len(errors)):
        if idx_list[:, :].flatten()[i] == 0:
            n_out1+=1
        elif abs(errors[i])>e_max:
            n_out2+=1
    n_out2_rate = (n_out2/N)*100

    return n_out1, n_out2, n_out2_rate, N

def plot_error_boxplot(error_data, outlier_counts):
    """
    Función para generar un diagrama de caja (box plot) de los errores de índice de distintas soluciones.

    Args:
    - error_data: Un diccionario donde las claves son los nombres de las soluciones y los valores son listas de errores de índice.
    - outlier_counts: Un diccionario donde las claves son los nombres de las soluciones y los valores son tuplas que contienen el número de outliers de cada tipo.
    """
    # Crear una lista de errores para cada solución
    errors = list(error_data.values())

    # Crear una lista de nombres de soluciones
    N = outlier_counts[list(outlier_counts.keys())[0]][-1]
    solution_names = list(error_data.keys())
    n_out_rates = [round(outlier_counts[ky][2], 2) for ky in outlier_counts.keys()]
    solution_names2 = [solution_names[i] + '\nn_out2: ' + str(n_out_rates[i]) + '%' for i in range(len(n_out_rates))]
    plt.ion()
    # Generar el box plot principal

    fig, axs = plt.subplots(2,figsize=(10, 12), sharex=True, sharey=False)

    axs[0].boxplot(errors, labels=solution_names2,vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     showfliers=False)

    #axs[0].set_title('Box Plot de Errores de Índice por Solución')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Index Error')
    axs[0].grid(True)
    axs[1].set_title('N = ' + str(N))
    # axs[1].set_xticklabels(solution_names2, fontsize=9)

    # Obtener el número total de outliers de cada tipo para cada solución
    total_outliers_type_1 = [outlier_counts[solution][0] for solution in solution_names]
    total_outliers_type_2 = [outlier_counts[solution][1] for solution in solution_names]

    # Añadir subplot encima del box plot principal para mostrar outliers
    # Añadir los marcadores 'X' en el segundo subplot para outliers de tipo 1
    axs[1].scatter(range(1,len(solution_names)+1), total_outliers_type_1, marker='X', color='red', label='Outliers Type 1')
    axs[1].set_ylabel('Number of Outliers')
    axs[1].grid(True)
    axs[1].legend()


    # Añadir los marcadores '0' en el segundo subplot para outliers de tipo 2
    axs[1].scatter(range(1,len(solution_names)+1), total_outliers_type_2, marker='o', color='blue', label='Outliers Type 2')
    axs[1].set_xlabel('Fine-Tuned layers')
    axs[1].legend()

    #plt.tight_layout()
    plt.show()


def plot_error_histogram_multi(errors_dict, bin_width=5, ylim_max=70000):
    """
    Función para generar un histograma de los errores de índice para varias soluciones, con el número de errores en el eje y y bins fijos en el eje x.

    Args:
    - errors_dict: Un diccionario donde las claves son los nombres de las soluciones y los valores son listas de errores de índice correspondientes a cada solución.
    """
    num_solutions = len(errors_dict)
    # bin_width = 5  # Ancho del bin en el eje x

    # Crear subplots para cada solución
    fig, axs = plt.subplots(num_solutions, figsize=(8, 2*num_solutions), sharex=True)


    num_bins = int((40 - (-40)) / bin_width) + 1

    # Generar bins para el histograma
    bins = np.linspace(-40, 40, num_bins + 1)  # Bins para valores dentro del rango [-40, 40]


    for i, (solution, errors) in enumerate(errors_dict.items()):
            # Calcular los valores dentro y fuera del rango
            in_range = [error for error in errors if -40 <= error <= 40]
            out_of_range_left = [error for error in errors if error < -40]
            out_of_range_right = [error for error in errors if error > 40]

            # Generar el histograma en el subplot correspondiente
            counts, bins, _ = axs[i].hist(in_range, bins=bins, color='blue', edgecolor='black', alpha=0.7)
            axs[i].set_title(f'- {solution}')
            axs[i].set_ylabel('Errores')
            axs[i].grid(True)

            # Añadir barras para valores fuera del rango
            if out_of_range_left:
                axs[i].bar(-45, len(out_of_range_left), width=5, color='red', label='Outliers left')
            if out_of_range_right:
                axs[i].bar(45, len(out_of_range_right), width=5, color='red', label='Outliers right')

            # Anotar el número total de outliers en el gráfico
            total_outliers = len(out_of_range_left) + len(out_of_range_right)
            if total_outliers > 0:
                axs[i].text(0.85, 0.85, f'Total de Outliers: {total_outliers}', transform=axs[i].transAxes,
                            fontsize=10, ha='center', va='center', color='red')
            axs[i].set_ylim((0, ylim_max))

    # Ajustar los ejes x
    plt.xticks(np.arange(-40, 41, bin_width))

    # Añadir etiqueta y título común
    axs[-1].set_xlabel('Error de índice')
    #plt.title('Index error per solution')

    plt.tight_layout()
    plt.show()

#---------------------- APLICANDO FUNCIONES -------------------------------#
idx_vnet_fts = {}; n_outliers = {}; errors_ft = {}; errors_ft_filt = {}
for k in models_ft.keys():
    model_k = models_ft[k]
    idx_vnet = umbralizar(0.5, [0, 1000], model_k)
    idx_vnet_fts[k] = idx_vnet
    n_out1_vnet, n_out2_vnet, n_out_vnet_rate, N = get_outlier(idx_vnet[:, :, 0])  # Getting outliers
    n_outliers[k] = (n_out1_vnet, n_out2_vnet, n_out_vnet_rate, N)
    err_vnet = (idx_vnet[:, :, 0] - tof['test'].reshape(-1, 128)).flatten()  # errors of predictions
    # err_vnet = (idx_vnet[:, :, 0] - tof['test'].reshape(-1, 121)).flatten()  # errors of predictions
    errors_ft[k] = err_vnet
    errors_ft_filt[k] = err_vnet[np.abs(err_vnet)<40]

# kys_slice = ['ft_15.h5', 'ft_16.h5', 'ft_17.h5', 'ft_18.h5']
# slice_err = {k:errors_ft[k] for k in kys_slice}

plot_error_boxplot(errors_ft_filt, n_outliers)

#--------------------------------------------------------------------------#