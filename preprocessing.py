
#%%
from os import listdir
import numpy as np
from numpy import savez_compressed
import os
import re
from os.path import join
#%%

def create_dataset(tom1_path, tom2_path, dataset_name='dataset_inicial.npz'):
    """
    Esta función procesa archivos de dos carpetas distintas (tom1 y tom2), combinando sus partes reales e imaginarias
    para formar tomogramas, y se calcula la intensidad.
    Finalmente, guarda estos datos en un archivo .npz.

    Entradas:
    - tom1_path (str): Ruta del directorio que contiene los archivos reales e imaginarios de tom1.
    - tom2_path (str): Ruta del directorio que contiene los archivos reales e imaginarios de tom2.
    - dataset_name (str, opcional): Nombre del archivo .npz donde se guardarán los datos. Por defecto es 'dataset_inicial.npz'.
    
    Salidas:
    - Un archivo .npz que contiene dos arrays: inputs_intensidad y targets_intensidad, correspondientes
      a las entradas y objetivos del modelo.
    """
    
    tom1_real_files = sorted([file for file in os.listdir(tom1_path) if file.startswith('tom1_real')])
    tom1_imag_files = sorted([file for file in os.listdir(tom1_path) if file.startswith('tom1_imag')])
    tom2_real_files = sorted([file for file in os.listdir(tom2_path) if file.startswith('tom2_real')])
    tom2_imag_files = sorted([file for file in os.listdir(tom2_path) if file.startswith('tom2_imag')])
    
    inputs_ampli = []
    targets_ampli = []
    const = 1e-9  

    for filename_tom1_real, filename_tom1_imag, filename_tom2_real, filename_tom2_imag in zip(tom1_real_files, tom1_imag_files, tom2_real_files, tom2_imag_files):
        z, x, y = [int(re.search(f'{dim}=([0-9]+)', filename_tom1_real).group(1)) for dim in ['z', 'x', 'y']]

        tom1_real = np.fromfile(join(tom1_path, filename_tom1_real), dtype='float32').reshape((z, x, y), order='F')
        tom1_imag = np.fromfile(join(tom1_path, filename_tom1_imag), dtype='float32').reshape((z, x, y), order='F')
        
        tom2_real = np.fromfile(join(tom2_path, filename_tom2_real), dtype='float32').reshape((z, x, y), order='F')
        tom2_imag = np.fromfile(join(tom2_path, filename_tom2_imag), dtype='float32').reshape((z, x, y), order='F')

        tom1 = tom1_real + 1j * tom1_imag
        tom2 = tom2_real + 1j * tom2_imag

        for i in range(y):
            transformed_input = 20 * np.log10(np.abs(tom2[:, :, i]) + const)
            transformed_target = 20 * np.log10(np.abs(tom1[:, :, i]) + const)
            inputs_ampli.append(transformed_input)
            targets_ampli.append(transformed_target)

    inputs_amplitud = np.expand_dims(np.array(inputs_ampli, dtype='float32'), -1)
    targets_amplitud = np.expand_dims(np.array(targets_ampli, dtype='float32'), -1)

    savez_compressed(dataset_name, inputs_amplitud=inputs_amplitud, targets_amplitud=targets_amplitud)
    print('Dataset guardado como:', dataset_name)


def create_combined_dataset(dataset_path, test_ratio=0.2, output_filename='dataset_nos.npz', num_samples_to_remove=5):
    data = np.load(dataset_path)
    dominio_1 = data['inputs_amplitud']
    dominio_2 = data['targets_amplitud']
    
    if num_samples_to_remove > 0 and num_samples_to_remove < len(dominio_1):
        dominio_1 = np.delete(dominio_1, np.s_[:num_samples_to_remove], axis=0)
    
    np.random.shuffle(dominio_1)
    np.random.shuffle(dominio_2)
    
    test_size_1 = int(len(dominio_1) * test_ratio)
    test_size_2 = int(len(dominio_2) * test_ratio)
    
    test_dominio_1 = dominio_1[:test_size_1]
    train_dominio_1 = dominio_1[test_size_1:]
    test_dominio_2 = dominio_2[:test_size_2]
    train_dominio_2 = dominio_2[test_size_2:]
    
    np.savez_compressed(output_filename, train_dominio_1=train_dominio_1, test_dominio_1=test_dominio_1, train_dominio_2=train_dominio_2, test_dominio_2=test_dominio_2)
    print(f'Dataset combinado guardado como: {output_filename}')


def create_dataset_solap(path, dataset_name='dataset_intensidad.npz', 
                         solap_initial=20, maxSolap_multiplier=0.6, porcentaje=3.4):
    file_list = os.listdir(path)
    tom1_real_files = sorted([file for file in file_list if file.startswith('tom1_real')])
    tom1_imag_files = sorted([file for file in file_list if file.startswith('tom1_imag')])
    tom1_data = {}

    for filename_tom1_real, filename_tom1_imag in zip(tom1_real_files, tom1_imag_files):
        z, x, y = [int(re.search(f'{dim}=([0-9]+)', filename_tom1_real).group(1)) for dim in ['z', 'x', 'y']]
        tom1_real = np.fromfile(join(path, filename_tom1_real), dtype='single').reshape((z, x, y), order='F')
        tom1_imag = np.fromfile(join(path, filename_tom1_imag), dtype='single').reshape((z, x, y), order='F')
        tom1 = tom1_real + 1j * tom1_imag
        filename_tom1 = filename_tom1_real.replace('_real', '')
        tom1_data[filename_tom1] = tom1

    solapValues = []
    solap = solap_initial
    maxSolap = z * maxSolap_multiplier

    while solap <= maxSolap:
        solapValues.append(solap)
        solap = int(solap * porcentaje)

    solapValues = [int(value) for value in solapValues]

    inputs_ampli, targets_ampli = [], []

    for filename, tom1 in tom1_data.items():
        for solap in solapValues:
            z, x, y = tom1.shape
            newTom = np.zeros((2*z, x, y), dtype=complex)
            newTom2 = np.zeros((2*z, x, y), dtype=complex)

            newTom[solap:512+solap, :, :] = tom1
            newTom2[solap:512+solap, :, :] = tom1
            newTom[512-solap:1024-solap, :, :] += np.flip(tom1, axis=0)

            center = newTom2.shape[0] // 2
            newTom2_cut = newTom2[center-256:center+256, :, :]
            newTom_cut = newTom[center-256:center+256, :, :]
            const = 0.0000001

            for i in range(newTom_cut.shape[2]):
                transformed_input = 20 * np.log10(np.abs(newTom_cut[:, :, i]) + const)
                transformed_target = 20 * np.log10(np.abs(newTom2_cut[:, :, i]) + const)
                inputs_ampli.append(transformed_input)
                targets_ampli.append(transformed_target)

    inputs_amplitud = np.expand_dims(np.array(inputs_ampli, dtype='float32'), -1)
    targets_amplitud = np.expand_dims(np.array(targets_ampli, dtype='float32'), -1)

    savez_compressed(dataset_name, inputs_amplitud=inputs_amplitud, targets_amplitud=targets_amplitud)
    print('Dataset guardado como:', dataset_name)


create_dataset(r'C:\Users\Vale\Desktop\tdg\experimentales_tom1',
                r'C:\Users\Vale\Desktop\tdg\experimentales_tom2','dataset_inicial.npz')

create_dataset_solap(r'C:\Users\Vale\Desktop\tdg\experimentales_tom1', 'dataset_solap.npz')

create_combined_dataset('dataset_inicial.npz', 0.2, 'dataset_cycle.npz', num_samples_to_remove=3)



#########################################

def tom_completo(path):
    file_list = os.listdir(path)
    tom_real_files = sorted([file for file in file_list if file.startswith('tom1_real')])
    tom_imag_files = sorted([file for file in file_list if file.startswith('tom1_imag')])

    tom_data = {}

    for filename_tom_real, filename_tom_imag in zip(tom_real_files, tom_imag_files):
        z, x, y = [int(re.search(f'{dim}=([0-9]+)', filename_tom_real).group(1)) for dim in ['z', 'x', 'y']]

        tom_real = np.fromfile(join(path, filename_tom_real), 'single').reshape((z, x, y), order='F')
        tom_imag = np.fromfile(join(path, filename_tom_imag), 'single').reshape((z, x, y), order='F')

        tom = tom_real + 1j * tom_imag
        del tom_real, tom_imag
        filename_tom1 = filename_tom_real.replace("_real", "")
        
        tom_data[filename_tom1] = tom
    return tom_data

def log(tom_data):
    info = {}
    for key, tom in tom_data.items():
        epsilon = 1e-7
        tom = 20 * np.log10(np.abs(tom) + epsilon)
        norm = (tom - np.min(tom)) / (np.max(tom) - np.min(tom))
        min_val = np.min(tom) 
        max_val = np.max(tom)
        info[key] = (norm, min_val, max_val)
    return info



def invlog(info):
    tom_original = {}
    for key, value in info.items():
        norm, min_val, max_val = value
        tom = (norm * (max_val - min_val)) + min_val
        tom = 10 ** (tom / 20)
        tom_original[key] = (tom.real, tom.imag)
    return tom_original