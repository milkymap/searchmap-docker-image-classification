import cv2 

import json 
import pickle
import pandas as pd 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

from os import path 
from PIL import Image 
from glob import glob
from time import time 

from collections import Counter
from torchvision import transforms as T  
from torchvision.utils import make_grid
from torchvision import models as models_downloader

from rich.progress import track 
from libraries.log import logger 

map_serializers = {json: ('r', 'w'), pickle: ('rb', 'wb')}

def is_valid(var_value, var_name, var_type=None):
    if var_value is None:
        raise ValueError(f'{var_name} is not defined | please look the helper to see available env variables')
    if var_type is not None:
        if not op.attrgetter(var_type)(path)(var_value):
            raise ValueError(f'{var_name} should be a valid file or dir')

def measure(func):
    @ft.wraps(func)
    def _measure(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            duration = end_ if end_ > 0 else 0
            logger.debug(f"{func.__name__:<20} total execution time: {duration:04d} ms")
    return _measure

def read_image(path2image):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    cv_image = cv2.resize(cv_image, (256, 256))
    return cv_image 

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def th2cv(th_image):
    red, green, blue = th_image.numpy() # unpack
    return cv2.merge((blue, green, red))

def merge_images(path2images):
    acc = []
    for img_path in path2images:
        cv_image = read_image(img_path)
        th_image = cv2th(cv_image)
        acc.append(th_image)
    
    th_image = make_grid(acc, 4)
    return th2cv(th_image)

def pull_files(path2directory, extension='*'):
    all_paths = sorted(glob(path.join(path2directory, extension)))
    return all_paths 

def serialize(data, location, serializer):
    modes = map_serializers.get(serializer, None)
    if modes is None:
        raise Exception('serializer has to be [pickle or json]')
    with open(location, mode=modes[1]) as fp:
        serializer.dump(data, fp)
        logger.success(f'data was dumped at {location}')
    
def deserialize(location, serializer):
    modes = map_serializers.get(serializer, None)
    if modes is None:
        raise Exception('serializer has to be [pickle or json]')
    with open(location, mode=modes[0]) as fp:
        data = serializer.load(fp)
        logger.success(f'data was loaded from {location}')
    return data 

def read_description(csv_filename):
    try:
        dataframe = pd.read_csv(csv_filename, sep=',')
        reindexed_dataframe = dataframe.set_index(dataframe.columns[0])
        zipped_labels = list(map(list,reindexed_dataframe.to_records(index=False)))
        nb_labels = len(dataframe.columns)

        print(dataframe)

        images = list(reindexed_dataframe.index)
        map_image2labels = dict(zip(images, zipped_labels))
        index2label = list(dataframe.columns)[1:]
        return {
            'map_image2labels': map_image2labels,
            'index2label': index2label, 
            'nb_labels': nb_labels
        }
    except Exception as e:
        logger.error(e)
        raise ValueError(
        """
        can not parse csv_filename | columns schema : images, class0, class1, ..., classN
        """
    )

def vectorize(input_tensor, vectorizer, device='cpu'):
    embedding = vectorizer(input_tensor[None, ...].to(device)).cpu().numpy()
    return np.ravel(embedding)  # flatten 

def vectorize_images(map_image2labels, image_paths, path2vectorizer, worker_id, barrier, readyness, queue_, device='cpu'):
    try:
        vectorizer = load_vectorizer(path2vectorizer, device)
        nb_images = len(image_paths)
        
        logger.debug(f'{worker_id:03d} is ready to process {nb_images:03d}')
        barrier.wait()  # wait other workers to be ready 
        readyness.wait()  # wait to be notifed by the parent process 

        queue_.put({'worker_id': worker_id, 'event': 'join', 'content': None})
        for path2image in image_paths:
            try:
                _, image_filename = path.split(path2image)
                labels = map_image2labels[image_filename]
                positive_label = labels.index(1)
                bgr_image = read_image(path2image)
                tensor_image = cv2th(bgr_image)
                tensor_image = prepare_image(tensor_image)
                with th.no_grad():
                    extracted_features = vectorize(tensor_image, vectorizer, device)
                    queue_.put({
                        'worker_id': worker_id, 
                        'event': 'step', 
                        'content': (extracted_features, positive_label)
                    })
            except Exception as e:
                logger.error(e)
        # end ...!
    except KeyboardInterrupt:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        queue_.put({'worker_id': worker_id, 'event': 'stop', 'content': None})
        logger.debug(f'{worker_id:03d} has put its result into the queue memory')
        
def scoring(fingerprint, fingerprint_matrix):
    scores = fingerprint @ fingerprint_matrix.T 
    X = np.linalg.norm(fingerprint)
    Y = np.linalg.norm(fingerprint_matrix, axis=1)
    W = X * Y 
    weighted_scores = scores / W 
    return weighted_scores

def top_k(weighted_scores, k=16):
    scores = th.as_tensor(weighted_scores).float()
    _, indices = th.topk(scores, k, largest=True)
    return indices.tolist()

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)

def load_vectorizer(path2vectorizer, device='cpu'):
    if path.isfile(path2vectorizer):
        logger.debug('the model was found ... it will be loaded')
        vectorizer = th.load(path2vectorizer, map_location=device) 
    else:
        logger.debug('the model was not found, download will start')
        _, vectorizer_filename = path.split(path2vectorizer)
        vectorizer_name, _ = vectorizer_filename.split('.') 
        try:
            vectorizer = op.attrgetter(vectorizer_name)(models_downloader)(pretrained=True)
            th.save(vectorizer, path2vectorizer)
        except Exception as e:
            raise ValueError(f'{vectorizer_name} is not a valid model. check torchvision models list')
    
    vectorizer = nn.Sequential(*list(vectorizer.children())[:-1]).eval()
    vectorizer.eval()
    for prm in vectorizer.parameters():
        prm.requires_grad = False
    vectorizer.to(device)
    return vectorizer.eval()




    

        







