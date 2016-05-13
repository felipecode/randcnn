
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
import gzip
import os
import numpy as np
import Image, colorsys
from scipy import misc
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import glob


import matplotlib.pyplot as plt
import time

# def shufle(vec1,vec2,vec3):
#   perm = np.arange(len(vec1))
#   np.random.shuffle(perm)
#   copy_vec1=list(vec1)
#   copy_vec2=list(vec2)
#   copy_vec3=list(vec3)
#   for n in range(0,len(vec1)):
#     copy_vec1[n] = vec1[perm[n]]
#     copy_vec2[n] = vec2[perm[n]]
#     copy_vec3[n] = vec3[perm[n]]
#   for n in range(0,len(vec1)):
#     vec1[n] = vec1[perm[n]]
#     vec2[n] = vec2[perm[n]]
#     vec3[n] = vec3[perm[n]]



  


# We don't need this actually, just one folder for each object. We change directly the folder name

class DataSet(object):
  def __init__(self, path,positive_window_names ,negative_window_names,input_size,network_size):
    self._num_examples = len(positive_window_names) +  len(negative_window_names) 
    #self._images_names = images_names
    self._path = path
    self._negative_window_atributes = negative_window_names
    self._positive_window_atributes= positive_window_names
    self._epochs_completed = 0
    self._index_in_epoch_positive = 0
    self._index_in_epoch_negative = 0
    self._positive_percentage = 0.25
    self._input_size= input_size
    self._network_size = network_size
    self._loaded_full_images = np.empty(0)
    self._loaded_full_images_names = []
    #self._output_size=output_size

  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed


  def extract_window(self,image,scale,x1,x2,y1,y2):

    #print image
    #print scale

    #print x1
    #print x2
    #print y1
    #print y2
    
    #print image.shape
    image_rscale = misc.imresize(image, scale, interp='bilinear', mode=None)


    window = image_rscale[x1:x2,y1:y2,:]

    
    """ TODO: MAYBE CHANGE .... for now we are going to rescale the stuff """

    window = misc.imresize(window, self._network_size, interp='bilinear', mode=None)
    return window



  def read_image(self,image_name):
    image =Image.open(image_name).convert('RGB')
    #image =  image.resize((self._input_size[0], self._input_size[1]), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image

  def read_image_gray_scale(self,image_name):
    image =Image.open(image_name).convert('L')
    #image =  image.resize((self._input_size[0], self._input_size[1]), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image
  def random_line(self,afile):
    line = next(afile)
    for num, aline in enumerate(afile):
      if random.randrange(num + 2): continue
      line = aline
    return line.split()


  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start_neg = self._index_in_epoch_negative
    start_pos = self._index_in_epoch_positive
    """ I decided to go one batch per image """

    

    if (self._index_in_epoch_positive + self._index_in_epoch_negative) >= self._num_examples:
      # Finished epoch
      print 'end epoch'
      self._epochs_completed += 1
      # Shuffle the data
      """ Shufling all the Images with a single permutation """
      #shufle(self._images_names,self._negative_window_names,self._positive_window_names)
      # Start next epoch
      start = 0
      #self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    #if batch_size >  (self._num_examples - self._index_in_epoch):

    #  batch_size = self._num_examples - self._index_in_epoch

    #self._index_in_epoch += batch_size
    
    images = np.empty((batch_size, self._network_size[0], self._network_size[1],self._network_size[2]))
    #images = np.empty((batch_size, self._input_size[0], self._input_size[1],self._input_size[2]))


    labels = []

    # We need fast memoization of images and also considering certain scales.... Use the entire memory beforehand with images in their scale.


    for n in range(batch_size):

      """ Randomly set to read a positive or a negative """
      if random.randint(1, 100) > 100*self._positive_percentage or (start_pos+self._index_in_epoch_positive) > len(self._positive_window_atributes):
       
        image_name,scale,x1,x2,y1,y2  = self._negative_window_atributes[start_neg+self._index_in_epoch_negative].split()
        #if image_name 

        self._index_in_epoch_negative += 1 
        start_time = time.time()
        full_image = self.read_image(self._path + '/' + image_name + '.png')
        duration = time.time() - start_time 
        print 'readImageDuration ', duration 
        start_time = time.time()
        images[n] = self.extract_window(full_image,float(scale),int(x1),int(x2),int(y1),int(y2))
        duration = time.time() - start_time 
        print 'cutwindowntime ', duration 

        labels.append(0)
      else:

        image_name,scale,x1,x2,y1,y2  = self._positive_window_atributes[start_pos+self._index_in_epoch_positive].split()
        self._index_in_epoch_positive += 1 
        start_time = time.time()
        full_image = self.read_image(self._path + '/' + image_name + '.png')
        duration = time.time() - start_time 
        print 'readImageDuration ', duration 
        start_time = time.time()
        images[n] = self.extract_window(full_image,float(scale),int(x1),int(x2),int(y1),int(y2))
        duration = time.time() - start_time 
        print 'cutwindowntime ', duration 

        labels.append(1)

      #if len(self._output_size) > 2:
      #  labels[n] = self.read_image(self._labels_names[start+n])
      #else:
        
    return images, labels


class DataSetManager(object):

  def __init__(self, path, path_windows_p, path_windows_n, input_size,network_size):
    self.input_size = input_size
    #self.output_size =output_size
    """ Get all the image names for training images on a path folder """
    #self.im_names  = glob.glob(path + "/*.png")
    

    f= open(path_windows_p + "/p1.txt")


    self.p_windows_list = f.readlines()
    f= open(path_windows_n + "/n1.txt")

    self.n_windows_list = f.readlines()
    #self.im_labels = glob.glob(path + "/*.jpg")


    #self.im_names_labels = glob.glob(path_truth + "/*.jpg")

    #print len(self.im_names)
    #print len(self.pwindow_files)
    #print len(self.nwindow_files)

    """ Shufling all the Images with a single permutation"""
    #shufle(self.im_names,self.pwindow_files,self.nwindow_files)
    
    #random.shuffle(self.p_windows_list)
    #random.shuffle(self.n_windows_list)

    #self.im_names_val = glob.glob(path_val + "/*.jpg")
    

    #self.im_names_val_labels = glob.glob(path_val_truth + "/*.jpg")
    self.train = DataSet(path, self.p_windows_list, self.n_windows_list,input_size,network_size)

    """ Take part of it for validation, lets check"""

    #self.validation = DataSet(self.im_names_val, self.im_names_val_labels,input_size,output_size)

  def getNImagesDataset(self):
    return len(self.im_names)

  def getNImagesValiton(self):
    return len(self.im_names_val)