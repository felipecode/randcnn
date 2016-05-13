
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
from time import time

def shufle(vec1,vec2,vec3):
  perm = np.arange(len(vec1))
  np.random.shuffle(perm)
  copy_vec1=list(vec1)
  copy_vec2=list(vec2)
  copy_vec3=list(vec3)
  for n in range(0,len(vec1)):
    copy_vec1[n] = vec1[perm[n]]
    copy_vec2[n] = vec2[perm[n]]
    copy_vec3[n] = vec3[perm[n]]
  for n in range(0,len(vec1)):
    vec1[n] = vec1[perm[n]]
    vec2[n] = vec2[perm[n]]
    vec3[n] = vec3[perm[n]]

  


# We don't need this actually, just one folder for each object. We change directly the folder name

class DataSet(object):
  def __init__(self, images_names, positive_window_names ,negative_window_names,input_size,network_size):
    self._num_examples = len(images_names)
    self._images_names = images_names
    self._negative_window_names = negative_window_names
    self._positive_window_names=positive_window_names
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._positive_percentage = 0.25
    self._input_size= input_size
    self._network_size = network_size
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
    start = self._index_in_epoch

    """ I decided to go one batch per image """

    self._index_in_epoch += 1

    if self._index_in_epoch >= self._num_examples:
      # Finished epoch
      print 'end epoch'
      self._epochs_completed += 1
      # Shuffle the data
      """ Shufling all the Images with a single permutation """
      shufle(self._images_names,self._negative_window_names,self._positive_window_names)
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    #if batch_size >  (self._num_examples - self._index_in_epoch):

    #  batch_size = self._num_examples - self._index_in_epoch


    images = np.empty((batch_size, self._network_size[0], self._network_size[1],self._network_size[2]))
    #if len(self._output_size) > 2:
    #  labels = np.empty((batch_size, self._output_size[0], self._output_size[1],self._output_size[2]))
    #else:
    #  labels = np.empty((batch_size, self._output_size[0], self._output_size[1]))
    labels = []

    

    for n in range(batch_size):

      """ READ THE IMAGE """
      full_image = self.read_image(self._images_names[start+n])
      #print 'leu'
      #print self._images_names[start+n]
      #print 'shape'
      #print full_image.shape
      """ GET THE window """
      if random.randint(1, 100) > 100*self._positive_percentage or (os.stat(self._positive_window_names[start+n]).st_size == 0):
        f = open(self._negative_window_names[start+n],'r')
        scale,x1,x2,y1,y2 = self.random_line(f)
        images[n] = self.extract_window(full_image,float(scale),int(x1),int(x2),int(y1),int(y2))
        f.close()
        labels.append(0)
      else:
        f = open(self._positive_window_names[start+n],'r')
        scale,x1,x2,y1,y2 = self.random_line(f)
        images[n] = self.extract_window(full_image,float(scale),int(x1),int(x2),int(y1),int(y2))
        f.close()
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
    self.im_names  = glob.glob(path + "/*.png")
    
    self.pwindow_files = glob.glob(path_windows_p + "/*.txt")

    self.nwindow_files = glob.glob(path_windows_n + "/*.txt")
    #self.im_labels = glob.glob(path + "/*.jpg")


    #self.im_names_labels = glob.glob(path_truth + "/*.jpg")

    #print len(self.im_names)
    #print len(self.pwindow_files)
    #print len(self.nwindow_files)

    """ Shufling all the Images with a single permutation"""
    shufle(self.im_names,self.pwindow_files,self.nwindow_files)
    


    #self.im_names_val = glob.glob(path_val + "/*.jpg")
    

    #self.im_names_val_labels = glob.glob(path_val_truth + "/*.jpg")
    self.train = DataSet(self.im_names, self.pwindow_files, self.nwindow_files,input_size,network_size)

    """ Take part of it for validation, lets check"""

    #self.validation = DataSet(self.im_names_val, self.im_names_val_labels,input_size,output_size)

  def getNImagesDataset(self):
    return len(self.im_names)

  def getNImagesValiton(self):
    return len(self.im_names_val)