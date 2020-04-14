#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
png_list = os.listdir('./VOCdevkit/VOC2012/SegmentationObject/')


# In[2]:


png_list


# In[3]:


random.shuffle(png_list)


# In[4]:


train_png = png_list[:len(png_list)//2]
val_png = png_list[len(png_list)//2:]


# In[5]:


for x in train_png:
    os.rename('./VOCdevkit/VOC2012/SegmentationObject/' + x, 'VOCdevkit/VOC2012/train_masks/' + x)
    os.rename('./VOCdevkit/VOC2012/JPEGImages/' + x.replace('.png','.jpg'), 'VOCdevkit/VOC2012/train_frames/' + x.replace('.png','.jpg'))


# In[6]:


for x in val_png:
    os.rename('./VOCdevkit/VOC2012/SegmentationObject/' + x, 'VOCdevkit/VOC2012/val_masks/' + x)
    os.rename('./VOCdevkit/VOC2012/JPEGImages/' + x.replace('.png','.jpg'), 'VOCdevkit/VOC2012/val_frames/' + x.replace('.png','.jpg'))

