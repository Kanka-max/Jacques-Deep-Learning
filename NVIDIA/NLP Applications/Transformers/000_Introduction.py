#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.nvidia.com/dli"> <img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/> </a>

# # Building Transformer-Based Natural Language Processing Applications
# ### Part 1: Machine Learning in NLP
# 
# In this lab, you will explore the concept of *attention*, and look at how it powers the Transformer architecture, which was introduced in the ["Attention is All You Need!"(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) paper. The Transformer architecture is the precursor to large-scale language models such as [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) and [Megatron](https://developer.nvidia.com/blog/language-modeling-using-megatron-a100-gpu/), which have provided a leap in accuracy for natural language processing (NLP) tasks and brought high-quality language-based services within the reach of companies across many industries.
# 
# The specific NLP task we'll focus on in this section is *neural machine translation (NMT)*.  We'll step through the fundamentals of the Transformer architecture, and use it to translate English sentences into German sentences. 
# 
# <center><img src="images/enc_dec.png" width="400"></center>
# <center> NMT </center>
# 
# Throughout the notebooks, we'll use the NVIDIA [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) Transformer GitHub repo. Please consult the repo README for the full usage guide.

# ## Table of Contents
# 
# 1. [How the Transformer Architecture Changed NMT](010_Transformer.ipynb)<br>
#     key is attention...<br>
#     You'll learn about:
#     - Neural Machine Translation
#     - Transformer Architecture
#     - Attention
#     - Transformer Model
# <br><br>
# 1. [Encoder](020_Encoder.ipynb)<br>
#     You'll learn about:
#     - Encoder Mechanics
#     - Embedding
#     - Positional Encoding
#     - Multi-Head Attention
# <br><br>
# 1. [Decoder](030_Decoder.ipynb)<br>
#     You'll learn about:
#     - Decoder Mechanics
#     - Masked Multi-Head Attention
# 

# ### JupyterLab
# For this hands-on lab, we use [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) to manage our environment.  The [JupyterLab Interface](https://jupyterlab.readthedocs.io/en/stable/user/interface.html) is a dashboard that provides access to interactive iPython notebooks, as well as the folder structure of our environment and a terminal window into the Ubuntu operating system. The first view you'll see includes a **menu bar** at the top, a **file browser** in the **left sidebar**, and a **main work area** that is initially open to the "Launcher" page. 
# 
# <img src="images/jl_launcher.png">
# 
# The file browser can be navigated just like any other file explorer. A double click on any of the items will open a new tab with its content.
# 
# The main work area includes tabbed views of open files that can be closed, moved, and edited as needed. 
# 
# The notebooks, including this one, consist of a series of content and code **cells**.  To execute code in a code cell, press `Shift+Enter` or the "Run" button in the menu bar above, while a cell is highlighted. Sometimes, a content cell will get switched to editing mode. Pressing `Shift+Enter` will switch it back to a readable form.
# 
# Try executing the simple print statement in the cell below.

# In[1]:


# Highlight this cell and click [Shift+Enter] to execute
print('This is just a simple print statement')


# <a href="https://www.nvidia.com/dli"> <img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/> </a>
