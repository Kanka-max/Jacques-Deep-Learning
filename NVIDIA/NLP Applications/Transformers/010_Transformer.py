#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.nvidia.com/dli"> <img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/> </a>

# # 1.0 How the Transformer Architecture Changed NMT
# In this notebook, we'll take a big-picture look at the Transformer architecture and model and its impact on NMT.

# **[1.1 Neural Machine Translation (NMT)](#1.1-Neural-Machine-Translation-(NMT))<br>**
# **[1.2 Transformer Architecture Overview](#1.2-Transformer-Architecture-Overview)<br>**
# **[1.3 What is Attention?](#1.3-What-is-Attention?)<br>**
# **[1.4 Transformer Model Overview](#1.4-Transformer-Model-Overview)<br>**

# # 1.1 Neural Machine Translation (NMT)

# NMT is an end-to-end learning approach for automated translation of languages. It has the potential to overcome many of the weaknesses of conventional phrase-based statistical machine translation systems. Machine translation has achieved great success in the last few years, however it remains a challenging task.
# 
# Let's look at the recent history of NMT and language understanding models.

# <center><img src="images/nlp.png" width="800"></center>

# The Transformer architecture, introduced in 2017, provided a significant milestone in NMT.  In addition, the Transformer became the basis for the many models developed afterwards, that are used for a range of NLP tasks, which is why we are taking a deep dive now into how it all works.

# # 1.2 Transformer Architecture Overview

# The Transformer is a competitive alternative to the models using Recurrent Neural Networks (RNNs) for a range of sequence modeling tasks. That's because the Transformer addresses a significant shortcoming of RNNs: their computations are inherently sequential. RNNs must read one word at a time, performing multiple steps to make decisions about the relevance of nearby words to meaning. 
# 
# In contrast, Transformers rely *entirely* on self-attention mechanisms that directly model relationships between all words in a sentence.  A vector is computed for each input symbol (such as a word) containing this context information.  The network is more easily parallelized, and thus more efficient. Transformer, with its self attention mechanism and feed-forward connections, has further advanced the field of NMT, both in terms of translation quality and speed of convergence.

# <center><img src="images/Transformer_architecture.png" width="400"></center>
# 
# Image credit: [Attention is all you need](https://arxiv.org/abs/1706.03762).

# In summary, the advantages of the Transformer over the RNN-based sequence-to-sequence (Seq2Seq) models/networks are:
# 1.	Transformer achieves parallelization by replacing recurrence with attention and encoding the position of each symbol within the input sequence. This results in shorter training time.
# 2.	Transformer reduces the number of sequential operations to relate two symbols from input/output sequences to a constant O(1) number of operations. Transformer achieves this with the attention mechanism that allows it to model dependencies regardless of their distance in input or output sentence.

# Generally, seq2seq models consist of an encoder and a decoder. The encoder takes the input sequence and maps it into a higher dimensional space, as an n-dimensional vector. This abstract vector is then fed into the decoder, which turns it into an output sequence. The output sequence can be in another language, symbols, or even a picture.

# <center><img src="images/enc_dec.png" width="400"></center>
# <center> Figure 2: Encoder-Decoder representation. </center>

# # 1.3 What is Attention?

# The backbone of the Transformer model is the "attention" mechanism. 
# 
# The intuition here is to think of attention in deep learning as an imitation of how a human might look at a visual scene. We don't typically scan everything in our view, but rather focus on the important features, depending on the context of the scene.  Similarly, in language we focus more on certain important words as they apply to other words, again based on context.
# 
# An attention mechanism looks at an input sequence and decides, at each step, which other parts of the sequence are important. Attention in deep learning can be interpreted as a vector of importance weights. In the example below, we see that “ball” has strong attention to both “tennis” and “playing”, but “tennis” and “dog” are weakly connected.

# <center><img src="images/attention1.png" width="600"></center>
# 
# <center> Figure 3: Hypotetical example for attention mechanism.

# # 1.4 Transformer Model Overview
# 
# Let's take a look at the basic Transformer model code, based on the [PyTorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) base class. The base model is shown in the `forward` method below. Data is passed through the encoder, and then through the decoder.

# In[1]:


import torch.nn as nn
class TransformerModel(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self._is_generation_fast = False
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out, padding_mask = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, padding_mask)
        return decoder_out


# <h2 style="color:green;">Congratulations!</h2>
# 
# You've learned that 
# * Transformer "transformed" NMT by removing the need for RNNs
# * Attention mechanisms are the key to the Transormer architecture
# * The model consists of an encoder and a decoder
# 
# You'll examine the encoder next - move on to [2.0 Tranformer Encoder](020_Encoder.ipynb).

# <a href="https://www.nvidia.com/dli"> <img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/> </a>
