### Neural Networks

- Common use cases:
- Binary classification, Multiclass classification and scalar regression.

- The network, composed of *layers* that are chained together, maps the input data to predictions. 
- The *loss function* then compares these predictions to the targets, producing a loss value: a measure
of how well the network’s predictions match what was expected. 
- The *optimizer* uses this **loss value** to update the network’s weights.

**Layers**

- Data-processing module that takes as input one or more tensors and that outputs one or more tensors.
- Some layers are stateless, but more frequently layers have a state:
- Layer’s *weights*, one or several tensors learned with *stochastic gradient descent*, which together contain the network’s *knowledge*.
- Simple vector data, stored in 2D tensors of shape (samples, features), is often processed by *densely connected* layers. (**fully connected**)
- Sequence data, stored in 3D tensors of shape (samples, timesteps, features), is typically processed by recurrent layers such as an **LSTM layer**.
- Image data, stored in 4D tensors, is usually processed by **2D convolution layers (Conv2D)**.
- Building deep-learning models in Keras is done by clipping together compatible layers to form useful data-transformation pipelines.
- *Layer compatibility* here refers specifically to the fact that every layer will only accept input tensors of a certain shape and will return output tensors of a certain
shape.
- In *Keras* the layers you add to your models are dynamically built to match the shape of the incoming layer.
    - Second layer automatically inferred its input shape as being the output shape of the layer that came before