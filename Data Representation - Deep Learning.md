### Data Representation

- A scalar tensor has 0 axes. (xndim=0)
- The number of axes of a tensor is also called its *rank*.
- A 1D tensor is said to have exactly *one axis*.
- A vector can have five entries and so is called a *5-dimensional* vector. Don’t confuse a *5D vector* with a *5D tensor!*
- *Dimensionality* can denote either the number of entries along a specific
axis.
    - or the number of axes in a tensor (such as a **5D tensor**)
- Talk about a tensor of *rank 5* (the rank of a tensor being the number of axes).
- A matrix has two axes. An array of 2 Vectors. 
- A *cube* of numbers. 3D Tensors/ higher dimensional tensors
- Deep learning you generally manipulate 0D to 4D tensors; 5D for video processing.

**Key Attributes**

- *Number of axes (rank)*: Tensors ndim in NumPy.
- *Shape*: Tuple of integers that describes how many dimensions the tensor has along each axis.
    - 3D = shape(*3, 3, dimensionality*); 2D = shape(*rows,cols*); 1D = shape(*dimensionality,*)
- *Data type*: (dtype) Type of the data contained in the tensor. (float32, uint8,float64,)

**The notion of batches**
- The first axis (axis 0, indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis. 
- Also called the *samples dimension.*
- Deep learning models do not process an entire dataset at once, rather in **batches**.
- When considering such a batch tensor, the first axis (axis 0) is called the *batch axis* or
*batch dimension*.

#### Real Word Examples of data tensors

*Vector Data*: shape(*samples, features*)
    - Batch of data is encoded as a 2D tensor.
    - An actuarial dataset of people, where we consider each person’s age, ZIP code,and income.
    - Each person can be characterized as a vector of 3 values.
    - Entire dataset of 100,000 people can be stored in a 2D tensor of shape(100000, 3).

**Use Case Two**
    - A dataset of text documents, where we represent each document by the countsof how many times each word appears in it.
    - (out of a dictionary of 20,000 common words).
    - Each document can be encoded as a vector of 20,000 values (one count per word in the dictionary).
    - An entire dataset of 500 documents can be stored in a tensor of shape (500, 20000).
    
*Time series/ Sequence data*: shape(*samples, timesteps, features*)
    - Whenever time matters .in your data; the notion of sequence order.
    - Store it in a 3D tensor with an explicit time axis.
    - Each sample can be encoded as a sequence of vectors.
    - Batch of data will be encoded as a 3D tensor.
    - Time axis is always axis = 1, second axis.
        - A dataset of stock prices.
        - Every minute we store the current price of the stock, the highest price in the previous minute;
        - And the lowest price in the past minute.
        - Every minute is encoded as a 3D vector.
        - An entired day of trading is encoded as a 2D tensor of shape(390,3) i.e 390 minutes in a trading day.
        - 250 days' worth of data can be stored in a 3D tensor of shape(250, 390, 3).
        - Each sample with be one day's worth of data.
        
*Images*: 4D Tensors; shape(*samples, height, width, channels*) or shape(*samples, channels, height, width*)
    - Images typically have 3 dimensions. (eg. height, width, color depth).
    - Gray scale images have have only a single color channel; could be stored in 2D tensors.
    - By convention, image tensors are always 3D, with one dimensional color for grayscale images.
    - Batch of 128 grayscale images of size 256 X 256;
    - Stored in a tensor of shape(128, 256, 256, 1)
    - Batch of 128 color images of size 256 X 256;
    - Stored in a tensor of shape(128, 256, 256, 3).
    - Conventions for shapes of images:
        - *channel-first*: Used by Tensorflow; places the color-depth axis at the end: (samples, height, width, color_depth).
        - *channel-last*: Used by Theanos; places the color depth axis right after the batch axis: (samples, color_depth, height, width).
        - Keras framework provides support for both formats.
        
*Videos*: 5D Tensors; shape(*samples, frames, height, width, channels*);
     shape(*samples, frames, channels, height, width*).
         - Can be understood as a sequence of frames, each frame being a color image.
         - Each frame can be stored in a 3D tensor (height, width, color_depth);
         - A sequence of frames can be stored in a 4D tensor (frames, height, width, color_depth);
         - a batch of different videos can be stored in a 5D tensor;
         - of shape(samples, frames, height, width, color_depth).


