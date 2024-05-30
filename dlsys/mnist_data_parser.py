import numpy as np
from matplotlib import pyplot as plt


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

     TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255.
    0 means background (white), 255 means foreground (black).

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with open(image_filename, "rb") as file:
        magic_num = int.from_bytes(file.read(4), byteorder="big")
        num_images = int.from_bytes(file.read(4), byteorder="big")
        rows = int.from_bytes(file.read(4), byteorder="big")
        cols = int.from_bytes(file.read(4), byteorder="big")
        assert rows == 28
        assert cols == 28
        X = np.ndarray((num_images, rows * cols), dtype=np.float32)
        for i in range(num_images):
            img_data = []
            for _ in range(rows * cols):
                val = float(ord(file.read(1)))
                img_data.append(val / 255)
            X[i] = img_data
    with open(label_filename, "rb") as file:
        magic_num = int.from_bytes(file.read(4), byteorder="big")
        num_labels = int.from_bytes(file.read(4), byteorder="big")
        Y = np.ndarray(num_images, dtype=np.int)
        for i in range(num_labels):
            Y[i] = int(ord(file.read(1)))

    return X, Y


if __name__ == "__main__":
    
    X, Y = parse_mnist(
        "data_mnist/train-images-idx3-ubyte/train-images.idx3-ubyte",
        "data_mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte",
    )
