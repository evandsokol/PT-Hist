
# %windir%\system32\cmd.exe "/K" C:\Users\evands\AppData\Local\Anaconda3.6\Scripts\activate.bat C:\Users\evands\AppData\Local\Anaconda3.6
import torch.nn as nn
import torch
import math
import numpy as np


class HistLayer(nn.Module):
    def __init__(self, D_in, padding, stride, filt_dim, histBins):

        #inheirit nn.module
        super(nn.Module, self).__init__()


        #define layer properties

        #histogram bin data
        self.histBins = histBins
        self.histBins.sort()
        self.numBins = len(histBins)

        #initialize input dimensions
        # if type(D_in).__name__ == 'int':
        #     self.D_in = [D_in, D_in]
        # elif type(D_in).__name__ == 'list':
        #     if len(D_in) == 2:
        #         self.D_in = D_in
        # else:
        #     print("ERROR, UNACCEPTABLE INPUT DIMENSION")
        self.D_in = D_in

        #initialize stride
        # if type(stride).__name__ == 'int':
        #     self.stride = [stride, stride]
        # elif type(stride).__name__ == 'list':
        #     if len(stride) == 2:
        #         self.stride = stride
        # else:
        #     print("ERROR, UNACCEPTABLE STRIDE LENGTH")
        self.stride = stride

        #initialize filter dimensions
        # if type(filt_dim).__name__ == 'int':
        #     self.filt_dim = tuple([filt_dim, filt_dim])
        # elif type(filt_dim).__name__ == 'list' or 'tuple':
        #     if len(filt_dim) == 2:
        #         self.filt_dim = tuple(filt_dim)
        # else:
        #     print("ERROR, UNACCEPTABLE FILTER SIZE")
        self.filt_dim = filt_dim

        #initialize padding size
        # if type(padding).__name__ == 'int':
        #     self.padding = padding
        # else :
        #     print("ERROR, UNACCEPTABLE PADDING")
        self.padding = padding

        #initialize padding value
        self.padvalue = histBins[len(histBins)-1]+1

        #determine output dimensions
        D_out = [0, 0]
        D_out[0] = math.floor((D_in[0]-filt_dim[0] + 2*padding)/stride[0])+1
        D_out[1] = math.floor((D_in[1]-filt_dim[1] + 2*padding)/stride[1])+1

        self.D_out = D_out

        # allocate space for backprop gradient values(?)
        self.gradient = torch.IntTensor(D_out[0], D_out[1], D_in[0], D_in[1]).zero_()

    def bin_img(self, img):
        #input is a window of full image, holds float values
        #output is a the binned window, holds histBin indices/ints
        #modeVect holds histogram frequencies, holds ints
        #mode holds the frequency of the most popular bin, holds int
        #modeBin holds the bin value with the highest frequencies, holds float

        output = torch.IntTensor(self.filt_dim[0], self.filt_dim[1]).zero_()
        modeVect = np.zeros((1, self.numBins), dtype=np.intp)
        for i in range(0, self.filt_dim[0]):
            for j in range(0, self.filt_dim[1]):
                for k in range(0, self.numBins):
                    if img[i, j] < self.histBins[k]:
                        output[i, j] = k
                        break
                    modeVect[k] += 1
        mode = np.max(modeVect)
        modeBin = np.argmax(modeVect)
        return output, mode, modeBin


    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        p = self.padding #layer padding
        output = torch.FloatTensor(torch.zeros(self.D_out)) ##tensor for output



        ##pad input
        temp_in = torch.zeros([self.D_in[0]+p, self.D_in[1]+p])
        temp_in[p:self.D_in[0]+1, p:self.D_in[1]+1] = xx

        ##operate on padded input
        for i in range(0, self.D_in[0], self.stride[0]): #row/height loop
            for j in range(0, self.D_in[1], self.stride[1]): #column/width loop
                temp_window = xx[i:i+filt_dim[0], j:j+filt_dim[1]] #isolate window
                temp_binned, mode, modeBin = self.bin_img(temp_window)
                print(output.size())
                print(self.stride)
                print(i), print(j)
                output[i, j] = float(mode)
                self.gradient[i, j, i:i+filt_dim[0], j:j+filt_dim[1]] = temp_binned.eq(int(mode))
        return output


    def backward(self, grad_output):
        ##need to multiply grad value by grad output.. poorly phrasedbut need to actually backprop.
        output = torch.zeros(self.D_out[0], self.D_out[1], self.D_in[0], self.D_in[1]).register_buffer()
        for i in range(0, self.D_in[1], self.stride[1]): #row/height loop
            for j in range(0, self.D_in[0], self.stride[0]): #column/width loop
               output[i,j] = self.gradient[i,j]*grad_output[i,j]
        return output

    def zero_grad(self):
        ##zero grad output
        self.gradient.zero_()

D_in = [15, 10]
padding = 0
stride = [3, 5]
filt_dim = [3, 5]
histBins = [0.5, 1]

input = torch.zeros(D_in)
input[6:9,2:7] = 1.0
# print(input)


net = HistLayer(D_in, padding, stride, filt_dim, histBins)
net.forward(input)