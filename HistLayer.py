
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

        self.D_in = D_in

        self.stride = stride

        self.filt_dim = filt_dim

        self.padding = padding

        #initialize padding value
        self.padvalue = histBins[len(histBins)-1]+1

        #determine output dimensions
        D_out = [0, 0]
        D_out[0] = math.floor((D_in[0]-filt_dim[0] + 2*padding)/stride[0])+1 #output row dim
        D_out[1] = math.floor((D_in[1]-filt_dim[1] + 2*padding)/stride[1])+1 #output column dim

        self.D_out = D_out

        # allocate space for backprop gradient values(?)
        self.gradient = torch.IntTensor(D_out[0], D_out[1], len(histBins), D_in[0], D_in[1]).zero_()

    def bin_img(self, img):
        #input is a window of full image, holds float values
        #output is a the binned window, holds histBin indices/ints
        #modeVect holds histogram frequencies, holds ints
        #mode holds the frequency of the most popular bin, holds int
        #modeBin holds the bin value with the highest frequencies, holds float

        binnedImg = torch.IntTensor(self.filt_dim[0], self.filt_dim[1]).zero_()
        histFreq = torch.FloatTensor(self.numBins).zero_()
        for i in range(0, self.filt_dim[0]):
            for j in range(0, self.filt_dim[1]):
                for k in range(0, self.numBins):
                    if img[i, j] <= self.histBins[k]:
                        binnedImg[i, j] = k
                        break
                histFreq[k] += 1
        return binnedImg, histFreq

    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##output is output of the layer
        ##each element of output is the frequency for the bin for that window

        p = self.padding #layer padding
        output = torch.FloatTensor(torch.zeros(self.D_out[0], self.D_out[1], self.numBins)) ##tensor for output

        ##pad input
        ##inserting input into a larger array filled with pad values
        ##not currently being used
        temp_in = torch.zeros([self.D_in[0]+p, self.D_in[1]+p])
        temp_in[p:self.D_in[0]+1, p:self.D_in[1]+1] = xx

        ##operate on padded input
        for i in range(0, self.D_out[0]): #row/height loop
            i_in = self.stride[0]*i
            for j in range(0, self.D_out[1]): #column/width loop
                j_in = self.stride[1]*j
                for k in range(0, self.numBins):
                    temp_window = xx[i_in:i_in+filt_dim[0], j_in:j_in+filt_dim[1]] #isolate input window
                    temp_binned, histFreqs = self.bin_img(temp_window)
                    output[i, j, k] = histFreqs[k]
                    self.gradient[i, j, k, i_in:i_in+filt_dim[0], j_in:j_in+filt_dim[1]] = temp_binned.eq(int(histFreqs[k]))
        return output


    def backward(self, grad_output):

        #initialize backprop tensor
        backprop = torch.zeros(self.D_in[0], self.D_in[1]) #float tensor

        #for each previsouly outputted element
        # for i in range(0, self.D_out[0]): #row/height loop
        #     i_in = self.stride[0]*i
        #     for j in range(0, self.D_out[1]): #column/width loop
        #         j_in = self.stride[1]*j
        #         # print(grad_output[out_i,out_j], self.gradient[out_i,out_j])
        #         backprop[i_in:i_in+self.stride[0],j_in:j_in+self.stride[1]] = torch.matmul(self.gradient[i,j].float(),
        backprop = self.matmul(self.gradient.float(), grad_output)
        return backprop

    def zero_grad(self):
        ##zero grad output
        self.gradient.zero_()

D_in = [15, 10]
padding = 0
stride = [3, 5]
filt_dim = [3, 5]
histBins = [0.5, 1]

testWin = torch.ones(filt_dim)

input = torch.zeros(D_in)
input[6:9,2:7] = 1.0

net = HistLayer(D_in, padding, stride, filt_dim, histBins)
# print(net.bin_img(testWin))


# print(input)
yy = net.forward(input)
print((net.gradient.size()))
# print(yy)
zz = net.backward(yy)
print(yy)
print(zz[2,0])