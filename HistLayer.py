
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
        self.gradient = torch.IntTensor(D_out[0], D_out[1], D_in[0], D_in[1]).zero_()

    def bin_img(self, img):
        #input is a window of full image, holds float values
        #output is a the binned window, holds histBin indices/ints
        #modeVect holds histogram frequencies, holds ints
        #mode holds the frequency of the most popular bin, holds int
        #modeBin holds the bin value with the highest frequencies, holds float

        output = torch.IntTensor(self.filt_dim[0], self.filt_dim[1]).zero_()
        histFreq = np.zeros((self.numBins), dtype=np.intp)
        for i in range(0, self.filt_dim[0]):
            for j in range(0, self.filt_dim[1]):
                for k in range(0, self.numBins):
                    # print("img["+str(i)+", "+ str(j) +"] = ",img[i,j])
                    if img[i, j] <= self.histBins[k]:
                        # print("histBin["+str(k)+"] = ", histBins[k])
                        output[i, j] = k
                        break

                histFreq[k] += 1
        modeFreq = np.max(histFreq)
        modeBin = np.argmax(histFreq)
        return output, modeFreq, modeBin


    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        p = self.padding #layer padding
        output = torch.FloatTensor(torch.zeros(self.D_out)) ##tensor for output

        ##pad input
        ##inserting input into a larger array filled with pad values
        ##not currently being used
        temp_in = torch.zeros([self.D_in[0]+p, self.D_in[1]+p])
        temp_in[p:self.D_in[0]+1, p:self.D_in[1]+1] = xx

        ##operate on padded input
        for i in range(0, self.D_in[0], self.stride[0]): #row/height loop
            out_i = int(i / self.stride[0])
            for j in range(0, self.D_in[1], self.stride[1]): #column/width loop
                out_j = int(j/self.stride[1])
                temp_window = xx[i:i+filt_dim[0], j:j+filt_dim[1]] #isolate window
                temp_binned, mode_freq, modeBin = self.bin_img(temp_window)
                # print(temp_binned, mode_freq, modeBin)
                output[out_i, out_j] = float(modeBin)
                self.gradient[out_i, out_j, i:i+filt_dim[0], j:j+filt_dim[1]] = temp_binned.eq(int(modeBin))
        return output


    def backward(self, grad_output):

        #initialize backprop tensor
        backprop = torch.zeros(self.D_out[0], self.D_out[1], self.D_in[0], self.D_in[1]) #float tensor
        # print(self.gradient)
        #for each previsouly outputted element
        for i in range(0, self.D_in[0], self.stride[0]): #row/height loop
            out_i = int(i / self.stride[0])
            for j in range(0, self.D_in[1], self.stride[1]): #column/width loop
                out_j = int(j / self.stride[1])
                # print(grad_output[out_i,out_j], self.gradient[out_i,out_j])
                backprop[out_i,out_j] = self.gradient[out_i,out_j].float()*grad_output[out_i,out_j]
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