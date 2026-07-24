import numpy as np
import sys

def reshape_single_core(LTC, w, h, z, c, dc, nc, clc):

    if DEBUG==1:
        print(0, LTC.shape)
    temp1 =[]

    for n in range(0, LTC.shape[0]):
        temp2=[]
        for row in range(0, LTC.shape[1], dc):
            temp3=[]
            for c in range(0, LTC.shape[2], 8//dc):
                if dc==2:
                    temp3.extend(np.concatenate((LTC[n, row, c:c+4], LTC[n, row+1, c:c+4]),axis=0).tolist())
                elif dc==1:
                    temp3.extend(LTC[n, row, c:c+8].tolist())
                else:
                    sys.exit("dc_stride Error!")
            temp2.append(np.asarray(temp3))
        temp1.append(np.asarray(temp2))
    LTC= np.asarray(temp1)

    if DEBUG==1:
        print(1, LTC.shape)
                            

    

