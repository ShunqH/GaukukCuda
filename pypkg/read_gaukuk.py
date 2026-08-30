import numpy as np

class ReadGaukuk:
    def __init__(self, filename, isCons = True, dtype=np.float64):
        self.filename = filename
        self.dtype = dtype

        with open(filename, "rb") as f:
            RealType = np.fromfile(f, dtype=np.int32, count=1)[0]
            if RealType == 8:
                dtype = np.float64
            elif RealType == 4:
                dtype = np.float32

            # frame information
            self.t = np.fromfile(f, dtype=dtype, count=1)[0]

            # header
            self.nvar = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.nx = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.ny = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.nz = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.nGhost = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.lenx = self.nx + 2 * self.nGhost 
            self.leny = self.ny + 2 * self.nGhost
            self.lenz = self.nz + 2 * self.nGhost
            self.lenArr = self.lenz * self.leny * self.lenx 

            # mesh
            self.xcg = np.fromfile(f, dtype=dtype, count=self.lenx)
            self.ycg = np.fromfile(f, dtype=dtype, count=self.leny)
            self.zcg = np.fromfile(f, dtype=dtype, count=self.lenz)

            self.xc = self.xcg[self.nGhost: self.nGhost+self.nx]
            self.yc = self.ycg[self.nGhost: self.nGhost+self.ny]
            self.zc = self.zcg[self.nGhost: self.nGhost+self.nz]

            # data
            self.dataLen = np.fromfile(f, dtype=np.int32, count=1)[0] 
            if (self.dataLen != self.nvar * self.lenArr ):
                print("data size error, reshape might fail")
            self.data = np.fromfile(f, dtype=dtype, count=self.dataLen)
            self.data = self.data.reshape(
                    (self.nvar, self.lenz, self.leny, self.lenx)
                )