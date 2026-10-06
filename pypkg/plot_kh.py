import numpy as np 
import matplotlib.pyplot as plt
from read_gaukuk import ReadGaukuk
import os 

DEN = 0 
MTX = 1 
MTY = 2 
MTZ = 3 
ENG = 4 
gamma = 1.4 
gm1Rec = 1.0 / (gamma - 1)
totalFrames = 51

tlist = np.zeros(totalFrames) 
englist = np.zeros(totalFrames)

ROOT = "/mnt/d/Linux/Document/cal_gaukuk/KH/double/"
path = ROOT + "r2048/"
# path = "../bin/"
sPath = path
tag = "kh"

for frameID in range(0, totalFrames):
    filename = path + "cons_" + str(frameID).zfill(5)
    savename = sPath + tag + "_" + str(frameID).zfill(5) + ".png"
    frame = ReadGaukuk(filename)

    k = int((frame.lenz)/2)
    kl = frame.nGhost
    kr = frame.nGhost + frame.nz 
    jl = frame.nGhost
    jr = frame.nGhost + frame.ny 
    il = frame.nGhost
    ir = frame.nGhost + frame.nx
    rho = frame.data[DEN,k,jl:jr,il:ir]
    mtx = frame.data[MTX,k,jl:jr,il:ir]
    mty = frame.data[MTY,k,jl:jr,il:ir]
    mtz = frame.data[MTZ,k,jl:jr,il:ir]
    eng = frame.data[ENG,k,jl:jr,il:ir]
    eng2 = frame.data[ENG,kl:kr,jl:jr,il:ir]
    tlist[frameID] = frame.t 
    englist[frameID] = np.sum(eng2)

    x = frame.xc
    y = frame.yc
    rhoMin = 1
    rhoMax = 2

    # print(il,ir,jl,jr)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, rho, cmap='viridis', shading='gouraud',
                   vmin=rhoMin, vmax=rhoMax)
    plt.colorbar(label='Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Density, t = {(frame.t):.2f}')
    plt.axis('equal')   
    plt.savefig(savename, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()
    print("frame = ", frameID, end="\r")
    del frame

np.save(sPath + "t_" + tag + ".npy", tlist)
np.save(sPath + "eng_" + tag + ".npy", englist)

# tlist = np.load(sPath + "t_" + tag + ".npy")
# englist = np.load(sPath + "eng_" + tag + ".npy")

plt.figure(figsize=(8, 6))
plt.plot(tlist, englist/englist[0] - 1)

emin = np.min(englist) / englist[0]
emax = np.max(englist) / englist[0]
plt.ylim(-1e-15, 1e-15)
plt.ylim(-1e-14, 1e-14)

plt.xlim(tlist[0], tlist[-1])
plt.xlabel('t', fontsize = 15)
plt.ylabel(r'$E/E0 - 1$', fontsize = 15)
plt.grid()
plt.title('Energy Conservation', fontsize = 15) 
plt.savefig(sPath + tag + "_energy_conservation.png", bbox_inches='tight', dpi=300)
# plt.show()
plt.close()

os.system("ffmpeg -y -framerate 10 -i " + sPath + tag + "_%05d.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p -crf 23 " + sPath + tag + ".mp4")
