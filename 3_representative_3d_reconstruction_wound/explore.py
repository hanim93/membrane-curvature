from inspect_crops import *
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
import csv

def zsheet(n):
    d=np.load(OUT/'endpoint_data'/f'crop{n}.npz')
    a,b=d['first'],d['last']
    fig,axs=plt.subplots(6,12,figsize=(20,10))
    for t,vol in enumerate([a,b]):
        hi=np.percentile(vol,99.7)
        for z in range(min(len(vol),36)):
            ax=axs[t*3+z//12,z%12]
            ax.imshow(vol[z],cmap='gray',vmin=90,vmax=hi)
            ax.set_title(f'{"first" if t==0 else "last"} z={z}',fontsize=9)
            ax.axis('off')
    fig.tight_layout(pad=.4)
    fig.savefig(OUT/'screening'/f'crop{n}_zslices.jpg',dpi=145)
    plt.close(fig)

if __name__=='__main__':
    for n in map(int,sys.argv[1:]): zsheet(n)
