import sys, os, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.hypo_dependencies'))
os.environ['MPLCONFIGDIR']=str(ROOT/'hyper_surface_output'/'.mplconfig')
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path(os.environ.get('RAW_DATA_DIR', str(ROOT / f'Crop_hyperosmo')))  # EDIT: point at the folder containing crop*.tif for the hyperosmotic condition, or set the RAW_DATA_DIR environment variable
OUT=ROOT/'hyper_surface_output'
OUT.mkdir(exist_ok=True)
for n in [1,2]:
    with tifffile.TiffFile(SRC/f'crop{n}.tif') as tf:
        md=tf.imagej_metadata
        nz,nt=md['slices'],md['frames']
        res=tf.pages[0].tags['XResolution'].value
        meta=dict(shape=tf.series[0].shape,axes=tf.series[0].axes,xy_um=res[1]/res[0],**{k:v for k,v in md.items() if k in ['spacing','finterval','frames','slices','unit']})
        print(n,meta,flush=True)
        vols=[np.stack([tf.pages[t*nz+z].asarray() for z in range(nz)]) for t in [0,nt-1]]
    np.savez_compressed(OUT/f'crop{n}_endpoints.npz',first=vols[0],last=vols[1])
    (OUT/f'crop{n}_metadata.json').write_text(json.dumps(meta,indent=2))
    fig,axs=plt.subplots(4,7,figsize=(17,10))
    for t,v in enumerate(vols):
        for i,z in enumerate(range(0,min(nz,28),2)):
            ax=axs[t*2+i//7,i%7];ax.imshow(v[z],cmap='gray',vmin=np.percentile(v,5),vmax=np.percentile(v,99.7));ax.set_title(f't={0 if t==0 else nt-1}, z={z}');ax.tick_params(labelsize=6)
    fig.tight_layout();fig.savefig(OUT/f'crop{n}_slices.png',dpi=120);plt.close(fig)

