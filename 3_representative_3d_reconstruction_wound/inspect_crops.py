import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_dependencies'))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'representative_3d' / '.mplconfig')
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, re
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = ROOT / 'representative_3d'
(OUT / 'screening').mkdir(exist_ok=True)
(OUT / 'endpoint_data').mkdir(exist_ok=True)

def inspect(p):
    n = int(re.match(r'crop(\d+)', p.name)[1])
    with tifffile.TiffFile(p) as tf:
        s = tf.series[0]
        md = tf.imagej_metadata or {}
        nz = md.get('slices', 1)
        nt = md.get('frames', 1)
        info = {k:v for k,v in md.items() if k.lower() in ['slices','frames','channels','unit','finterval','spacing','images']}
        info.update(file=p.name, crop=n, axes=s.axes, shape=s.shape, pages=len(tf.pages), dtype=str(s.dtype))
        for a in ['XResolution','YResolution']:
            v = tf.pages[0].tags[a].value
            info[a] = list(v)
        info['xy_um'] = info['XResolution'][1] / info['XResolution'][0]
        a = np.stack([tf.pages[i].asarray() for i in range(nz)])
        b = np.stack([tf.pages[i].asarray() for i in range((nt-1)*nz,nt*nz)])
        np.savez_compressed(OUT/'endpoint_data'/f'crop{n}.npz',first=a,last=b)
        (OUT/'endpoint_data'/f'crop{n}.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(info),flush=True)
        return n, a,b,info

if __name__=='__main__':
    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        fs = {ex.submit(inspect,p):p for p in ROOT.glob('crop*.tif')}
        for f in as_completed(fs):
            try:
                n,a,b,md=f.result();results[n]=(a,b,md)
            except Exception as e: print('ERROR',fs[f],repr(e),flush=True)
    (OUT/'metadata_all.json').write_text(json.dumps([results[k][2] for k in sorted(results)],indent=2))
    for base in range(1,31,5):
        fig,axes=plt.subplots(5,4,figsize=(12,15))
        for row,n in enumerate(range(base,base+5)):
            if n not in results:continue
            a,b,md=results[n]
            hi=max(np.percentile(a,99.7),np.percentile(b,99.7))
            for col,vol in enumerate([a,b]):
                ax=axes[row,col*2]
                ax.imshow(vol.max(0),cmap='gray',vmin=80,vmax=hi)
                ax.set_title(f'crop{n} | '+('t=0' if col==0 else f't={md["frames"]-1}')+' | XY max',fontsize=11)
                ax=axes[row,col*2+1]
                ax.imshow(vol.max(1),cmap='gray',vmin=80,vmax=hi,aspect=md['spacing']/md['xy_um'])
                ax.set_title('XZ max (physical aspect)',fontsize=10)
            for ax in axes[row]: ax.axis('off')
        fig.tight_layout()
        fig.savefig(OUT/'screening'/f'overview_{base:02d}_{base+4:02d}.jpg',dpi=140)
        plt.close(fig)
