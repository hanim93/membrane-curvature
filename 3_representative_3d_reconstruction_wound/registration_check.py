from reconstruct_preview import *
from skimage.registration import phase_cross_correlation
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

def estimate_shift(a,b,fp):
    a=ndi.gaussian_filter(a.astype(float),(.7,1.5,1.5))
    b=ndi.gaussian_filter(b.astype(float),(.7,1.5,1.5))
    # Membrane contrast, with per-stack brightness normalization.
    a=np.log1p(np.maximum(a-85,0));b=np.log1p(np.maximum(b-85,0))
    a-=np.mean(a);b-=np.mean(b)
    p,error,phase=phase_cross_correlation(a,b,upsample_factor=10,normalization=None)
    outside=~morphology.dilation(fp,morphology.disk(8))
    zz,yy,xx=np.indices(a.shape)
    keep=outside[None,:,:] & (zz>=3) & (zz<=min(20,a.shape[0]-5)) & (yy>=5)&(yy<a.shape[1]-5)&(xx>=5)&(xx<a.shape[2]-5)
    coords=np.array(np.where(keep))[:,::3].astype(float);ref=a[keep][::3];ref=ref-ref.mean()
    def loss(shift):
        q=ndi.map_coordinates(b,coords-np.asarray(shift)[:,None],order=1,mode='nearest');q-=q.mean()
        return -np.dot(ref,q)/np.sqrt(np.dot(ref,ref)*np.dot(q,q))
    fits=[minimize(loss,start,method='Powell',bounds=[(-6,6),(-25,25),(-25,25)],options=dict(maxiter=60,xtol=.02)) for start in [np.clip(p,[-6,-25,-25],[6,25,25]),np.array([0,0,0])]]
    best=min(fits,key=lambda r:r.fun)
    return best.x,{'whole_stack_shift_zyx':p.tolist(),'neighbor_shift_zyx':best.x.tolist(),'neighbor_ncc':float(-best.fun),'neighbor_ncc_without_shift':float(-loss([0,0,0]))}

if __name__=='__main__':
 for n in [25]:
    d=np.load(OUT/'endpoint_data'/f'crop{n}.npz');a,b=d['first'],d['last'];cfg=CFG[n];base=cfg['baseline_z0idx'];seed=(cfg['seed_x'],cfg['seed_y'])
    fp=footprint(a,seed,base)
    shift,info=estimate_shift(a,b,fp)
    print(n,info,flush=True)
    (OUT/'endpoint_data'/f'crop{n}_registration.json').write_text(json.dumps(info,indent=2))
    br=ndi.shift(b.astype(float),shift,order=1,mode='constant',cval=85)
    np.savez_compressed(OUT/'endpoint_data'/f'crop{n}_registered.npz',first=a,last=br,shift=shift)
    fps=[footprint(v,seed,base) for v in [a,br]]
    masks=[segment(v,fp,base) for v,fp in zip([a,br],fps)]
    np.savez_compressed(OUT/'endpoint_data'/f'crop{n}_registered_masks.npz',first=masks[0],last=masks[1])
    md=json.loads((OUT/'endpoint_data'/f'crop{n}.json').read_text());meshes=[mesh(mask,md['xy_um'],md['spacing']) for mask in masks]
    av=np.concatenate([v for v,f in meshes]);lims=list(zip(av.min(0)-3,av.max(0)+3))
    fig=plt.figure(figsize=(11,6))
    for t in range(2):
        ax=fig.add_subplot(1,2,t+1,projection='3d');render(ax,*meshes[t],lims,az=-45,el=30);ax.set_title(f'crop{n} {t} rigid registration')
    fig.tight_layout();fig.savefig(OUT/'screening'/f'crop{n}_registered.jpg',dpi=170);plt.close(fig)
