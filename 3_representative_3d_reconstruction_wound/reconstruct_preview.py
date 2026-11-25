from explore import *
from skimage.segmentation import watershed
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

CFG={n:dict(seed_x=x,seed_y=y,baseline_z0idx=z) for n,x,y,z in [(4,70,90,13),(5,95,100,17),(6,65,65,13),(10,63,99,10),(12,75,99,11),(13,85,70,12),(17,90,80,9),(23,85,105,15),(24,85,90,13),(25,55,85,13),(26,55,70,12),(30,64,83,10)]}

def footprint(vol,seed,zbase):
    # Isolate the dark lumen inside the membrane ring; use a global crop window.
    img=ndi.gaussian_filter(vol[max(0,zbase-2):zbase+1].max(0).astype(float),1.3)
    hi=filters.threshold_otsu(img)
    labels=measure.label(img<hi*.65)
    lab=labels[int(seed[1]),int(seed[0])]
    mask=ndi.binary_fill_holes(labels==lab) if lab else np.zeros_like(img,dtype=bool)
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        return None
    mask=morphology.binary_closing(mask,morphology.disk(2))
    mask=ndi.binary_fill_holes(mask)
    return mask

def segment(vol,fp,base):
    smooth=ndi.gaussian_filter(vol[:base+1].astype(float),(.6,1.2,1.2))
    markers=np.zeros(smooth.shape,dtype=np.int32)
    outer=~morphology.binary_dilation(fp,morphology.disk(5))
    markers[:,outer]=1
    markers[0]=1
    markers[-1,~fp]=1
    markers[-1,morphology.binary_erosion(fp,morphology.disk(2))]=2
    labels=watershed(smooth,markers)
    mask=labels==2
    return mask

def mesh(mask,xy,zs):
    # A half-voxel buffer keeps the top and lateral surface closed; base is flat.
    field=ndi.gaussian_filter(np.pad(mask.astype(float),((2,2),(6,6),(6,6))),(.8,2.5,2.5))
    verts,faces,_,_=measure.marching_cubes(field,.5,spacing=(zs,xy,xy))
    verts=verts[:,[2,1,0]]-np.array([6*xy,6*xy,2*zs])
    verts[:,2]=mask.shape[0]*zs-zs-verts[:,2]
    return verts,faces

def render(ax,verts,faces,lims,az=-65,el=25):
    pc=Poly3DCollection(verts[faces],facecolors='#67beba',edgecolors=None,shade=True,lightsource=LightSource(azdeg=315,altdeg=55),rasterized=True)
    ax.add_collection3d(pc)
    for setter,v in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],lims):setter(*v)
    ax.set_box_aspect([v[1]-v[0] for v in lims]);ax.view_init(el,az);ax.set_proj_type('ortho');ax.axis('off')

def go(n):
    print('START',n,flush=True)
    r=CFG[n];seed=(int(r['seed_x']),int(r['seed_y']))
    data=np.load(OUT/'endpoint_data'/f'crop{n}.npz');md=json.loads((OUT/'endpoint_data'/f'crop{n}.json').read_text())
    vols=[data['first'],data['last']]
    zbase=int(r['baseline_z0idx'])
    masks=[];fps=[];vs=[]
    for vol in vols:
        fp=footprint(vol,seed,zbase)
        if fp is None: print('failed',n,flush=True);return
        mask=segment(vol,fp,zbase);masks.append(mask);fps.append(fp)
        vs.append(mesh(mask,md['xy_um'],md['spacing']))
    allv=np.concatenate([v for v,f in vs]);low=allv.min(0)-4;high=allv.max(0)+4;lims=list(zip(low,high))
    fig=plt.figure(figsize=(12,9))
    for t in range(2):
        ax=fig.add_subplot(2,2,t+1,projection='3d');render(ax,*vs[t],lims)
        ax.set_title(f'crop{n}, '+('first' if t==0 else 'last'))
        ax=fig.add_subplot(2,2,t+3)
        ax.imshow(vols[t][zbase],cmap='gray',vmin=90,vmax=np.percentile(vols[t],99.7));ax.contour(fps[t],levels=[.5],colors=['cyan'])
        ax.contour(masks[t][-1],levels=[.5],colors=['orange']);ax.set_title(f'base slice z={zbase}');ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'screening'/f'crop{n}_watershed.jpg',dpi=150);plt.close(fig)
    np.savez_compressed(OUT/'endpoint_data'/f'crop{n}_preview_masks.npz',first=masks[0],last=masks[1])
    print('DONE',n,'vol_ratio',masks[1].sum()/masks[0].sum(),flush=True)

if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore',category=FutureWarning)
    for n in map(int,sys.argv[1:]):
        try:go(n)
        except Exception as e:print('FAIL',n,str(e),flush=True)
