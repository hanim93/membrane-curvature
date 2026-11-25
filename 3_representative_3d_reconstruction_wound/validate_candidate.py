from reconstruct_preview import *
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
n=25
d=np.load(OUT/'endpoint_data'/f'crop{n}.npz');md=json.loads((OUT/'endpoint_data'/f'crop{n}.json').read_text())
vols=[d['first'],d['last']];m=np.load(OUT/'endpoint_data'/f'crop{n}_preview_masks.npz');masks=[m['first'],m['last']]
meshes=[mesh(a,md['xy_um'],md['spacing']) for a in masks]
allv=np.concatenate([a for a,b in meshes]);lims=list(zip(allv.min(0)-3,allv.max(0)+3))
fig=plt.figure(figsize=(16,12))
for row,az in enumerate([-135,-45,45,135]):
 for t in range(2):
  ax=fig.add_subplot(4,2,row*2+t+1,projection='3d');render(ax,*meshes[t],lims,az=az,el=32)
  ax.set_title(f'{"first" if t==0 else "last"} az={az}')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=-.4,hspace=-.12)
fig.savefig(OUT/'screening'/'crop25_angles.jpg',dpi=140);plt.close(fig)

fig,axs=plt.subplots(4,4,figsize=(14,13))
for col,t in enumerate([0,1]):
 vol=vols[t];mask=masks[t]; hi=np.percentile(vol,99.9)
 for row,y in enumerate([55,75,95,110]):
  ax=axs[row,col*2];ax.imshow(vol[:,y],cmap='gray',vmin=85,vmax=hi,aspect=3/md['xy_um']);ax.contour(mask[:,y],levels=[.5],colors=['#f27b36'],linewidths=1);ax.set_title(f'{"first" if t==0 else "last"}, XZ y={y}')
 for row,z in enumerate([4,7,10,13]):
  ax=axs[row,col*2+1];ax.imshow(vol[z],cmap='gray',vmin=85,vmax=hi);ax.contour(mask[z],levels=[.5],colors=['#f27b36'],linewidths=1);ax.set_title(f'XY z={z}')
for ax in axs.ravel():ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'screening'/'crop25_validation.jpg',dpi=150);plt.close(fig)
