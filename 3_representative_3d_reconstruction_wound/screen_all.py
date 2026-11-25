from explore import *
CONFIG=[(85,84,10),(63,51,10),(75,87,11),(63,87,11),(99,87,17),(75,51,13),(75,75,13),(75,123,13),(63,75,14),(63,99,10),(80,65,15),(75,99,11),(95,90,9),(70,90,12),(130,45,14),(45,50,10),(90,80,9),(71,67,13),(75,75,9),(58,55,10),(63,89,9),(80,50,12),(90,60,15),(90,90,11),(55,85,13),(55,60,12),(45,75,17),(70,75,13),(66,65,14),(64,83,10)]

for base in range(1,31,5):
 fig,axs=plt.subplots(5,4,figsize=(13,14))
 for row,n in enumerate(range(base,base+5)):
  d=np.load(OUT/'endpoint_data'/f'crop{n}.npz');vols=[d['first'],d['last']]
  x,y,z=CONFIG[n-1]
  hi=max(np.percentile(v[z],99.5) for v in vols)
  for t,v in enumerate(vols):
   axs[row,2*t].imshow(v[z],cmap='gray',vmin=85,vmax=hi)
   axs[row,2*t].plot(x,y,'r+',ms=5)
   axs[row,2*t].set_title(f'crop{n} {"first" if t==0 else "last"} z={z}')
   axs[row,2*t+1].imshow(v[:,max(0,y-1):y+2].mean(1),cmap='gray',vmin=85,vmax=hi,aspect=3/.863168)
   axs[row,2*t+1].axhline(z,c='cyan',lw=.5)
   axs[row,2*t+1].set_title(f'XZ at y={y}')
  for ax in axs[row]:ax.axis('off')
 fig.tight_layout();fig.savefig(OUT/'screening'/f'sections_{base:02d}_{base+4:02d}.jpg',dpi=145);plt.close(fig)
