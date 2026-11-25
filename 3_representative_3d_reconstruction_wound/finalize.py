from reconstruct_preview import *
import warnings, hashlib
warnings.filterwarnings('ignore',category=FutureWarning)
from scipy.spatial import cKDTree
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

FINAL=OUT/'final';FINAL.mkdir(exist_ok=True)
N=25
md=json.loads((OUT/'endpoint_data'/f'crop{N}.json').read_text())
reg=json.loads((OUT/'endpoint_data'/f'crop{N}_registration.json').read_text())
data=np.load(OUT/'endpoint_data'/f'crop{N}.npz')
PAD_Z,PAD_XY=8,26
padding=((PAD_Z,PAD_Z),(PAD_XY,PAD_XY),(PAD_XY,PAD_XY))
vols=[np.pad(data['first'].astype(float),padding,constant_values=85),ndi.shift(np.pad(data['last'].astype(float),padding,constant_values=85),reg['neighbor_shift_zyx'],order=1,mode='constant',cval=85)]
xy=md['xy_um'];zs=md['spacing'];base=13+PAD_Z
fps=[footprint(v,(55+PAD_XY,85+PAD_XY),base) for v in vols]
masks=[segment(v,fp,base) for v,fp in zip(vols,fps)]
assert all(m[:3].sum()==0 for m in masks), 'Top crop boundary reached'
assert all(not (m[:,:,0].any() or m[:,:,-1].any() or m[:,0,:].any() or m[:,-1,:].any()) for m in masks), 'Lateral crop boundary reached'

def finalmesh(mask,sigma_um=2.4):
    pad=((3,3),(8,8),(8,8))
    field=ndi.gaussian_filter(np.pad(mask.astype(float),pad),(sigma_um/zs,sigma_um/xy,sigma_um/xy))
    # Interpolate for display only; this does not improve the acquired Z resolution.
    field=ndi.zoom(field,(3,1,1),order=3)
    dz=zs*(mask.shape[0]+5)/(field.shape[0]-1)
    v,f,_,_=measure.marching_cubes(field,.5,spacing=(dz,xy,xy))
    v=v[:,[2,1,0]]-np.array([8*xy,8*xy,3*zs]);v[:,2]=base*zs-v[:,2]
    return v,f

meshes=[finalmesh(mask) for mask in masks]
# One common translation centers the registered pair; neither mesh is rescaled.
vv=np.concatenate([v for v,f in meshes]);origin=np.array([np.mean([vv[:,0].min(),vv[:,0].max()]),np.mean([vv[:,1].min(),vv[:,1].max()]),0])
meshes=[(v-origin,f) for v,f in meshes]
vv=np.concatenate([v for v,f in meshes]);lims=list(zip(vv.min(0)-3,vv.max(0)+3))

def draw(ax,v,f,az=-45,el=29):
    tm=trimesh.Trimesh(vertices=v,faces=f,process=False)
    # Normals are recalculated after reversing the display Z axis.
    tm.fix_normals()
    vn=tm.vertex_normals
    light=np.array([.6,-.5,1.0]);light/=np.linalg.norm(light)
    lum=np.clip(vn@light,0,1)
    rgb=np.array(to_rgb('#69b8b2'))
    vc=rgb[None,:]*(.53+.47*lum[:,None])
    fc=vc[f].mean(1)
    pc=Poly3DCollection(v[f],facecolors=fc,edgecolors=None,linewidths=0,antialiased=False,rasterized=True)
    ax.add_collection3d(pc)
    for setter,l in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],lims):setter(*l)
    ax.set_box_aspect([l[1]-l[0] for l in lims],zoom=1.13)
    ax.view_init(el,az);ax.set_proj_type('ortho');ax.axis('off')
    return tm

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig=plt.figure(figsize=(12,6.2),facecolor='white')
axes=[]
for t,(v,f) in enumerate(meshes):
    ax=fig.add_axes([.015+t*.5,.10,.485,.77],projection='3d');draw(ax,v,f);axes.append(ax)
    fig.text(.265+t*.5,.875,'t = 0' if t==0 else 't = last  (~30 min)',ha='center',fontsize=19,weight='bold',color='#182d38')
fig.text(.035,.955,'Mesophyll cell surface',fontsize=21,weight='bold',color='#182d38')
fig.text(.966,.955,'crop25',fontsize=16,ha='right',color='#536b75')
fig.text(.5,.077,'Same physical scale and view  |  Membrane-based 3D reconstruction',ha='center',fontsize=11,color='#4a5c65')
fig.text(.5,.04,'Upper cell surface above a common registered cut plane; whole-cell volume is not measured.',ha='center',fontsize=9,color='#69777e')
# A scale bar is placed in the 3D world, so perspective and camera angle are respected.
ax=axes[0];pos=np.array([lims[0][0]+5,lims[1][0]-1,1.0]);end=pos+np.array([10.,0.,0.])
ax.plot([pos[0],end[0]],[pos[1],end[1]],[pos[2],end[2]],c='#233641',lw=3)
fig.canvas.draw()
mid=(pos+end)/2
xp,yp,_=proj3d.proj_transform(*mid,ax.get_proj())
ax.annotate('10 µm',xy=(xp,yp),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=11,color='#233641')
fig.savefig(FINAL/'crop25_t0_vs_tlast_3D.png',dpi=320)
fig.savefig(FINAL/'crop25_t0_vs_tlast_3D.svg',dpi=320)
axes[1].annotate('Local shoulder',xy=(.643,.548),xycoords=fig.transFigure,xytext=(.585,.695),textcoords=fig.transFigure,ha='center',fontsize=12,color='#b35422',arrowprops=dict(arrowstyle='->',color='#b35422',lw=1.6,connectionstyle='arc3,rad=-.12'))
fig.savefig(FINAL/'crop25_t0_vs_tlast_3D_annotated.png',dpi=320)
plt.close(fig)
Image.open(FINAL/'crop25_t0_vs_tlast_3D.png').save(FINAL/'crop25_t0_vs_tlast_3D.tif',compression='tiff_lzw',dpi=(320,320))

for t,(v,f) in enumerate(meshes):
    name='t0' if t==0 else 'tlast'
    fig=plt.figure(figsize=(7,6),facecolor='none');ax=fig.add_axes([0,0,1,1],projection='3d');tm=draw(ax,v,f)
    fig.savefig(FINAL/f'crop25_{name}_transparent.png',dpi=300,transparent=True);plt.close(fig)
    tm.export(FINAL/f'crop25_{name}.ply')
    np.savez_compressed(FINAL/f'crop25_{name}_mesh.npz',vertices_um=v,faces=f)

# A synchronized, offline 3D viewer with physical aspect ratios.
pf=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],subplot_titles=['t = 0','t = last (~30 min)'])
for t,(v,f) in enumerate(meshes):
    pf.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='#69b8b2',flatshading=False,lighting=dict(ambient=.5,diffuse=.75,specular=.17,roughness=.8),lightposition=dict(x=100,y=-100,z=200),name=['t0','tlast'][t],hovertemplate='X %{x:.1f} µm<br>Y %{y:.1f} µm<br>Z %{z:.1f} µm<extra></extra>'),row=1,col=t+1)
scene=dict(aspectmode='data',xaxis=dict(title='X (µm)',range=list(lims[0])),yaxis=dict(title='Y (µm)',range=list(lims[1])),zaxis=dict(title='Height (µm)',range=list(lims[2])),camera=dict(eye=dict(x=1.35,y=-1.35,z=1.0),projection=dict(type='orthographic')))
pf.update_layout(scene=scene,scene2=scene,title=dict(text='crop25 | Mesophyll cell surface — drag to rotate',x=.5),margin=dict(l=5,r=5,b=45,t=75),height=700,paper_bgcolor='white',font=dict(family='Arial'),annotations=[*pf.layout.annotations,dict(text='Same physical scale · estimated translation registration · imaged upper surface only',xref='paper',yref='paper',x=.5,y=-.08,showarrow=False)])
sync="""const p=document.getElementById('{plot_id}');let busy=false;p.on('plotly_relayout',function(e){if(busy)return;let u={};if(e['scene.camera'])u['scene2.camera']=e['scene.camera'];else if(e['scene2.camera'])u['scene.camera']=e['scene2.camera'];else return;busy=true;Plotly.relayout(p,u).then(()=>busy=false);});"""
pf.write_html(FINAL/'crop25_interactive_3D.html',include_plotlyjs=True,full_html=True,post_script=sync)

# Orthogonal raw-data checks for the displayed surfaces.
fig,axs=plt.subplots(3,4,figsize=(13,10))
for t,vol in enumerate(vols):
    mask=masks[t];hi=max(np.percentile(v[PAD_Z:PAD_Z+19,PAD_XY:-PAD_XY,PAD_XY:-PAD_XY],99.6) for v in vols)
    for row,y in enumerate([60,90,120]):
        ax=axs[row,t*2];ax.imshow(vol[PAD_Z-3:PAD_Z+20,y+PAD_XY,PAD_XY:-PAD_XY],cmap='gray',vmin=85,vmax=hi,aspect=zs/xy)
        ax.contour(mask[PAD_Z-3:,y+PAD_XY,PAD_XY:-PAD_XY],levels=[.5],colors=['#f5a34b'],linewidths=.9)
        ax.set_title(f'{"t=0" if t==0 else "t=last, registered"} | XZ, y={y}',fontsize=10)
    for row,z in enumerate([5,9,13]):
        ax=axs[row,t*2+1];ax.imshow(vol[z+PAD_Z,PAD_XY:-PAD_XY,PAD_XY:-PAD_XY],cmap='gray',vmin=85,vmax=hi);ax.contour(mask[z+PAD_Z,PAD_XY:-PAD_XY,PAD_XY:-PAD_XY],levels=[.5],colors=['#f5a34b'],linewidths=.9);ax.set_title(f'XY, registered z index={z}',fontsize=10)
for ax in axs.ravel():ax.axis('off')
fig.suptitle('crop25 | Segmentation over raw membrane signal',fontsize=17)
fig.tight_layout();fig.savefig(FINAL/'crop25_raw_data_QC.png',dpi=230);plt.close(fig)

# Reproducibility data and provenance, without unsupported biological metrics.
tifffile.imwrite(FINAL/'crop25_segmentation_registered.tif',np.asarray(masks,dtype=np.uint8)*255,imagej=True,metadata={'axes':'TZYX','spacing':zs,'unit':'um'},resolution=(1/xy,1/xy))
meta=dict(source=md,registration=reg,display_origin_xyz_um_in_padded_coordinates=origin.tolist(),padding_zyx=[PAD_Z,PAD_XY,PAD_XY],basal_cut_z_index_in_source_reference=13,basal_cut_um_in_source_reference=13*zs,smoothing_sigma_um=2.4,rendered_frame_indices=[0,120],duration_minutes=(md['frames']-1)*md['finterval']/60,interpretation='Exploratory, membrane-derived upper-surface shape comparison. No conservation of whole-cell volume is imposed or inferred.',mesh_units='micrometers',analysis_parameters=dict(footprint_projection_z_indices=[11,12,13],footprint_gaussian_sigma_px=1.3,footprint_threshold='0.65 x local Otsu threshold',watershed_intensity_gaussian_sigma_zyx=[.6,1.2,1.2],display_interpolation_factor_z=3),original_file_modified=False)
(FINAL/'reconstruction_metadata.json').write_text(json.dumps(meta,indent=2))
print('DONE',FINAL,flush=True)
