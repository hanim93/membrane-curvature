import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.hypo_dependencies'))
OUT=ROOT/'hypo_surface_output'
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import numpy as np
from scipy import ndimage as ndi
from skimage import filters,morphology,measure
from skimage.segmentation import watershed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors,cm
import trimesh
from PIL import Image

def build(n):
    data=np.load(OUT/f'crop{n}_endpoints.npz');md=json.loads((OUT/f'crop{n}_metadata.json').read_text())
    xy,zs=md['xy_um'],md['spacing'];base=13
    seed={16:(70,45),17:(80,55),23:(60,60),24:(65,70),21:(65,65),22:(60,60)}[n]
    vols=[data['first'],data['last']];masks=[];meshes=[];fps=[]
    fp_zlos=[16,11] if n==17 else [11,11]
    for phase,vol in enumerate(vols):
        fp_zlo=fp_zlos[phase]
        img=ndi.gaussian_filter(vol[fp_zlo:fp_zlo+3].max(0).astype(float),1.3)
        thr_factor=.55 if n==23 else .65
        labels=measure.label(img<filters.threshold_otsu(img)*thr_factor)
        lab=labels[seed[1],seed[0]];assert lab>0
        fp=ndi.binary_fill_holes(labels==lab)
        fp=ndi.binary_fill_holes(morphology.binary_closing(fp,morphology.disk(2)))
        if n==17:
            # Break the narrow threshold leak into the adjacent cell, then
            # retain only the component containing the target interior seed.
            opened=morphology.opening(fp,morphology.disk(4))
            labs=measure.label(opened);lab=labs[seed[1],seed[0]]
            assert lab>0
            fp=ndi.binary_fill_holes(labs==lab)
        smooth=ndi.gaussian_filter(vol[:base+1].astype(float),(.6,1.2,1.2))
        markers=np.zeros(smooth.shape,np.int32)
        outer=~morphology.binary_dilation(fp,morphology.disk(5))
        markers[:,outer]=1;markers[0]=1;markers[-1,~fp]=1
        markers[-1,morphology.binary_erosion(fp,morphology.disk(2))]=2
        mask=watershed(smooth,markers)==2
        masks.append(mask);fps.append(fp)
        pad=((3,3),(8,8),(8,8));sigma=2.4
        field=ndi.gaussian_filter(np.pad(mask.astype(float),pad),(sigma/zs,sigma/xy,sigma/xy))
        field=ndi.zoom(field,(3,1,1),order=3)
        dz=zs*(mask.shape[0]+5)/(field.shape[0]-1)
        v,f,_,_=measure.marching_cubes(field,.5,spacing=(dz,xy,xy))
        v=v[:,[2,1,0]]-np.array([8*xy,8*xy,3*zs]);v[:,2]=base*zs-v[:,2]
        meshes.append((v,f))
    fig,axs=plt.subplots(3,4,figsize=(14,10))
    for t,(vol,mask) in enumerate(zip(vols,masks)):
        for j,z in enumerate([5,9,13]):
            ax=axs[j,t*2];ax.imshow(vol[z],cmap='gray',vmin=np.percentile(vol,5),vmax=np.percentile(vol,99.5));ax.contour(mask[z],levels=[.5],colors=['orange']);ax.set_title(f't={t*(md["frames"]-1)} XY z={z}')
        for j,y in enumerate([30,50,70]):
            ax=axs[j,t*2+1];ax.imshow(vol[:17,y,:],cmap='gray',vmin=np.percentile(vol,5),vmax=np.percentile(vol,99.5),aspect=zs/xy);ax.contour(mask[:,y,:],levels=[.5],colors=['orange']);ax.set_title(f't={t*(md["frames"]-1)} XZ y={y}')
    fig.tight_layout();fig.savefig(OUT/f'crop{n}_raw_data_QC.png',dpi=150);plt.close(fig)
    vv=np.concatenate([v for v,f in meshes]);origin=(vv.min(0)+vv.max(0))/2;origin[2]=0
    meshes=[(v-origin,f) for v,f in meshes]
    clipped=[bool(m[:,:,0].any() or m[:,:,-1].any() or m[:,0,:].any() or m[:,-1,:].any()) for m in masks]
    print(n,'fp areas',[int(f.sum()) for f in fps],'clipped',clipped,flush=True)
    lims=[(-55,55),(-50,50),(-3,40)];norm=colors.Normalize(0,40,clip=True);cmap=matplotlib.colormaps['coolwarm']
    fig=plt.figure(figsize=(14,7.2),facecolor='white')
    for t,(v,f) in enumerate(meshes):
        tm=trimesh.Trimesh(vertices=v,faces=f,process=False);tm.fix_normals()
        light=np.array([.65,-.55,1.1]);light/=np.linalg.norm(light)
        rgba=cmap(norm(v[:,2]));rgba[:,:3]*=(.78+.22*np.maximum(tm.vertex_normals@light,0))[:,None]
        ax=fig.add_axes([.015+t*.447,.13,.423,.69],projection='3d')
        ax.add_collection3d(Poly3DCollection(v[f],facecolors=rgba[f].mean(1),edgecolors=None,linewidths=0,antialiased=False,rasterized=True))
        ax.set_xlim(*lims[0]);ax.set_ylim(*lims[1]);ax.set_zlim(*lims[2]);ax.set_box_aspect([b-a for a,b in lims],zoom=.99);ax.view_init(32,-48);ax.set_proj_type('ortho')
        ax.set_xlabel('X (µm)',labelpad=7);ax.set_ylabel('Y (µm)',labelpad=9);ax.set_zlabel('Z / Height (µm)',labelpad=7)
        ax.set_xticks([-40,-20,0,20,40]);ax.set_yticks([-40,-20,0,20,40]);ax.set_zticks([0,10,20,30,40]);ax.tick_params(labelsize=9,pad=1)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.set_pane_color((.97,.98,.99,1));axis._axinfo['grid'].update(color=(.78,.81,.84,.65),linewidth=.6);axis.line.set_color('#7d8a94')
        title='t = 0' if t==0 else f"t = last  (~{(md['frames']-1)*md['finterval']/60:.1f} min)"
        ax.set_title(title,fontsize=18,weight='bold',pad=17,color='#203641')
        np.savez_compressed(OUT/f'crop{n}_{["t0","tlast"][t]}_mesh.npz',vertices_um=v,faces=f)
    fig.text(.035,.944,'Mesophyll cell surface',fontsize=23,weight='bold',color='#203641')
    fig.text(.965,.946,f'crop{n} · hypoosmo · height coloring',fontsize=14,ha='right',color='#526876')
    cax=fig.add_axes([.924,.285,.016,.375]);cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[0,10,20,30,40]);cb.set_label('Height above reference plane (µm)',labelpad=12)
    fig.text(.5,.085,'Blue = lower  ·  Red = higher  |  Shared color scale: 0–40 µm',ha='center',fontsize=12,color='#344f5e')
    note='X, Y, Z in µm · Physical aspect ratio 1:1:1 · Common acquisition Z plane; no drift correction'
    fig.text(.5,.048,note,ha='center',fontsize=10,color='#647580')
    if any(clipped): fig.text(.5,.019,'Partial cell surface: the source crop cuts the cell boundary; cut faces are display closures.',ha='center',fontsize=9,color='#9a582b')
    name=f'crop{n}_t0_vs_tlast_height_color'
    fig.savefig(OUT/f'{name}.png',dpi=300);plt.close(fig)
    with Image.open(OUT/f'{name}.png') as im: im.convert('RGB').save(OUT/f'{name}.tif',compression='tiff_lzw',dpi=(300,300))
    np.savez_compressed(OUT/f'crop{n}_masks.npz',first=masks[0],last=masks[1])
    md.update(baseline_z_index=base,footprint_threshold_factor=thr_factor,footprint_projection_z_indices_by_phase=[list(range(z,z+3)) for z in fp_zlos],footprint_opening_radius_px=4 if n==17 else 0,smoothing_sigma_um=2.4,registration='None; acquisition coordinates retained',source_boundary_clipped=clipped,origin_xyz_um=origin.tolist(),method='Membrane intensity watershed; smoothed marching cubes; common basal display closure',color_scale_um=[0,40])
    (OUT/f'crop{n}_reconstruction.json').write_text(json.dumps(md,indent=2))

if __name__=='__main__':
    for n in (list(map(int,sys.argv[1:])) or [16,17]):build(n)

