from hypo_surface_build import *
import hashlib

for condition,n in [('hypo',21),('hyper',2)]:
    folder=ROOT/'drift_corrected'/f'{condition}_crop{n}'
    md=json.loads((folder/f'crop{n}_reconstruction.json').read_text())
    for phase in ['t0','tlast']:
        src=folder/f'crop{n}_{phase}_mesh.npz'
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        with np.load(src) as data:
            v,f=data['vertices_um'],data['faces']
        lims=[(-55,55),(-50,50),(-3,40)]
        norm=colors.Normalize(0,40,clip=True);cmap=matplotlib.colormaps['coolwarm']
        tm=trimesh.Trimesh(vertices=v,faces=f,process=False);tm.fix_normals()
        light=np.array([.65,-.55,1.1]);light/=np.linalg.norm(light)
        rgba=cmap(norm(v[:,2]));rgba[:,:3]*=(.78+.22*np.maximum(tm.vertex_normals@light,0))[:,None]
        fig=plt.figure(figsize=(8.2,7.2),facecolor='white')
        ax=fig.add_axes([.005,.15,.79,.70],projection='3d')
        ax.add_collection3d(Poly3DCollection(v[f],facecolors=rgba[f].mean(1),edgecolors=None,linewidths=0,antialiased=False,rasterized=True))
        ax.set_xlim(*lims[0]);ax.set_ylim(*lims[1]);ax.set_zlim(*lims[2])
        ax.set_box_aspect([b-a for a,b in lims],zoom=.99);ax.view_init(32,-48);ax.set_proj_type('ortho')
        ax.set_xlabel('X (µm)',labelpad=7);ax.set_ylabel('Y (µm)',labelpad=9);ax.set_zlabel('Z / Height (µm)',labelpad=7)
        ax.set_xticks([-40,-20,0,20,40]);ax.set_yticks([-40,-20,0,20,40]);ax.set_zticks([0,10,20,30,40]);ax.tick_params(labelsize=9,pad=1)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.set_pane_color((.97,.98,.99,1));axis._axinfo['grid'].update(color=(.78,.81,.84,.65),linewidth=.6);axis.line.set_color('#7d8a94')
        title='t = 0' if phase=='t0' else f"t = last  (~{(md['frames']-1)*md['finterval']/60:.1f} min)"
        fig.text(.405,.868,title,ha='center',fontsize=20,weight='bold',color='#203641')
        fig.text(.04,.952,'Mesophyll cell surface',fontsize=19,weight='bold',color='#203641')
        fig.text(.955,.91,f'{condition}osmo · crop{n}',fontsize=13,ha='right',color='#526876')
        cax=fig.add_axes([.866,.305,.024,.36]);cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[0,10,20,30,40]);cb.set_label('Height above reference plane (µm)',labelpad=10)
        fig.text(.5,.091,'Blue = lower · Red = higher | Height: 0–40 µm',ha='center',fontsize=11,color='#344f5e')
        fig.text(.5,.054,'X, Y, Z in µm · Physical aspect ratio 1:1:1',ha='center',fontsize=10,color='#647580')
        fig.text(.5,.024,'Common reference plane; XYZ translation corrected',ha='center',fontsize=9,color='#647580')
        png=folder/f'crop{n}_{phase}_height_color.png'
        fig.savefig(png,dpi=300);plt.close(fig)
        with Image.open(png) as im:
            im.convert('RGB').save(png.with_suffix('.tif'),compression='tiff_lzw',dpi=(300,300))
        assert hashlib.sha256(src.read_bytes()).hexdigest()==digest
        print(png,flush=True)

