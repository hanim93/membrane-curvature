"""Re-render the saved crop25 meshes with shared height colors and XYZ axes.

The source meshes are read only. No segmentation, registration, or geometry
changes are made by this script.
"""
import sys, os, json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_dependencies'))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'representative_3d' / '.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

OUT = ROOT / 'representative_3d' / 'final'
meshes = []
hashes = {}
for phase in ['t0', 'tlast']:
    p = OUT / f'crop25_{phase}_mesh.npz'
    hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    d = np.load(p)
    meshes.append((d['vertices_um'], d['faces']))

# Shared physical coordinates, limits, and color normalization for both times.
LIMITS = [(-40, 40), (-55, 55), (-3, 40)]
COLOR_LIMITS = [0, 40]
NORM = colors.Normalize(*COLOR_LIMITS, clip=True)
CMAP = matplotlib.colormaps['coolwarm']
COLOR_SCALE = [[i / 255, colors.to_hex(CMAP(i / 255))] for i in range(256)]
TITLES = ['t = 0', 't = last  (~30 min)']

def draw(ax, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    light = np.array([.65, -.55, 1.1])
    light /= np.linalg.norm(light)
    brightness = .78 + .22 * np.maximum(mesh.vertex_normals @ light, 0)
    rgba = CMAP(NORM(vertices[:, 2]))
    rgba[:, :3] *= brightness[:, None]
    collection = Poly3DCollection(
        vertices[faces], facecolors=rgba[faces].mean(axis=1),
        edgecolors=None, linewidths=0, antialiased=False, rasterized=True)
    ax.add_collection3d(collection)
    ax.set_xlim(*LIMITS[0]); ax.set_ylim(*LIMITS[1]); ax.set_zlim(*LIMITS[2])
    ax.set_box_aspect([b-a for a,b in LIMITS], zoom=.99)
    ax.view_init(elev=32, azim=-48)
    ax.set_proj_type('ortho')
    ax.set_xlabel('X (µm)', labelpad=7)
    ax.set_ylabel('Y (µm)', labelpad=9)
    ax.set_zlabel('Z / Height (µm)', labelpad=7)
    ax.set_xticks([-40,-20,0,20,40]); ax.set_yticks([-40,-20,0,20,40])
    ax.set_zticks([0,10,20,30,40])
    ax.tick_params(labelsize=9, pad=1)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((.97,.98,.99,1))
        axis._axinfo['grid'].update(color=(.78,.81,.84,.65), linewidth=.6)
        axis.line.set_color('#7d8a94')

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})

# Separate panels retain identical camera, dimensions, axes, and color limits.
for t, (phase, (v, f)) in enumerate(zip(['t0', 'tlast'], meshes)):
    fig = plt.figure(figsize=(8.2, 7.2), facecolor='white')
    ax = fig.add_axes([.005, .15, .79, .70], projection='3d')
    draw(ax, v, f)
    fig.text(.405, .878, TITLES[t], ha='center', fontsize=20,
             weight='bold', color='#203641')
    fig.text(.04, .954, 'Mesophyll cell surface', fontsize=19,
             weight='bold', color='#203641')
    fig.text(.955, .954, 'crop25', fontsize=14, ha='right', color='#526876')
    cax = fig.add_axes([.866, .305, .024, .36])
    cbar = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=CMAP),
                       cax=cax, ticks=[0, 10, 20, 30, 40])
    cbar.set_label('Height above reference plane (µm)', labelpad=10)
    cbar.outline.set_edgecolor('#89959d')
    fig.text(.5, .091, 'Blue = lower  ·  Red = higher  |  Height: 0–40 µm',
             ha='center', fontsize=11, color='#344f5e')
    fig.text(.5, .054, 'X, Y, Z in µm  ·  Physical aspect ratio 1:1:1',
             ha='center', fontsize=10, color='#647580')
    png = OUT / f'crop25_{phase}_height_color.png'
    fig.savefig(png, dpi=300)
    fig.savefig(OUT / f'crop25_{phase}_height_color.svg', dpi=300)
    plt.close(fig)
    with Image.open(png) as im:
        im.save(OUT / f'crop25_{phase}_height_color.tif',
                compression='tiff_lzw', dpi=(300, 300))
        print(json.dumps(dict(file=str(png),size=im.size)))

if '--individual-only' in sys.argv:
    assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h
               for name,h in hashes.items())
    raise SystemExit(0)

fig = plt.figure(figsize=(14,7.2), facecolor='white')
for t,(v,f) in enumerate(meshes):
    ax = fig.add_axes([.015 + t*.447,.13,.423,.69], projection='3d')
    draw(ax,v,f)
    ax.set_title(TITLES[t], fontsize=18, weight='bold', pad=17, color='#203641')
fig.text(.035,.944,'Mesophyll cell surface', fontsize=23, weight='bold', color='#203641')
fig.text(.965,.946,'crop25 · height coloring', fontsize=15, ha='right', color='#526876')
cax = fig.add_axes([.924,.285,.016,.375])
cbar = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=[0,10,20,30,40])
cbar.set_label('Height above reference plane (µm)',labelpad=12)
cbar.outline.set_edgecolor('#89959d')
fig.text(.5,.085,'Blue = lower  ·  Red = higher  |  Shared color scale: 0–40 µm',ha='center',fontsize=12,color='#344f5e')
fig.text(.5,.048,'X, Y, Z in µm  ·  Physical aspect ratio 1:1:1  ·  Z = 0: common registered reference plane',ha='center',fontsize=10,color='#647580')
png = OUT / 'crop25_t0_vs_tlast_height_color.png'
fig.savefig(png, dpi=300)
fig.savefig(OUT / 'crop25_t0_vs_tlast_height_color.svg', dpi=300)
plt.close(fig)
Image.open(png).save(OUT/'crop25_t0_vs_tlast_height_color.tif',compression='tiff_lzw',dpi=(300,300))

# The same color limits and actual Z values are used in the interactive viewer.
pf=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
                 column_widths=[.5,.5],horizontal_spacing=.025,
                 subplot_titles=TITLES)
for t,(v,f) in enumerate(meshes):
    pf.add_trace(go.Mesh3d(
        x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        intensity=v[:,2],intensitymode='vertex',colorscale=COLOR_SCALE,
        cmin=0,cmax=40,cauto=False,showscale=t==1,
        colorbar=dict(title=dict(text='Height (µm)',side='right'),tickvals=[0,10,20,30,40],
                      x=1.025,y=.5,len=.68,thickness=16,outlinewidth=.6),
        flatshading=False,lighting=dict(ambient=.78,diffuse=.35,specular=.08,roughness=.85),
        lightposition=dict(x=100,y=-100,z=180),name=TITLES[t],
        hovertemplate='X %{x:.1f} µm<br>Y %{y:.1f} µm<br>Height %{z:.1f} µm<extra></extra>'),row=1,col=t+1)

def axis(title,limits,ticks):
    return dict(title=dict(text=title),range=list(limits),tickvals=ticks,
                showbackground=True,backgroundcolor='#f7f9fa',gridcolor='#ccd5da',
                zerolinecolor='#a6b4bd',showline=True,linecolor='#8799a4',
                showspikes=False)
scene=dict(aspectmode='data',
           xaxis=axis('X (µm)',LIMITS[0],[-40,-20,0,20,40]),
           yaxis=axis('Y (µm)',LIMITS[1],[-40,-20,0,20,40]),
           zaxis=axis('Z / Height (µm)',LIMITS[2],[0,10,20,30,40]),
           camera=dict(eye=dict(x=1.5,y=-1.6,z=1.35),projection=dict(type='orthographic')))
pf.update_layout(scene=scene,scene2=scene,
                 title=dict(text='crop25 | Height-colored cell surface',x=.5),
                 margin=dict(l=15,r=115,b=65,t=90),height=760,
                 paper_bgcolor='white',font=dict(family='Arial',size=12),
                 annotations=[*pf.layout.annotations,dict(
                     text='Blue: lower · Red: higher | Shared 0–40 µm scale | X:Y:Z = 1:1:1 | Drag either cell to rotate both',
                     xref='paper',yref='paper',x=.5,y=-.075,showarrow=False,font=dict(size=12))])
sync="""const p=document.getElementById('{plot_id}');let busy=false;p.on('plotly_relayout',function(e){if(busy)return;let u={};if(e['scene.camera'])u['scene2.camera']=e['scene.camera'];else if(e['scene2.camera'])u['scene.camera']=e['scene2.camera'];else return;busy=true;Plotly.relayout(p,u).then(()=>busy=false);});"""
html=pf.to_html(include_plotlyjs=True,full_html=True,post_script=sync,
                config=dict(responsive=True,displaylogo=False,toImageButtonOptions=dict(format='png',filename='crop25_height_color',width=2200,height=1100,scale=2)))
(OUT/'crop25_interactive_height_color.html').write_text(html,encoding='utf-8')
# Also refresh the file used by the already-open viewer.
(OUT/'crop25_interactive_3D.html').write_text(html,encoding='utf-8')
style=dict(color_quantity='Height in micrometers above the existing shared reference plane',
           color_limits_um=COLOR_LIMITS,colormap='coolwarm',xyz_limits_um=LIMITS,
           vertical_exaggeration=1,geometry_unchanged=True,source_mesh_sha256=hashes,
           view=dict(elevation_deg=32,azimuth_deg=-48),
           note='Minor values below Z=0 belong to the existing smoothed display cut face and are clipped to the lowest color.')
(OUT/'height_color_metadata.json').write_text(json.dumps(style,indent=2),encoding='utf-8')
assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
assert all(trace.cmin==0 and trace.cmax==40 for trace in pf.data)
print(json.dumps(dict(png=str(png),shape=Image.open(png).size,meshes_unchanged=True,shared_scale_um=COLOR_LIMITS)))
