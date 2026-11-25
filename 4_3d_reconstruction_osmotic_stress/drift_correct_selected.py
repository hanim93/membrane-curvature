"""Endpoint translation registration outside the target cell, followed by reconstruction.
Original TIFFs and earlier outputs are read-only inputs.
"""
from hypo_surface_build import *
from scipy.optimize import minimize
from skimage.registration import phase_cross_correlation
import types, shutil, os

def register(condition,n):
    source=ROOT/f'{condition}_surface_output'
    dest=ROOT/'drift_corrected'/f'{condition}_crop{n}'
    dest.mkdir(parents=True,exist_ok=True)
    d=np.load(source/f'crop{n}_endpoints.npz');a=d['first'].astype(float);b=d['last'].astype(float)
    masks=np.load(source/f'crop{n}_masks.npz')
    md=json.loads((source/f'crop{n}_reconstruction.json').read_text())
    target=masks['first'].any(0)|masks['last'].any(0)
    exclude=morphology.dilation(target,morphology.disk(6))
    zz,yy,xx=np.indices(a.shape)
    keep=(~exclude)[None,:,:] & (zz>=3)&(zz<=22)&(yy>=6)&(yy<a.shape[1]-6)&(xx>=6)&(xx<a.shape[2]-6)
    assert keep.sum()>1000
    bg=float(np.percentile(np.concatenate([a.ravel(),b.ravel()]),10))
    def prep(v):
        return np.log1p(np.maximum(ndi.gaussian_filter(v,(.7,1.5,1.5))-bg,0))
    aa,bb=prep(a),prep(b)
    coords=np.array(np.where(keep))[:,::2].astype(float)
    ref=aa[keep][::2]
    def loss(shift,select=None):
        c=coords if select is None else coords[:,select]
        r=ref if select is None else ref[select]
        q=ndi.map_coordinates(bb,c-np.asarray(shift)[:,None],order=1,mode='constant',cval=np.nan)
        good=np.isfinite(q);r=r[good];q=q[good];r=r-r.mean();q=q-q.mean()
        return -float(np.dot(r,q)/np.sqrt(np.dot(r,r)*np.dot(q,q)))
    whole,_,_=phase_cross_correlation(aa,bb,upsample_factor=10,normalization=None)
    bounds=[(-4,4),(-10,10),(-10,10)]
    starts=[np.zeros(3),np.clip(whole,[-4,-10,-10],[4,10,10])]
    fits=[minimize(loss,s,method='Powell',bounds=bounds,options=dict(xtol=.005,ftol=1e-7,maxiter=80)) for s in starts]
    fit=min(fits,key=lambda f:f.fun);shift=fit.x
    # Independently fit alternate spatial blocks; score each on the other half.
    checker=((coords[1]//12+coords[2]//12).astype(int)%2)==0
    checks=[]
    for selection in [checker,~checker]:
        result=minimize(lambda s:loss(s,selection),shift,method='Powell',bounds=bounds,options=dict(xtol=.01,ftol=1e-6,maxiter=60))
        checks.append(dict(shift_zyx_voxels=result.x.tolist(),heldout_ncc_before=-loss([0,0,0],~selection),heldout_ncc_after=-loss(result.x,~selection),success=bool(result.success)))
    # source_tiff below is recorded for provenance metadata only; set RAW_DATA_DIR
    # to the folder containing Crop_hyperosmo/ and Crop_hypoosmo/ if you want it accurate.
    info=dict(condition=condition,crop=n,source_tiff=str(Path(os.environ.get('RAW_DATA_DIR', str(ROOT)))/f'Crop_{condition}osmo'/f'crop{n}.tif'),frame_indices=[0,md['frames']-1],shift_applied_to_last_zyx_voxels=shift.tolist(),shift_applied_to_last_zyx_um=(shift*np.array([md['spacing'],md['xy_um'],md['xy_um']])).tolist(),neighbor_ncc_before=-loss([0,0,0]),neighbor_ncc_after=-fit.fun,whole_stack_shift_zyx_voxels=whole.tolist(),spatial_split_checks=checks,background_offset=bg,exclusion_dilation_px=6,reference_z_indices=[3,22],reference_voxels=int(keep.sum()),bounds_zyx_voxels=bounds,optimizer_success=bool(fit.success),interpretation='Estimated global translation from surrounding membrane signal, not fiducial-validated stage motion; surrounding tissue may itself deform.')
    print(json.dumps(info),flush=True)
    (dest/'registration.json').write_text(json.dumps(info,indent=2))
    assert fit.success and -fit.fun>=-loss([0,0,0])
    assert all(abs(shift[k])<bounds[k][1]-.2 for k in range(3)), 'Registration reached search bound'
    reg=ndi.shift(b,shift,order=1,mode='constant',cval=bg,prefilter=False)
    np.savez_compressed(dest/f'crop{n}_endpoints.npz',first=a,last=reg)
    (dest/f'crop{n}_metadata.json').write_text(json.dumps(md,indent=2))
    np.savez_compressed(dest/'registration_reference.npz',excluded_xy=exclude,reference_mask=keep)
    # Quantitative and visual alignment QA, using the same intensity normalization.
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    raw=[a,b,reg]
    for j,v in enumerate(raw):
        axes[0,j].imshow(v[3:18].max(0),cmap='gray',vmin=bg,vmax=np.percentile(a[3:18],99.5))
        axes[0,j].contour(exclude,levels=[.5],colors=['cyan'],linewidths=.7)
        axes[0,j].set_title(['t=0 (reference)','t=last (before)','t=last (registered)'][j])
    lo,hi=np.percentile(a[3:18].max(0),[5,99.5])
    def scale(v):return np.clip((v-lo)/(hi-lo),0,1)
    for j,v in enumerate([b,reg]):
        rgb=np.zeros((*a.shape[1:],3));rgb[:,:,0]=scale(a[3:18].max(0));rgb[:,:,1]=scale(v[3:18].max(0))
        axes[1,j].imshow(rgb);axes[1,j].contour(exclude,levels=[.5],colors=['cyan'],linewidths=.7)
        axes[1,j].set_title(['XY overlay before','XY overlay after'][j]+' (red: t0, green: last)')
    x=int(a.shape[2]*.8)
    rgb=np.zeros((23,a.shape[1],3));rgb[:,:,0]=scale(a[:23,:,x]);rgb[:,:,1]=scale(reg[:23,:,x])
    axes[1,2].imshow(rgb,aspect=md['spacing']/md['xy_um']);axes[1,2].set_title(f'YZ overlay after, x={x}')
    fig.suptitle(f'{condition} crop{n} | Translation registration | Outside cyan = fit region\nNeighbor NCC: {info["neighbor_ncc_before"]:.3f} → {info["neighbor_ncc_after"]:.3f}')
    fig.tight_layout();fig.savefig(dest/'registration_QC.png',dpi=170);plt.close(fig)
    # Re-run segmentation on registered intensities, with the same basal plane
    # and segmentation settings as the original selected crop.
    code=(ROOT/'hypo_surface_build.py').read_text(encoding='utf-8-sig')
    code=code.replace('seed={16:', 'seed={2:(70,60),16:')
    code=code.replace("base=13",f"base={md['baseline_z_index']}")
    code=code.replace('[5,9,13]',f"[5,9,{md['baseline_z_index']}]")
    if condition=='hyper':
        code=code.replace('else [11,11]','else [10,10]')
        # Apply the same opening to both endpoints to remove narrow bridges
        # to neighboring cell interiors after interpolation.
        code=code.replace('if n==17:', 'if n in (2,17):')
        code=code.replace('4 if n==17 else 0', '4 if n in (2,17) else 0')
        code=code.replace('thr_factor=.55 if n==23 else .65','thr_factor=.55 if n in (2,23) else .65')
    code=code.replace('· hypoosmo ·',f'· {condition}osmo ·')
    code=code.replace('Common acquisition Z plane; no drift correction','Common reference plane; XYZ translation corrected')
    code=code.replace("'None; acquisition coordinates retained'","'XYZ translation; see registration.json'")
    module=types.ModuleType('registered_build');module.__file__=str(ROOT/'hypo_surface_build.py')
    exec(compile(code,module.__file__,'exec'),module.__dict__);module.OUT=dest
    module.build(n)
    meta=json.loads((dest/f'crop{n}_reconstruction.json').read_text());meta['registration_details']=info
    (dest/f'crop{n}_reconstruction.json').write_text(json.dumps(meta,indent=2))
    shutil.copy2(ROOT/'drift_correct_selected.py',dest/'drift_correct_selected.py')

if __name__=='__main__':
    for condition,n in [('hypo',21),('hyper',2)]:register(condition,n)
