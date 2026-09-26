#!/usr/bin/env python3
"""Non-destructive TUM overlay: optical +Z defines horizontal forward."""
import argparse,json,pickle,os,hashlib
from pathlib import Path
import numpy as np
from functools import lru_cache


def camera_forward_yaw(quaternion):
    q=np.asarray(quaternion,dtype=np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=q.T
    fx,fy=2*(x*z+w*y),2*(y*z-w*x)
    if np.any(np.hypot(fx,fy)<1e-6):raise ValueError('Camera points vertically; planar yaw undefined')
    return np.arctan2(fy,fx)


@lru_cache(maxsize=32)
def heading_from_metadata(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    return camera_forward_yaw([r['quaternion_xyzw'] for r in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);records=[]
    for child in a.source.iterdir():
        if not child.is_dir():
            target=a.output/child.name
            if not target.exists():target.symlink_to(child)
            continue
        metadata=child/'frame_metadata.jsonl'
        if not metadata.exists():continue
        rows=[json.loads(line) for line in metadata.read_text().splitlines()]
        old=pickle.loads((child/'traj_data.pkl').read_bytes())
        yaw=camera_forward_yaw([r['quaternion_xyzw'] for r in rows])
        if len(yaw)!=len(old['yaw']):raise ValueError(child)
        out=a.output/child.name;out.mkdir(parents=True,exist_ok=True)
        for f in child.iterdir():
            if f.name=='traj_data.pkl':continue
            dest=out/f.name
            if not dest.exists():dest.symlink_to(f)
        new={**old,'yaw':yaw};(out/'traj_data.pkl').write_bytes(pickle.dumps(new,protocol=4))
        records.append(dict(trajectory=child.name,frames=len(yaw),source_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),new_pose_sha256=hashlib.sha256((out/'traj_data.pkl').read_bytes()).hexdigest(),median_heading_change_radians=float(np.median(np.arctan2(np.sin(yaw-old['yaw']),np.cos(yaw-old['yaw']))))))
    (a.output/'camera_heading_correction.json').write_text(json.dumps(dict(source=str(a.source),definition='yaw=atan2((R(q)*optical_z).y,(R(q)*optical_z).x); world XY plane',trajectories=records),indent=2))
    print('Corrected',len(records),'trajectories',flush=True)

if __name__=='__main__':main()
