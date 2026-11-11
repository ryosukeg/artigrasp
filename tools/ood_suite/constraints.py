"""Validate requested goals against collision geometry and hand root limits.

This checks hard geometric constraints, not whether a learned controller will
find a contact-rich motion. No object/hand joint limits or dynamics are changed.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .runtime import Controller, prepare_data, timing_for
from .io import file_sha256


def quaternion_matrix(q):
    w,x,y,z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def vector(value, default=(0, 0, 0)):
    return np.array([float(x) for x in value.split()] if value else default)


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


class CollisionModel:
    def __init__(self, root, name):
        path = Path(root) / ('rsc/arctic/{0}/{0}.urdf'.format(name))
        tree = ET.parse(path).getroot()
        joints = tree.findall('joint')
        children = {j.find('child').get('link') for j in joints}
        self.root = next(l.get('name') for l in tree.findall('link') if l.get('name') not in children)
        self.joints = joints
        self.meshes = []
        self.paths = [path]
        for link in tree.findall('link'):
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                vertices = []
                mesh_path = path.parent / mesh.get('filename')
                self.paths.append(mesh_path)
                with mesh_path.open() as handle:
                    for line in handle:
                        if line.startswith('v '):
                            vertices.append([float(v) for v in line.split()[1:4]])
                vertices = np.array(vertices)*vector(mesh.get('scale'), (1, 1, 1))
                origin = collision.find('origin')
                if origin is not None:
                    rx, ry, rz = vector(origin.get('rpy'))
                    rot = rotation([0,0,1],rz)@rotation([0,1,0],ry)@rotation([1,0,0],rx)
                    vertices = vertices@rot.T + vector(origin.get('xyz'))
                self.meshes.append((link.get('name'), vertices))

    def vertices(self, pose):
        transform = np.eye(4)
        transform[:3,:3] = quaternion_matrix(pose[3:7])
        transform[:3,3] = pose[:3]
        transforms = {self.root:transform}
        pending = self.joints[:]
        while pending:
            for joint in pending[:]:
                parent,child = joint.find('parent').get('link'),joint.find('child').get('link')
                if parent not in transforms:
                    continue
                t = np.eye(4)
                origin = joint.find('origin')
                if origin is not None:
                    rx,ry,rz = vector(origin.get('rpy'))
                    t[:3,:3] = rotation([0,0,1],rz)@rotation([0,1,0],ry)@rotation([1,0,0],rx)
                    t[:3,3] = vector(origin.get('xyz'))
                if joint.get('type') != 'fixed':
                    t[:3,:3] = t[:3,:3]@rotation(vector(joint.find('axis').get('xyz')),pose[7])
                transforms[child]=transforms[parent]@t
                pending.remove(joint)
        return np.concatenate([v@transforms[n][:3,:3].T+transforms[n][:3,3] for n,v in self.meshes])


def validate_plan(plan, repo_root):
    """Resolve small support clearances, then reject impossible hard limits.

    All corrections are planned before any rollout and saved with every target.
    A lift goal remains at initial_z + requested_lift; placement clearance does
    not silently change the experimental lift height.
    """
    repo_root = Path(repo_root)
    settings = plan['config']['physics_validation']
    clearance = float(settings['surface_clearance_m'])
    max_correction = float(settings['maximum_placement_correction_m'])
    output=[]
    mesh_hashes={}
    root_limits=[]
    for hand in ('rhand_mano_demo.urdf','lhand_mano_demo.urdf'):
        tree=ET.parse(repo_root/'rsc/mano_double'/hand).getroot()
        limits=[j.find('limit') for j in tree.findall('joint') if j.get('type')=='prismatic']
        if len(limits)!=3:
            raise ValueError('Expected three translational hand root joints')
        root_limits.append([(float(j.get('lower')),float(j.get('upper'))) for j in limits])
    for name in plan['objects']:
        specs=[s for s in plan['episodes'] if s['object']==name]
        # Data labels are paired, so read each pair once then index by sample.
        base=[s for s in specs if s['profile']==plan['profiles'][0]]
        data=prepare_data(repo_root,base)
        from raisimGymTorch.helper import rotations
        model=CollisionModel(repo_root,name)
        mesh_hashes.update({str(path.relative_to(repo_root)):file_sha256(path) for path in model.paths})
        by_sample={s['sample_index']:i for i,s in enumerate(base)}
        for spec in specs:
            i=by_sample[spec['sample_index']]
            one={k:[v[i:i+1].copy() for v in data[k]] for k in ('grasp','articulation')}
            one.update({k:data[k][i:i+1].copy() for k in ('left','right','arti_left','arti_right')})
            c=Controller(None,one,[spec],timing_for(name))
            initial=one['grasp'][2][0]
            iv=model.vertices(initial)
            goal=c.goal[0].copy();goal[7]=c.angle[0,0]
            gv=model.vertices(goal)
            lift=c.lift_goal[0].copy();lift[7]=0
            lv=model.vertices(lift)
            # stage is 3x3m, centered at (0.75,0,0.551), thickness .1m.
            stage_top=float(one['grasp'][1][0,2]+one['grasp'][0][0,2]/2)
            correction=max(0.0,stage_top+clearance-float(gv[:,2].min()))
            if correction>max_correction:
                raise ValueError('{}: support clearance requires {:.4f} m (maximum {:.4f})'.format(
                    spec['episode_id'],correction,max_correction))
            spec['target']['placement_z_offset_m']=correction
            goal[2]+=correction
            gv[:,2]+=correction
            c.goal[0,2]+=correction
            tolerance=0.00002
            below=float(stage_top-gv[:,2].min())
            inside=bool(np.all(gv[:,:2].min(axis=0)>[-.75,-1.5]) and np.all(gv[:,:2].max(axis=0)<[2.25,1.5]))
            deltas=[]
            roots_valid=True
            for target in (c.goal[0],c.lift_goal[0]):
                base_r=rotations.quat2mat(initial[3:7]);goal_r=rotations.quat2mat(target[3:7])
                for hi,hand in enumerate((one['grasp'][3][0],one['grasp'][4][0])):
                    wrist=target[:3]+goal_r@base_r.T@(hand[:3]-initial[:3])
                    local=rotations.euler2mat(hand[3:6]).T@(wrist-hand[:3])
                    deltas.append(float(np.max(np.abs(local))))
                    margin=float(settings['wrist_translation_margin_m'])
                    roots_valid &= all(lo+margin<=v<=high-margin for v,(lo,high) in zip(local,root_limits[hi]))
            output.append(dict(episode_id=spec['episode_id'],object=name,profile=spec['profile'],
                               stage_penetration_m=below,tolerance_m=tolerance,
                               initial_min_z_m=float(iv[:,2].min()),final_min_z_m=float(gv[:,2].min()),
                               lift_min_z_m=float(lv[:,2].min()),within_stage=inside,
                               placement_z_offset_m=correction,
                               within_hand_root_limits=bool(roots_valid),max_hand_translation_m=max(deltas),
                               collision_clear=below<=tolerance,
                               target_within_articulation_limits=bool(spec['target_reachable'])))
        group=[r for r in output if r['object']==name]
        print(name,'collision_fail',sum(not r['collision_clear'] for r in group),
              'stage_fail',sum(not r['within_stage'] for r in group),
              'wrist_fail',sum(not r['within_hand_root_limits'] for r in group),
              'max_root',round(max(r['max_hand_translation_m'] for r in group),3),flush=True)
    failed=[r for r in output if not all(r[k] for k in (
        'within_stage','within_hand_root_limits','collision_clear','target_within_articulation_limits'))]
    if failed:
        raise ValueError('Hard goal constraints failed: '+str([r['episode_id'] for r in failed[:10]]))
    return dict(checked=len(output),failed=len(failed),goals=output,mesh_sha256=mesh_hashes,
                hand_translation_limits=root_limits,
                object_height_cap=None,
                note='Object base is floating; no height clamp. Root target clipping is checked in hand-local coordinates. '
                     'Collision geometry clears support at the final target. This is a hard-limit check, '
                     'not a guarantee that a learned policy finds a successful trajectory.')
