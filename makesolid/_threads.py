import numpy as np

from solid import *
import makesolid

__all__ = [
        "M_size",
        "thread_profile",
        "thread_part",
        "thread_entire",
        "thread_pointy",
        ]

M_size_table = {
    1: (0.25,0.2),
    1.2: (0.25,0.2),
    1.4: (0.3,0.2),
    1.6: (0.35,0.2),
    1.8: (0.35,0.2),
    2: (0.4,0.25),
    2.5: (0.45,0.35),
    3: (0.5,0.35),
    3.5: (0.6,0.35),
    4: (0.7,0.5),
    5: (0.8,0.5),
    6: (1,0.75),
    7: (1,0.5),
    8: (1.25,1),
    10: (1.5,1.25),
    12: (1.75,1.5),
    14: (2,1.5),
    16: (2,1.5),
    18: (2.5,2),
    20: (2.5,2),
    22: (2.5,2),
    24: (3,2),
    27: (3,2),
    30: (3.5,2),
    33: (3.5,2),
    36: (4,3),
    39: (4,3),
    42: (4.5,3),
    45: (4.5,3),
    48: (5,3),
    52: (5,4),
    56: (5.5,4),
    60: (5.5,4),
    64: (6,4),
}
preferred_iso_sizes = [1,1.2,1.6,2,2.5,3,4,5,6,8,10,12,16,20,24,30,36,42,48,56,64]
def M_size(D,fine=False):
    c,f = M_size_table(D)
    return D, f if fine else c


def thread_profile(D,P,inset,internal=True,base_pad=0.1):
    """ISO thread profile"""
    H = P*np.sqrt(3)/2
    Dm = D - 2*5*H/8
    Dp = D - 2*3*H/8
    if internal:
        return np.array([
            (-P/2,D/2+H/8+base_pad),
            (-P/2,D/2+H/8+inset),
            (-P/8,Dm/2+inset),
            (P/8,Dm/2+inset),
            (P/2,D/2+H/8+inset),
            (P/2,D/2+H/8+base_pad),
                       ])
    else:
        return np.array([
            (-P/2,Dm/2-H/4-base_pad),
            (-P/2,Dm/2-H/4-inset),
            (-P/16,D/2-inset),
            (P/16,D/2-inset),
            (P/2,Dm/2-H/4-inset),
            (P/2,Dm/2-H/4-base_pad),
                       ])

def thread_part(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64):
    tp = thread_profile(D,P,inset=inset,internal=internal,base_pad=base_pad)
    tp3 = np.zeros((len(tp),3))
    tp3[:,0] = tp[:,1]
    tp3[:,2] = tp[:,0]
    return makesolid.helix_extrude(
        tp3,
        vertical_motion_per_turn=-P,
        turns=turns,
        steps_per_turn=steps_per_turn)

def thread_entire(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64):
    chunks = np.ceil(4*turns)
    parts = []
    tp = thread_part(0.251,D,P,inset,
                     internal=internal,
                     base_pad=base_pad,
                     steps_per_turn=steps_per_turn)
    for i in range(int(chunks)):
        parts.append(
            rotate([0,0,i*90])(
                translate([0,0,i*P/4])(
                    tp)))
    return union()(parts)

def thread_pointy(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64):
    return intersection()(
        translate([0,0,turns*P/2])(cube([4*D,4*D,turns*P],center=True)),
        translate([0,0,-P])(
            thread_entire(turns+2,D,P,inset,
                          internal=internal,
                          base_pad=base_pad,
                          steps_per_turn=steps_per_turn)))

