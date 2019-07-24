import numpy as np

from solid import *
import makesolid

__all__ = [
        "preferred_iso_sizes",
        "M_size",
        "thread_dimensions",
        "thread_profile",
        "thread_part",
        "thread_entire",
        "thread_pointy",
        "thread_chamfered",
        "thread_tip",
        "thread_tapered",
        "thread",
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
    c,f = M_size_table[D]
    return D, (f if fine else c)


class _ThreadDimensions(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self,k,v)

    def __repr__(self):
        return ("<%s " % self.__class__.__qualname__
                + " ".join(["%s=%s" % (v, getattr(self,v)) for v in vars(self)])
                + ">")

def thread_dimensions(D,P,
        D_inner_thread = None,
        D_outer_thread = None,
        turns=1,
        starts=1,
        inset=0.25,
        base_pad=0.1,
        wall_thick=2,
        top_end="flat",
        bottom_end="flat",
        steps_per_turn=64,
        ):

    H = P*np.sqrt(3)/2
    if D is not None:
        D_inner_thread = D-7*H/4
    elif D_outer_thread is not None:
        D_inner_thread = D_outer_thread - 2*H
    D = D_inner_thread + 7*H/4
    D_outer_thread = D_inner_thread + 2*H
    screw_thick = (turns*starts+3)*P

    Dm = D - 2*5*H/8
    Dp = D - 2*3*H/8

    Di = D - 7*H/4
    Do = Di + 2*H
    _extras = dict(
            flat=0,
            chamfer=P/2,
            taper=P/2+starts*P/8,
        )
    extra_top_height = _extras[top_end] 
    extra_bottom_height = _extras[bottom_end] 
    extra_height = (extra_top_height + extra_bottom_height)
    del _extras
    h = (turns*starts)*P

    return _ThreadDimensions(**locals())

def thread_profile(D,P,inset,internal=True,base_pad=0.1):
    """ISO thread profile"""
    H = P*np.sqrt(3)/2
    Dm = D - 2*5*H/8
    Dp = D - 2*3*H/8
    if internal:
        return np.array([
            (-P/2,D/2+H/8+base_pad+inset),
            (-P/2,D/2+H/8+inset),
            (-P/8,Dm/2+inset),
            (P/8,Dm/2+inset),
            (P/2,D/2+H/8+inset),
            (P/2,D/2+H/8+base_pad+inset),
                       ])
    else:
        return np.array([
            (-P/2,Dm/2-H/4-base_pad-inset),
            (-P/2,Dm/2-H/4-inset),
            (-P/16,D/2-inset),
            (P/16,D/2-inset),
            (P/2,Dm/2-H/4-inset),
            (P/2,Dm/2-H/4-base_pad-inset),
                       ])

def thread_part(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64,starts=1):
    tp = thread_profile(D,P,inset=inset,internal=internal,base_pad=base_pad)
    tp3 = np.zeros((len(tp),3))
    tp3[:,0] = tp[:,1]
    tp3[:,2] = tp[:,0]
    return makesolid.helix_extrude(
        tp3,
        vertical_motion_per_turn=-P*starts,
        turns=turns,
        steps_per_turn=steps_per_turn)

def thread_entire(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64,starts=1):
    chunks = np.ceil(4*turns)
    parts = []
    tp = thread_part(0.251,D,P,inset,
                     internal=internal,
                     base_pad=base_pad,
                     steps_per_turn=steps_per_turn,
                     starts=starts)
    for i in range(int(chunks)):
        for j in range(starts):
            parts.append(
                rotate([0,0,i*90+j*360/starts])(
                    translate([0,0,i*starts*P/4])(
                        tp)))
    return union()(parts)

def thread_pointy(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64,starts=1):
    return intersection()(
        translate([0,0,turns*starts*P/2])(cube([4*D,4*D,turns*starts*P],center=True)),
        translate([0,0,-P])(
            thread_entire(turns+2,D,P,inset,
                          internal=internal,
                          base_pad=base_pad,
                          steps_per_turn=steps_per_turn,
                          starts=starts)))

def thread_chamfered(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64,starts=1):
    """Chamfer the top with a 30-degree chamfer"""
    t = translate([0,0,-P])(
            thread_entire(turns+4/starts,D,P,inset,
                          internal=internal,
                          base_pad=base_pad,
                          steps_per_turn=steps_per_turn,
                          starts=starts))
    H = P*np.sqrt(3)/2
    Di = D - 7*H/4
    Do = Di + 2*H
    h = (turns*starts+1)*P
    if internal:
        hole = translate([0,0,h-Do/(2*np.sqrt(3))])(
                        cylinder(d1=0, d2=Do, h=Do/(2*np.sqrt(3)), segments=steps_per_turn))
        return intersection()(
                difference()(
                    t,
                    hole),
                translate([0,0,h/2])(cube([4*Do,4*Do,h],center=True)))
    else:
        chop = cylinder(d1=Di+h*2*np.sqrt(3), d2=Di, h=h, segments=steps_per_turn)
        return intersection()(t, chop)
                
def thread_tip(D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64,starts=1):
    H = P*np.sqrt(3)/2
    Di = D - 7*H/4
    Do = Di + 2*H

    turns = 0.25
    tp = thread_profile(D,P,inset=inset,internal=internal,base_pad=base_pad)
    tp3 = np.zeros((len(tp),3))
    tp3[:,0] = tp[:,1]
    tp3[:,2] = tp[:,0]
    if internal:
        f = 2
        d = f*D
        t = translate([(D-d)/2,0,0])(
                makesolid.helix_extrude(
                    tp3+np.array([(d-D)/2,0,0])[None,:],
                    vertical_motion_per_turn=-P*starts*f,
                    turns=turns,
                    steps_per_turn=steps_per_turn))
        return intersection()(t,
                cylinder(d=Do+inset+2*base_pad, h=P+D, center=True, segments=steps_per_turn))
    else:
        f = 5/8
        d = f*D
        t = translate([(D-d)/2,0,0])(
                makesolid.helix_extrude(
                    tp3+np.array([(d-D)/2,0,0])[None,:],
                    vertical_motion_per_turn=-P*starts*f,
                    turns=turns,
                    steps_per_turn=steps_per_turn))
        return difference()(t,
                cylinder(d=Di-inset-2*base_pad, h=P+D, center=True, segments=steps_per_turn))

def thread_tapered(turns,D,P,inset,internal=True,base_pad=0.1,steps_per_turn=64,starts=1):
    """Taper the top starts"""
    chunks = np.ceil(4*(turns+4/starts))
    t = thread_part(0.251,
                    D,P,inset,
                    internal=internal,
                    base_pad=base_pad,
                    steps_per_turn=steps_per_turn,
                    starts=starts)
    p = rotate([180,0,0])(
            thread_tip(D,P,inset,
                   internal=internal,
                   base_pad=base_pad,
                   steps_per_turn=steps_per_turn,
                   starts=starts))
 
    H = P*np.sqrt(3)/2
    Di = D - 7*H/4
    Do = Di + 2*H
    h = (turns*starts+1)*P

    parts = [p]
    for i in range(int(chunks)):
        parts.append(
            translate([0,0,-i*P*starts/4])(
                rotate([0,0,-90*i])(t)))
    si = intersection()(
            union()(parts),
            cube([2*D,2*D,2*h],center=True))
    return translate([0,0,h])(
            union()([rotate([0,0,i*360/starts])(si) for i in range(starts)]))

def thread(turns,D_inner_thread,P,inset,
        internal=True,base_pad=0.1,steps_per_turn=64,starts=1,
        turns_pad=0.001,
        top_end="flat",
        bottom_end="flat",
        all_starts=True):
    """Construct a thread

    `turns` is the number of complete turns in the "normal" part of the thread
    (that is, unaffected by taper, chamfer, or cutting flat)
    """
    td = thread_dimensions(
            turns=turns,D=None,D_inner_thread=D_inner_thread,P=P,
            inset=inset,base_pad=base_pad,starts=starts,
            top_end=top_end, bottom_end=bottom_end)
    turns = td.turns
    starts = td.starts
    D = td.D
    h = td.h
    xt = dict(flat=1,
              chamfer=1,
              taper=0)
    tp = turns + (xt[top_end] + xt[bottom_end])/starts
    if top_end=="taper":
        # Only in this case we need to worry about the spiral being too long
        chunks = int(np.ceil(4*tp))-1
    else:
        chunks = int(np.ceil(4*tp))
    t = thread_part(0.25 + turns_pad,
                    D,P,inset,
                    internal=internal,
                    base_pad=base_pad,
                    steps_per_turn=steps_per_turn,
                    starts=starts)
    # If we extended the spiral to accomodate the bottom end design,
    # we also need to rotate it so that the top end is at the angle we
    # expect. This includes both during initial construction and
    # when adding the last fractional segment.
    down = xt[bottom_end]
    s = rotate([0,0,-down*360/starts])(
            translate([0,0,-down*P])(
                rotate([180,0,0])(
                    union()([
                        translate([0,0,-i*starts*P/4])(
                            rotate([0,0,-i*90])(t))
                    for i in range(chunks)]))))
    if top_end=="taper":
        # Only in this case we need to worry about the spiral being too long
        tf = tp - chunks/4
        if tf>0:
            r = 10*td.D_outer_thread
            i = chunks
            s += rotate([180,0,0])(
                    translate([0,0,-i*starts*P/4+xt[bottom_end]*P])(
                        rotate([0,0,down*360/starts-i*90])(
                            t*linear_extrude(height=8*P*starts,center=True)(
                                polygon([
                                    (0,0),
                                    (r,0),
                                    (r*np.cos(2*np.pi*tf), -r*np.sin(2*np.pi*tf)), 
                                    ])))))
    tip = thread_tip(D,P,inset,
                   internal=internal,
                   base_pad=base_pad,
                   steps_per_turn=steps_per_turn,
                   starts=starts)
    if bottom_end=="flat":
        hh = 2*h+8*P
        s_b = (s * 
            translate([0,0,hh/2])(
                cube([4*td.D_outer_thread, 4*td.D_outer_thread, hh], center=True)))
    elif bottom_end=="chamfer":
        if internal:
            hh = h+8*P
            f = 2*np.sqrt(3)
            s_b = (s - 
                translate([0,0,-hh])(
                    cylinder(
                        d1=td.D_outer_thread+f*(hh-P/2),
                        d2=td.D_outer_thread-f*P/2,
                        h=hh,
                        segments=steps_per_turn)))
        else:
            hh = h+8*P
            f = 2*np.sqrt(3)
            s_b = (s * translate([0,0,-P/2])(
                cylinder(
                    d1=td.D_outer_thread-f*P/2,
                    d2=td.D_outer_thread+f*(hh-P/2),
                    h=hh,
                    segments=steps_per_turn)))
    elif bottom_end=="taper":
        s_b = s + tip
    else:
        raise ValueError("Uknown bottom end mode %s" % bottom_end)

    if top_end=="flat":
        hh = 2*h+8*P
        s_a = (s_b * 
            translate([0,0,h-hh/2])(
                cube([4*td.D_outer_thread, 4*td.D_outer_thread, hh], center=True)))
    elif top_end=="chamfer":
        if internal:
            hh = h+8*P
            f = 2*np.sqrt(3)
            s_a = (s_b - 
                translate([0,0,h])(
                    cylinder(
                        d1=td.D_outer_thread-f*P/2,
                        d2=td.D_outer_thread+f*(hh-P/2),
                        h = hh,
                        segments=steps_per_turn)))
        else:
            hh = h+8*P
            f = 2*np.sqrt(3)
            s_a = (s_b * 
                translate([0,0,h+P/2-hh])(
                    cylinder(
                        d1=td.D_outer_thread+f*(hh-P/2),
                        d2=td.D_outer_thread-f*P/2,
                        h = hh,
                        segments=steps_per_turn)))
    elif top_end=="taper":
        s_a = (s_b + 
            rotate([0,0,360*turns])(
                translate([0,0,turns*starts*P])(
                    rotate([180,0,0])(
                        tip))))
    else:
        raise ValueError("Uknown top end mode %s" % top_end)


    if all_starts:
        return translate([0,0,td.extra_bottom_height])(
                union()([rotate([0,0,i*360/starts])(s_a) for i in range(starts)]))
    else:
        return translate([0,0,td.extra_bottom_height])(s_a)
