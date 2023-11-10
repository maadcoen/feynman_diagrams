import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as mpatch
import time
import logging
from typing import List

saved = 0
save_to = '/Users/maadcoen/Documents/PhD/teaching/SubA_II/suba_exercises/images'
logfile = 'feynman_QED.log'
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.INFO, force=True,
                    format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')


class SelectObject:
    def __init__(self, parent=None, ax=None):
        self.select_time = None
        self._parent = parent
        self.moving = False
        self.ax = self.parent.ax if parent is not None else (plt.gca() if ax is None else ax)

    @property
    def is_selected(self):
        return self.select_time is not None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    def select(self):
        if self.is_selected:
            self.select_time = time.time()
            return
        logging.info(f'selecting {self}')
        self.select_time = time.time()
        if self.parent is not None:
            self.parent.select()
        self.moving = True
        return self

    def deselect(self):
        if not self.is_selected:
            return None
        logging.info(f'deselecting {self}')
        self.select_time = None
        if self.parent is not None:
            self.parent.deselect()
        self.moving = False
        return self

    def freeze(self):
        self.moving = False


selected_object: SelectObject = None
copied_text = None


class Target(SelectObject):
    def __init__(self, parent, rel_loc=(0, 0), radius=0.15, passive_color=None, active_color='red', ax=None):
        super().__init__(parent, ax)
        self.rel_loc = rel_loc
        self.orginal_rel_loc = rel_loc
        self.active_color = active_color
        self.radius = radius
        self.passive_color = passive_color
        self._patch = None
        self._labels = []

    @property
    def connect(self):
        return None

    def __str__(self):
        return f'target belonging to {self.parent}'

    def _default_loc(self):
        return self.parent.loc[0] + self.rel_loc[0], self.parent.loc[1] + self.rel_loc[1]

    @property
    def patch(self):
        if self._patch is None:
            logging.info(f'creating patch for {self}')
            self._patch = mpatch.Circle(self._default_loc(), radius=self.radius, picker=True,
                                        visible=self.passive_color is not None, alpha=0.3,
                                        color=self.active_color if self.passive_color is None else self.passive_color)
            self.ax.add_patch(self._patch)
        return self._patch

    @patch.deleter
    def patch(self):
        if self._patch is None:
            logging.warning(f'trying to remove patch for {self} but no patch present')
            return
        logging.info(f'removing patch for {self}')
        self._patch.remove()
        self._patch = None

    @property
    def loc(self):
        return self.patch.center

    @loc.setter
    def loc(self, loc):
        self.move(loc, force=True)

    @property
    def labels(self) -> List['Text']:
        return self._labels

    @labels.deleter
    def labels(self):
        for l in self.labels:
            l.remove()
        self._labels = []

    def deselect(self):
        if super().deselect() is None:
            return
        if self.passive_color is None:
            self.patch.set_visible(False)
        else:
            self.patch.set_color(self.passive_color)
        self.patch.set_zorder(-1)
        for l in self.labels:
            l.deselect()
        return self

    def select(self):
        if super().select() is None:
            return
        self.patch.set_visible(True)
        self.patch.set_color(self.active_color)
        self.patch.set_zorder(np.inf)
        return self

    def __eq__(self, other):
        return other == self.patch

    def move(self, loc=None, force=False, redraw=True, restore=False):
        if not self.moving and not force:
            return
        if restore:
            self.rel_loc = self.orginal_rel_loc
            loc = None
        if loc is not None:
            self.rel_loc = (loc[0] - self.parent.loc[0], loc[1] - self.parent.loc[1])
            self.patch.set_center(loc)
        else:
            self.patch.set_center(self._default_loc())
        for l in self.labels:
            l.move(force=True)

    def remove(self):
        logging.info(f'removing target of {self}')
        del self.patch
        del self.labels

    def __copy__(self):
        return Target(self._default_loc, self.patch.radius, self.passive_color, self.active_color, self.ax)

    def hit(self, click_event):
        if click_event.artist == self:
            return self
        for l in self.labels:
            if click_event.artist == l:
                return l


class VertexTarget(Target):
    def __init__(self, vertex, rel_loc=(0, 0), radius=0.15, passive_color=None, active_color='red', ax=None,
                 arrow_out=None, shape=None):
        self.rel_loc = rel_loc
        ax = vertex.ax if vertex is not None else ax
        self.vertex = vertex
        super().__init__(vertex, rel_loc, radius, passive_color, active_color, ax)
        self.vertex = self.parent
        self._connect = None
        self.index = len(self.vertex.targets)
        self._leg = None if not self.index else Leg(self, arrow_out, shape)

    def __str__(self):
        return f'VertexTarget {self.index} from {self.vertex}'

    @property
    def leg(self):
        return self._leg

    @leg.deleter
    def leg(self):
        if self._leg is not None:
            self.leg.remove()
        self.leg.visible = False

    @property
    def arrow_out(self):
        return self.leg.arrow_out

    @arrow_out.setter
    def arrow_out(self, arrow_out):
        self.leg.arrow_out = arrow_out

    @property
    def shape(self):
        return self.leg.shape

    @shape.setter
    def shape(self, shape):
        self.leg.shape = shape

    @property
    def connect(self):
        return self._connect

    @connect.setter
    def connect(self, target):
        logging.info(f'connecting {self} to {target}')
        target: Target
        self.patch.set_picker(False)
        self._connect = target
        self.move(target.vertex.loc, force=True)

        target: VertexTarget
        target._connect = self
        target.move(self.vertex.loc, force=True)
        del target.leg
        target.patch.set_picker(False)

    @connect.deleter
    def connect(self):
        if self.connect is None:
            logging.warning(f'tried disconnecting but no connection')
            return
        logging.info(f'disconnecting {self} from {self.connect}')

        target: VertexTarget = self.connect
        self._connect = None
        logging.info(f'restoring position {self}')
        self.move(force=True, restore=True)
        # self.patch.set_picker(True)
        self.leg.visible = True
        self.leg.deselect()

        del target.connect

    def move(self, loc=None, force=False, redraw=True, restore=False):
        if loc is None and self.connect is not None:
            self.connect.move(self.vertex.loc, force)
        else:
            super().move(loc, force, redraw, restore)
        if self.leg is not None:
            self.leg.make_path()

    def remove(self):
        del self.connect
        if self.leg is not None:
            self.leg.remove()
        super().remove()

    def __copy__(self):
        return VertexTarget(self.vertex, self.rel_loc, self.patch.radius, self.passive_color, self.active_color,
                            self.ax)

    def hit(self, click_event):
        if self.connect is None:
            return super().hit(click_event)
        for t in [self, self.connect]:
            leg_hit = t.leg.hit(click_event)
            if leg_hit is not None:
                return leg_hit

    def deselect(self):
        if self.leg is not None:
            self.leg.deselect()
        if super().deselect() is None:
            return
        return self

    def select(self):
        if self.connect is not None:
            if self.leg is not None: # still select leg if connected
                self.leg.select()
            return
        if super().select() is None: # only selectable if not connected
            return
        if self.leg is not None:
            self.leg.select()
        return self


class LegTarget(Target):
    def __init__(self, leg, rel_loc=(0, 0), radius=0.15, passive_color=None, active_color='red', ax=None):
        super().__init__(leg, rel_loc, radius, passive_color, active_color, ax)
        self.leg: Leg = self.parent

    def move(self, loc=None, force=False, redraw=True, restore=False):
        if loc is not None:
            if abs(self.leg.signed_distance(loc)) > self.leg.path_length(*self.leg.path) / 2:
                return
            n0, n1 = self.leg.normal()
            x, y = self.leg.loc
            r = n0 * (loc[0] - x) + n1 * (loc[1] - y)
            loc = x + r * n0, y + r * n1
        super().move(loc, force, redraw, restore)
        if redraw:
            self.leg.make_path()

    @property
    def arrow_out(self):
        return self.leg.arrow_out

    @arrow_out.setter
    def arrow_out(self, arrow_out):
        self.leg.arrow_out = arrow_out
        if self.leg.leg_target.connect is not None:
            opposite = None if arrow_out is None else ~arrow_out
            self.leg.leg_target.connect.leg.arrow_out = opposite

    @property
    def shape(self):
        return self.leg.shape

    @shape.setter
    def shape(self, shape):
        self.leg.shape = shape
        if self.leg.leg_target.connect is not None:
            self.leg.leg_target.connect.leg.arrow_out = shape

    def disconnect(self):
        self.leg.disconnect()


class Leg(SelectObject):
    def __init__(self, target: VertexTarget, arrow_out=None, shape=None, color='black'):
        super().__init__(target)
        self.parent = None
        self.target = target
        self.vertex = self.target.vertex
        self.color = color
        self._patch = None
        self._arrow_out = arrow_out
        self._arrow_patch = None
        self._shape = shape
        self._visible = True
        self._leg_target = None
        self.make_path()

    def __str__(self):
        return f'leg from {self.target}'

    @property
    def patch(self):
        if self._patch is None:
            self.make_path()
        return self._patch

    @patch.setter
    def patch(self, path):
        if self._patch is None:
            logging.info(f'creating patch for {self}')
            self._patch = mpatch.PathPatch(path, **self.vertex.patch_kwargs, fill=False)
            self.vertex.ax.add_patch(self._patch)
        else:
            self._patch.set_path(path)

    @patch.deleter
    def patch(self):
        if self._patch is not None:
            self._patch.remove()
            self._patch = None

    @property
    def leg_target(self):
        if self._leg_target is None:
            self._leg_target = LegTarget(self, (0, 0), self.target.patch.radius,
                                         self.target.passive_color, self.target.active_color, self.target.ax)
        return self._leg_target

    @leg_target.deleter
    def leg_target(self):
        if self._leg_target is not None:
            self._leg_target.remove()
            self._leg_target = None

    @property
    def shape(self):
        return 'line' if self._shape is None else self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.make_path()

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible):
        logging.info(f'making {self} {"" if visible else "in"}visible')
        self._visible = visible
        self.make_path()

    @property
    def arrow_out(self):
        return self._arrow_out

    @arrow_out.setter
    def arrow_out(self, arrow_out):
        self._arrow_out = arrow_out
        self.draw_arrow()

    @property
    def arrow_patch(self):
        if self._arrow_patch is None:
            self.draw_arrow()
        return self._arrow_patch

    @arrow_patch.setter
    def arrow_patch(self, path):
        if self._arrow_patch is None:
            self._arrow_patch = mpatch.PathPatch(path, color=self.color)
            self.vertex.ax.add_patch(self._arrow_patch)
        else:
            self._arrow_patch.set_path(path)

    @arrow_patch.deleter
    def arrow_patch(self):
        if self._arrow_patch is not None:
            self._arrow_patch.remove()
            self._arrow_patch = None

    @property
    def path(self):
        return self.vertex.loc, self.target.loc

    def make_path(self):
        if not self.visible:
            self.remove()
            return

        self.leg_target.move(redraw=False, force=True)
        p0, pt = self.target.vertex.loc, self.target.loc
        l0 = self.path_length(p0, pt)
        shape = self.shape
        if l0 == 0:
            t = np.linspace(0, 1, 2)
            shape = 'line'
        else:
            t = np.arange(0, l0, 1e-3)
        codes = [Path.MOVETO] + [Path.LINETO for _ in t[1:]]

        if 'photon'.startswith(shape):
            x = t + 4
            y = 0.08 * np.sin(5 * np.pi * t)
        elif 'gluon'.startswith(shape):
            theta, r, w = np.pi / 4, 0.05, 2
            w = (2 * w + 1) * np.pi
            x = r * np.cos(w * t) * np.cos(theta) - 5 * r * t * np.sin(theta)
            y = 3 * r * np.sin(w * t)
        elif 'higgs'.startswith(shape):
            n = int(l0 // 0.1)
            if not n % 2:
                n += 1
            x = np.linspace(0, l0, n)
            codes = [Path.MOVETO]
            for i in range(1, n, 2):
                codes.extend([Path.LINETO, Path.MOVETO])
            y = 0 * x
        else:
            x = t
            y = 0 * t

        line_vertices = np.array([x, y])
        line_vertices -= line_vertices[:, :1]
        th0 = self.path_angle(p0, pt)
        th1 = self.path_angle(line_vertices[:, 0], line_vertices[:, -1])
        l1 = self.path_length(line_vertices[:, 0], line_vertices[:, -1])
        trans = np.array([[np.cos(th1), np.sin(th1)],
                          [-np.sin(th1), np.cos(th1)]])
        line_vertices = trans @ line_vertices
        line_vertices[0] /= l1
        line_vertices[1, -1] = 0
        if self.target.connect is not None:
            l2 = self.signed_distance(self.leg_target.loc)
            if np.abs(l2) > 1e-4:
                r = ((l0 / 2) ** 2 + l2 ** 2) / (l2 * 2)
                th = np.arcsin((l0 / 2) / r)
                xx = (x - x[0]) / (x[-1] - x[0])
                trans = (y + r) * np.exp(-1j * (xx * 2 * th + np.pi / 2 - th))
                line_vertices = np.array([trans.real + trans.real[0], trans.imag - trans.imag[0]])

                # th1 = self.path_angle(line_vertices[:, 0], line_vertices[:, -1])
                # l1 = self.path_length(line_vertices[:, 0], line_vertices[:, -1])
                # trans = np.array([[np.cos(th1), np.sin(th1)],
                #                   [-np.sin(th1), np.cos(th1)]])
                # line_vertices = trans @ line_vertices
                # line_vertices[0] *= l0 / l1
            else:
                line_vertices[0] *= l0
        else:
            line_vertices[0] *= l0

        trans = np.array([[np.cos(th0), -np.sin(th0)],
                          [np.sin(th0), np.cos(th0)]])
        line_vertices = (trans @ line_vertices + np.array(self.target.vertex.loc).reshape((2, 1))).T
        path = Path(line_vertices, codes)
        self.patch = path
        self.draw_arrow()
        return path

    def path_angle(self, p0, pt):
        return np.arctan2(pt[1] - p0[1], pt[0] - p0[0])

    def path_length(self, p0=None, pt=None):
        if p0 is None or pt is None:
            p0, pt = self.path
        return np.sqrt((pt[1] - p0[1]) ** 2 + (pt[0] - p0[0]) ** 2)

    def normal(self):
        p0, pt = self.path
        l0 = self.path_length()
        return (pt[1] - p0[1]) / l0, -(pt[0] - p0[0]) / l0

    def signed_distance(self, p):
        pt, p0 = self.target.loc, self.vertex.loc
        n0, n1 = self.normal()
        return n0 * p[0] + n1 * p[1] + (pt[0] * p0[1] - pt[1] * p0[0]) / self.path_length()

    @property
    def loc(self):
        p0, pt = self.vertex.loc, self.target.loc
        return (pt[0] + p0[0]) / 2, (pt[1] + p0[1]) / 2

    def draw_arrow(self, arr_ang=30, length=0.2):
        if not self.visible:
            return
        if self.arrow_out is None:
            del self.arrow_patch
            return
        p0, pt = self.vertex.loc, self.target.loc
        line_angle = self.path_angle(p0, pt)
        arr_ang *= np.pi / 180
        if self.arrow_out:
            arr_ang = np.pi - arr_ang
        arr_ang_1 = line_angle + arr_ang
        arr_ang_2 = line_angle - arr_ang
        d1 = (length * np.cos(arr_ang_1), length * np.sin(arr_ang_1))
        d2 = (length * np.cos(arr_ang_2), length * np.sin(arr_ang_2))
        pc = (self.leg_target.loc[0] - (d1[0] + d2[0]) / 3, self.leg_target.loc[1] - (d1[1] + d2[1]) / 3)
        p1 = (pc[0] + d1[0], pc[1] + d1[1])
        p2 = (pc[0] + d2[0], pc[1] + d2[1])
        path = Path([p1, pc, p2], [Path.MOVETO, Path.LINETO, Path.LINETO])
        self.arrow_patch = path

    def remove(self):
        logging.info(f'removing {self}')
        del self.patch
        del self.arrow_patch
        del self.leg_target

    def select(self):
        if super().select() is None:
            return
        if not self.visible:
            return self.target.connect.leg.select()
        self.patch.set_visible(True)
        self.patch.set_color(self.target.active_color)
        self.patch.set_zorder(np.inf)
        if self.arrow_patch is not None:
            self.arrow_patch.set_zorder(np.inf)
            self.arrow_patch.set_color(self.target.active_color)
        return self

    def deselect(self):
        if super().deselect() is None:
            return
        if self.target.connect is not None:
            logging.info(f'{self} already deselected')
            self.target.connect.leg.deselect()
        if not self.visible:
            return
        self.patch.set_color(self.target.passive_color)
        self.patch.set_zorder(-1)
        if self.arrow_patch is not None:
            self.arrow_patch.set_zorder(-1)
            self.arrow_patch.set_color(self.color)

    def hit(self, click_event):
        if click_event.artist == self.leg_target:
            return self.leg_target
        for l in self.leg_target.labels:
            if click_event.artist == l:
                return l

    def disconnect(self):
        del self.target.connect


class Text(Target):
    fontsize = 15
    def __init__(self, text, target, rel_loc=(0, 0), radius=0.15, text_passive_color='black', active_color='red',
                 **text_args):
        super().__init__(target, rel_loc, radius, None, active_color)
        self.text_passive_color = text_passive_color
        self._text = text
        def_text_args = dict(ha='center', va='center', fontsize=Text.fontsize, color=text_passive_color)
        def_text_args |= text_args
        self._text_patch = target.ax.text(*self.loc, text, **def_text_args)
        self._target = self.parent
        self.target.labels.append(self)
        self._pointer = 0
        self.pointer_symbol = '/'


    def move(self, loc=None, force=False, redraw=True, restore=False):
        super().move(loc, force, redraw, restore)
        self._text_patch.set_position(self.loc)

    @property
    def pointer(self):
        return self._pointer

    @pointer.setter
    def pointer(self, p):
        self._pointer = p
        self.draw()

    @property
    def target(self) -> Target:
        return self._target

    @target.setter
    def target(self, t):
        if self in self.target.labels:
            self.target.labels.remove(self)
        self.parent = t
        self._target = t
        self.target.labels.append(self)
        self.move(force=True)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
        self.draw()

    def select(self):
        if super().select() is None:
            return
        self.patch.set_visible(False)
        self.patch.set_color(self.passive_color)
        self._text_patch.set_color(self.target.active_color)
        self._text_patch.set_zorder(np.inf)
        self.draw()
        return self

    def deselect(self):
        if super().deselect():
            return
        self._text_patch.set_color(self.text_passive_color)
        self._text_patch.set_zorder(-1)
        self.draw()
        return self

    def __iadd__(self, other: str):
        self.text = self.text[:self.pointer] + other + self.text[self.pointer:]
        self.pointer += len(other)
        self.draw()
        return self

    def undo(self):
        self.text = self.text[:self.pointer - 1] + self.text[self.pointer:]
        self.pointer -= 1
        self.draw()

    def __str__(self):
        return f"Text('{self.text}') for {self.parent}"

    def change_fontsize(self, up=True):
        self._text_patch.set_fontsize(self._text_patch.get_fontsize() + (2 if up else -2))
        Text.fontsize = self._text_patch.get_fontsize()

    def draw(self):
        ps = self.pointer_symbol if self.is_selected else ''
        text = self.text[:self.pointer] + ps + self.text[self.pointer:]
        self._text_patch.set_text(text)

    def remove(self):
        logging.info(f'removing {self}')
        super().remove()
        if self in self.target.labels:
            self.target.labels.remove(self)
        self._text_patch.remove()

    def copy(self, target=None):
        t = Text(self.text, self.target if target is None else target, self.rel_loc,
                        self.radius, self.text_passive_color, self.active_color,
                        fontsize=self._text_patch.get_fontsize())
        t.pointer = self.pointer
        return t


class Vertex(SelectObject):
    count = 0
    selected_vertex = None
    vertices = []

    def __init__(self, x=0, y=0, r=1, th0=0, n_prongs=3, fixed=False, ax=None, fig=None,
                 arrows=None, center_mark=True, shapes=None, **patch_kwargs):
        super().__init__()
        if ax is None:
            ax = plt.gca()
        def_patch_kwargs = dict(color='black', facecolor=None)
        self.patch_kwargs = def_patch_kwargs | patch_kwargs
        self.index = self.count

        self.loc = (x, y)
        self.center = Target(self, radius=0.05 if center_mark else 0.15,
                             passive_color='black' if center_mark else None, ax=ax)
        self.center.index = 0
        self.targets = [self.center]
        self.r = r
        self.th0 = th0
        for k in range(n_prongs):
            th = 2 * np.pi * k / n_prongs
            self.targets.append(VertexTarget(self, (r * np.cos(th + th0), r * np.sin(th + th0)), ax=ax,
                                             arrow_out=None if arrows is None else arrows[k],
                                             shape=None if shapes is None else shapes[k]))

        self.select_time = None
        self.select_target = None
        self.fixed = fixed

        Vertex.vertices.append(self)
        Vertex.count += 1

    def __str__(self):
        return f'vertex {self.index}'

    def move(self, loc, force=False):
        if (self.moving or force) and not self.fixed:
            x, y = loc
            if self.select_target is None:
                return
            if self.select_target != self.targets[0]:
                return
            target_point: Target = self.select_target
            x -= target_point.loc[0]
            y -= target_point.loc[1]

            self.loc = (self.loc[0] + x, self.loc[1] + y)
            for i, t in enumerate(self.targets):
                t.move(redraw=False, force=True)

    def hit(self, click_event):
        for t in self.targets:
            hit = t.hit(click_event)
            if hit is not None:
                self.select_target = hit
                return True
        return False

    def select(self):
        if self.select_target is None:
            self.select_target = self.targets[0]
        logging.debug(f'calling {self.select_target}.select()')
        sel_obj = self.select_target.select()
        logging.debug(f'{self.select_target}.select() returned {sel_obj}')
        if sel_obj not in self.targets:
            return sel_obj
        if self.select_target == self.center:
            for t in self.targets[1:]:
                t.leg.select()
            self.select_target.select()
            sel_obj = self
            super().select()
        return sel_obj

    def deselect(self, keep_target=False):
        if super().deselect() is None:
            return
        for t in self.targets:
            t.deselect()
        if not keep_target:
            self.select_target = None
        return self

    def get_decoratable_target(self, t_idx=None):
        if len(self.targets) == 2:
            return self.targets[1]
        if t_idx is None:
            if self.select_target is None:
                return
            t = self.select_target
        else:
            t = self.targets[t_idx]
        if t != self.center:
            return t

    def make_arrow(self, t_idx=None, out=None):
        t = self.get_decoratable_target(t_idx)
        if t is not None:
            t.arrow_out = out

    def make_shape(self, t_idx=None, shape=None):
        t = self.get_decoratable_target(t_idx)
        if t is not None:
            t.shape = shape

    def connect(self, other):
        if other.select_target not in other.targets:
            return
        t_self = self.targets[1] if len(self.targets) == 2 else self.select_target
        t_other = other.targets[1] if len(other.targets) == 2 else other.select_target

        if t_self == self.center or t_other == other.center:
            return
        if t_self.vertex == t_other.vertex:
            return
        if t_self.leg.shape != t_self.leg.shape:
            return
        if t_self.leg.arrow_out is None or t_other.leg.arrow_out is None:
            if not (t_self.leg.arrow_out is None and t_other.leg.arrow_out is None):
                return
        elif not (t_self.leg.arrow_out ^ t_other.leg.arrow_out):
            return
        t_self.connect = t_other

    def remove(self):
        logging.info(f'remove {self}')
        self.deselect()
        for t in self.targets:
            t.remove()
        Vertex.vertices.remove(self)


def on_key_press(event):
    global selected_object, saved, copied_text
    k = event.key
    if isinstance(selected_object, Text):
        logging.debug(f'{selected_object} is selected')
        if k == 'escape':
            selected_object.deselect()
            selected_object.remove()
            selected_object = None
        elif k == 'enter':
            selected_object.deselect()
            selected_object = None
        elif k == 'backspace':
            selected_object.undo()
        elif k in ['left', 'right']:
            selected_object.pointer += (-1 if k == 'left' else 1)
        elif k in ['up', 'down']:
            selected_object.change_fontsize(up=k == 'up')
        elif k == 'control':
            copied_text = selected_object
        elif len(k) == 1:
            selected_object += k
        return fig.canvas.draw()

    if k == 't' and selected_object is not None:
        logging.debug(f'pressed t and {selected_object} is selected')
        if isinstance(selected_object, Vertex):
            target = selected_object.center
        elif isinstance(selected_object, Target):
            target = selected_object
        else:
            return
        selected_object.deselect()
        selected_object = Text('', target)
        selected_object.select()
        selected_object.move((event.xdata, event.ydata), force=True)

    if k == 'control' and isinstance(copied_text, Text):
        logging.debug(f'pressed {k} and {copied_text} is copied')
        if isinstance(selected_object, (Target, Vertex)):
            logging.debug(f'pressed {k} and {selected_object} is selected')
            copied_text.copy(selected_object if isinstance(selected_object, Target) else selected_object.center)
        selected_object.deselect()
        selected_object = None
        return fig.canvas.draw()

    if k in 'sfqQ' or k.isnumeric():
        if selected_object is not None:
            logging.debug(f'pressed {k} and {selected_object} is selected')
            selected_object.deselect()
            selected_object = None
        kwargs = dict(n_prongs=int(k) if k.isnumeric() else (3 if k.lower() == 'q' else 1),
                      th0=np.pi if k == 'f' else 0,
                      arrows=[None, True, False] if k.lower() == 'q' else None,
                      shapes=['photon' if k == 'q' else 'gluon', None, None] if k.lower() == 'q' else None,
                      ax=event.inaxes, center_mark=k in 'sf')
        selected_object = Vertex(event.xdata, event.ydata, **kwargs)
        logging.debug(f'pressed {k} and adding {selected_object}')
        selected_object.select()

    if k == 'd' and selected_object is not None:
        if isinstance(selected_object, Vertex):
            selected_object.remove()
        elif isinstance(selected_object, LegTarget):
            selected_object.disconnect()
        selected_object.deselect()
        selected_object = None
    if isinstance(selected_object, (Vertex, VertexTarget, LegTarget)):
        t = selected_object
        if isinstance(selected_object, Vertex):
            t = selected_object.get_decoratable_target()
        if t is not None:
            if k in 'io':
                t.arrow_out = k == 'o'
                t.shape = None
            if k == '-':
                t.arrow_out = None
                t.shape = None
            if k in 'gph':
                t.shape = k

    if k in 'ex':
        for v in Vertex.vertices[::-1]:
            v.deselect() if k == 'e' else v.remove()
        selected_object = None
    if k == 'm':
        fig.savefig(os.path.join(save_to, f'feynman_diagram_{saved}.pdf'),
                    bbox_inches='tight')
        saved += 1
    fig.canvas.draw()


def on_pick(event):
    global selected_object
    logging.info('picking')
    if selected_object is not None:
        if time.time() - selected_object.select_time < 1e-1:
            logging.info('ignore second pick')
            return
        if selected_object.hit(event):
            logging.info(f'picked {selected_object} again')
            if isinstance(selected_object, Vertex):
                selected_object.deselect(keep_target=True)
            else:
                selected_object.deselect()
            selected_object = selected_object.select()
            logging.info(f'selected object is {selected_object}')
            return fig.canvas.draw()

    for v in Vertex.vertices:
        v: Vertex
        if v.hit(event):
            logging.info(f'hit {v}')
            if selected_object is None:
                selected_object = v.select()
            elif isinstance(selected_object, (VertexTarget, Vertex)):
                vert = selected_object if isinstance(selected_object, Vertex) else selected_object.vertex
                vert.connect(v)
                selected_object.deselect()
                v.deselect()
                selected_object = None
            logging.info(f'selected object is {selected_object}')
            return fig.canvas.draw()
    logging.info(f'selected object is {selected_object}')


permanent_selection_time = 0.15


def on_motion(event):
    if isinstance(selected_object, (Vertex, Target)) and event.button == 1:
        if time.time() - selected_object.select_time < permanent_selection_time:
            return
        selected_object.move((event.xdata, event.ydata))
        fig.canvas.draw()


# Function to reset the selected central point when releasing the mouse button
def on_release(event):
    global selected_object
    logging.info('releasing')

    if selected_object is not None:
        if time.time() - selected_object.select_time > permanent_selection_time:
            logging.info(f'releasing {selected_object}')
            selected_object.deselect()
            selected_object = None
            fig.canvas.draw()
        else:
            logging.info(f'freezing {selected_object}')
            selected_object.freeze()
    logging.info(f'selected object is {selected_object}')


# Create a Matplotlib figure and axis
grid = (1, 1)
fig, axes = plt.subplots(*grid, figsize=(8 * grid[1], 6 * grid[0]), squeeze=False)

fig.suptitle("Draw Feynman diagrams")

for ax in axes.flatten():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis(False)

for k in plt.rcParams:
    if 'keymap.' in k:
        plt.rcParams[k] = []

# Connect the events
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
