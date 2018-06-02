# -*- coding: utf-8 -*-

from __future__ import division, print_function

import subprocess
import tempfile
import solid

class OpenSCAD:
    def __init__(self, executable="openscad", extra_cmdline=[], header=""):
        self.executable = executable
        self.extra_cmdline = extra_cmdline
        self.running_instance = None
        self.running_file = tempfile.NamedTemporaryFile(
            suffix=".scad", delete=True)
        self.header = header

    def _popen(self, cmdline=[]):
        I = subprocess.Popen([self.executable]+self.extra_cmdline+cmdline)
        return I

    def show(self, thing):
        if (self.running_instance is None or
            self.running_instance.poll() is not None):
            self.running_instance = self._popen([
                self.running_file.name,
                ])
        solid.scad_render_to_file(thing,self.running_file.name,
                                  file_header=self.header)

    def render_to(self, thing, stl_name, header=None):
        if header is None:
            header = self.header
        f = tempfile.NamedTemporaryFile(suffix=".scad", delete=True)
        solid.scad_render_to_file(thing,f.name,
                                  file_header=header)
        I = self._popen(["-o", stl_name, f.name])
        if I.wait():
            raise subprocess.CalledProcessError(
                "OpenSCAD failed to render thing from %s to %s"
                % (f.name, stl_name),
                cmd="openscad")

