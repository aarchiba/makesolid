# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
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
        try:
            I = subprocess.run(["openscad","-o", stl_name, f.name],
                    capture_output=True,
                    check=True)
        except subprocess.CalledProcessError as e:
            sys.stdout.write(e.stdout)
            sys.stderr.write(e.stderr)
            raise
