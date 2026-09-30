#!/usr/bin/env python3
"""Georeference the supplied aerial screenshot to MORAI local map XY."""

import math
import os
import shutil
import tempfile
import hashlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates


class AerialMap:
    def __init__(self, image_path, calibration_path, checkpoints, resolution=0.5):
        self.image_path = Path(image_path).expanduser()
        self.calibration_path = Path(calibration_path).expanduser()
        self.image = np.asarray(Image.open(self.image_path).convert('L'), dtype=np.uint8)
        self.resolution = float(resolution)
        if not 0.25 <= self.resolution <= 2.0:
            raise ValueError('map resolution must be 0.25..2.0 m/pixel')
        with self.calibration_path.open() as stream:
            data = yaml.safe_load(stream)
        if data.get('image_size') != [self.image.shape[1], self.image.shape[0]]:
            raise ValueError('calibration image dimensions do not match aerial image')
        self.anchors = data['anchors']
        self.point_anchors = data.get('point_anchors', {})
        self.warp = data.get('warp', 'local')
        self.corners_world = data.get('corners_world')
        if self.warp not in ('affine', 'local'):
            raise ValueError('unsupported aerial warp mode')
        self.checkpoints = checkpoints
        self._fit()

    def _fit(self):
        if self.corners_world is not None:
            self.set_corners(self.corners_world)
            return
        world, pixels = [], []
        for name, pixel in self.anchors.items():
            if name not in self.checkpoints or len(pixel) != 2:
                raise ValueError('invalid image anchor: ' + name)
            world.append(self.checkpoints[name]['position'][:2])
            pixels.append(pixel)
        for name, anchor in self.point_anchors.items():
            if not name.startswith('point_') or set(anchor) != {'world', 'pixel'}:
                raise ValueError('invalid waypoint image anchor: ' + name)
            world.append(anchor['world'])
            pixels.append(anchor['pixel'])
        if len(world) < 3:
            raise ValueError('at least three image anchors are needed')
        self.world = np.asarray(world, dtype=float)
        self.pixels = np.asarray(pixels, dtype=float)
        if not np.isfinite(self.world).all() or not np.isfinite(self.pixels).all():
            raise ValueError('image calibration has non-finite values')
        self.affine = np.linalg.lstsq(
            np.column_stack((self.world, np.ones(len(self.world)))), self.pixels, rcond=None)[0]
        residual = self.pixels - np.column_stack((self.world, np.ones(len(self.world)))) @ self.affine
        # Local corrections absorb small differences between the supplied aerial
        # screenshot and the CSV's MORAI map revision.
        self.correction = (None if self.warp == 'affine' else
                           RBFInterpolator(self.world, residual,
                                           kernel='thin_plate_spline', smoothing=5.0))

    def set_corners(self, corners):
        corners = np.asarray(corners, dtype=float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError('image corners must be four finite XY positions')
        origin = corners[0]
        u = corners[1] - origin
        v = corners[3] - origin
        if abs(np.linalg.det(np.vstack((u, v)))) < 1.0:
            raise ValueError('image resize would collapse the photo')
        corners[2] = origin + u + v
        self.corners_world = corners.tolist()
        pixel_to_world = np.vstack((u / self.image.shape[1],
                                    v / self.image.shape[0]))
        linear = np.linalg.inv(pixel_to_world)
        self.affine = np.vstack((linear, -origin @ linear))
        self.correction = None

    def image_corners_world(self):
        if self.corners_world is not None:
            return np.asarray(self.corners_world, dtype=float)
        pixels = np.asarray([[0, 0], [self.image.shape[1], 0],
                             [self.image.shape[1], self.image.shape[0]],
                             [0, self.image.shape[0]]], dtype=float)
        return (pixels - self.affine[2]) @ np.linalg.inv(self.affine[:2])

    def resize_handle(self, name, dragged_world_xy, base_corners=None):
        """Resize like a paint editor: corners keep aspect ratio, edges stretch one axis."""
        corners = np.asarray(base_corners if base_corners is not None
                             else self.image_corners_world(), dtype=float)
        origin = corners[0]
        u = corners[1] - origin
        v = corners[3] - origin
        dragged = np.asarray(dragged_world_xy, dtype=float)
        if not np.isfinite(dragged).all():
            raise ValueError('invalid resize position')
        if name in ('tl', 'tr', 'br', 'bl'):
            indices = {'tl': (0, 2), 'tr': (1, 3),
                       'br': (2, 0), 'bl': (3, 1)}
            current_index, opposite_index = indices[name]
            opposite = corners[opposite_index]
            direction = corners[current_index] - opposite
            factor = float(np.dot(dragged - opposite, direction)
                           / np.dot(direction, direction))
            if not 0.1 <= factor <= 5.0:
                raise ValueError('image corner scale must stay between 10% and 500%')
            new_u, new_v = u * factor, v * factor
            if name == 'tl':
                new_origin = corners[2] - new_u - new_v
            elif name == 'tr':
                new_origin = corners[3] - new_v
            elif name == 'br':
                new_origin = origin
            else:
                new_origin = corners[1] - new_u
        elif name in ('top', 'right', 'bottom', 'left'):
            new_origin, new_u, new_v = origin.copy(), u.copy(), v.copy()
            if name == 'right':
                factor = float(np.dot(dragged - (origin + v / 2), u) / np.dot(u, u))
                new_u = u * factor
            elif name == 'left':
                factor = float(np.dot((origin + u + v / 2) - dragged, u)
                               / np.dot(u, u))
                new_u = u * factor
                new_origin = corners[1] - new_u
            elif name == 'bottom':
                factor = float(np.dot(dragged - (origin + u / 2), v) / np.dot(v, v))
                new_v = v * factor
            else:
                factor = float(np.dot((origin + u / 2 + v) - dragged, v)
                               / np.dot(v, v))
                new_v = v * factor
                new_origin = corners[3] - new_v
            if not 0.1 <= factor <= 5.0:
                raise ValueError('image edge scale must stay between 10% and 500%')
        else:
            raise ValueError('unknown image resize handle')
        new_corners = np.asarray([new_origin, new_origin + new_u,
                                  new_origin + new_u + new_v,
                                  new_origin + new_v])
        self.set_corners(new_corners)
        return self.corners_world

    def world_to_pixel(self, xy):
        world = np.asarray(xy, dtype=float).reshape(-1, 2)
        affine = np.column_stack((world, np.ones(len(world)))) @ self.affine
        return affine if self.correction is None else affine + self.correction(world)

    def update_checkpoint_anchor(self, name, clicked_world_xy):
        if name not in self.anchors:
            raise ValueError('unknown calibration checkpoint')
        pixel = self.world_to_pixel([clicked_world_xy])[0]
        if np.linalg.norm(pixel - self.anchors[name]) > 120:
            raise ValueError('선택한 사진 위치가 기존 체크포인트에서 120픽셀 이상 떨어져 있습니다. 다시 클릭하세요.')
        self.anchors[name] = [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]
        self._fit()
        return self.anchors[name]

    def update_waypoint_anchor(self, point_id, world_xy, clicked_world_xy):
        pixel = self.world_to_pixel([clicked_world_xy])[0]
        name = 'point_{}'.format(int(point_id))
        self.point_anchors[name] = {
            'world': [float(world_xy[0]), float(world_xy[1])],
            'pixel': [round(float(pixel[0]), 3), round(float(pixel[1]), 3)],
        }
        self._fit()
        return self.point_anchors[name]['pixel']

    def save_calibration(self):
        target = self.calibration_path
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp',
                                            dir=str(target.parent))
        try:
            with os.fdopen(descriptor, 'w') as stream:
                yaml.safe_dump({'image_size': [self.image.shape[1], self.image.shape[0]],
                                'warp': self.warp,
                                'corners_world': self.corners_world,
                                'anchors': self.anchors,
                                'point_anchors': self.point_anchors}, stream, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def color_mesh_resource(self, directory):
        """Make an RViz mesh whose texture follows the same four map corners."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        corners = self.image_corners_world()
        digest = hashlib.sha256(corners.tobytes() + str(self.image_path).encode()).hexdigest()[:16]
        mesh_path = directory / ('photo_' + digest + '.dae')
        texture_path = directory / 'photo.png'
        if not texture_path.exists():
            shutil.copyfile(self.image_path, texture_path)
        if not mesh_path.exists():
            positions = ' '.join('{:.9f} {:.9f} -0.1'.format(x, y) for x, y in corners)
            # Collada V runs upward; PNG rows run downward. Reverse triangle
            # winding because the image's world corner order faces down.
            document = '''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><contributor><authoring_tool>map_tunner</authoring_tool></contributor>
    <unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_images><image id="photo-image"><init_from>photo.png</init_from></image></library_images>
  <library_effects><effect id="photo-effect"><profile_COMMON>
    <newparam sid="surface"><surface type="2D"><init_from>photo-image</init_from></surface></newparam>
    <newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam>
    <technique sid="common"><lambert><diffuse><texture texture="sampler" texcoord="UVSET"/>
    </diffuse></lambert></technique></profile_COMMON></effect></library_effects>
  <library_materials><material id="photo-material"><instance_effect url="#photo-effect"/>
  </material></library_materials>
  <library_geometries><geometry id="photo-geometry"><mesh>
    <source id="positions"><float_array id="positions-array" count="12">{positions}</float_array>
      <technique_common><accessor source="#positions-array" count="4" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/>
        <param name="Z" type="float"/></accessor></technique_common></source>
    <source id="normals"><float_array id="normals-array" count="3">0 0 1</float_array>
      <technique_common><accessor source="#normals-array" count="1" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/>
        <param name="Z" type="float"/></accessor></technique_common></source>
    <source id="uv"><float_array id="uv-array" count="8">0 1 1 1 1 0 0 0</float_array>
      <technique_common><accessor source="#uv-array" count="4" stride="2">
        <param name="S" type="float"/><param name="T" type="float"/>
        </accessor></technique_common></source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="2" material="photo-material">
      <input semantic="VERTEX" source="#vertices" offset="0"/>
      <input semantic="NORMAL" source="#normals" offset="1"/>
      <input semantic="TEXCOORD" source="#uv" offset="2" set="0"/>
      <p>0 0 0 2 0 2 1 0 1 0 0 0 3 0 3 2 0 2</p>
    </triangles></mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene">
    <node id="photo-node"><instance_geometry url="#photo-geometry">
      <bind_material><technique_common><instance_material symbol="photo-material"
        target="#photo-material"><bind_vertex_input semantic="UVSET"
        input_semantic="TEXCOORD" input_set="0"/></instance_material>
      </technique_common></bind_material></instance_geometry></node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
'''.format(positions=positions)
            mesh_path.write_text(document)
        return mesh_path.as_uri()

    def raster(self, waypoint_points):
        x_values = [point[0] for point in waypoint_points]
        y_values = [point[1] for point in waypoint_points]
        # The route does not visit the bottom and far right of the photograph.
        # Include the image's corners in the world bounds so RViz shows the full
        # source photo instead of cropping it to the route bounding box.
        image_corners = np.asarray([[0, 0], [self.image.shape[1], 0],
                                    [0, self.image.shape[0]],
                                    [self.image.shape[1], self.image.shape[0]]],
                                   dtype=float)
        corner_world = (image_corners - self.affine[2]) @ np.linalg.inv(self.affine[:2])
        margin = 65.0
        origin_x = math.floor(min(min(x_values), corner_world[:, 0].min()) - margin)
        origin_y = math.floor(min(min(y_values), corner_world[:, 1].min()) - margin)
        max_x = max(max(x_values), corner_world[:, 0].max()) + margin
        max_y = max(max(y_values), corner_world[:, 1].max()) + margin
        width = int(math.ceil((max_x - origin_x) / self.resolution))
        height = int(math.ceil((max_y - origin_y) / self.resolution))
        if width * height > 5_000_000:
            raise ValueError('map raster exceeds five million cells')
        output = np.empty((height, width), dtype=np.uint8)
        columns = np.arange(width, dtype=float)
        # OccupancyGrid rows grow with world Y, whereas image rows grow downward.
        for first in range(0, height, 64):
            last = min(height, first + 64)
            rows, cols = np.meshgrid(np.arange(first, last, dtype=float), columns,
                                     indexing='ij')
            world = np.column_stack((origin_x + (cols.ravel() + 0.5) * self.resolution,
                                     origin_y + (rows.ravel() + 0.5) * self.resolution))
            pixels = self.world_to_pixel(world)
            gray = map_coordinates(self.image, [pixels[:, 1], pixels[:, 0]],
                                   order=1, mode='constant', cval=255)
            output[first:last] = np.clip(np.rint(100 - gray.reshape(last - first, width)
                                                     .astype(float) * 100 / 255),
                                         0, 100).astype(np.uint8)
            inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < self.image.shape[1])
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.image.shape[0]))
            output[first:last].reshape(-1)[~inside] = 100  # black outside the photo
        return origin_x, origin_y, output
