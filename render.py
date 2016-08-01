import bpy
import numpy as np
import math


# def blender_render():


def generate_animation(data, frame_divisor, muliplier=50):
    data_length = data.shape[0]
    no_frames = data_length / frame_divisor
    scene = bpy.context.scene
    # Add circle for path animation
    bpy.ops.curve.primitive_bezier_circle_add(radius=0.1)

    # Add bezier to show the path
    curvedata = bpy.data.curves.new(name='pathcurve', type='CURVE')
    curvedata.dimensions = '3D'
    objectdata = bpy.data.objects.new('pathobj', curvedata)
    objectdata.location = (0, 0, 0)  # object origin
    bpy.context.scene.objects.link(objectdata)

    polyline = curvedata.splines.new('BEZIER')
    polyline.bezier_points.add(no_frames)

    # Attach circle to the curve
    curvedata.bevel_object = scene.objects['BezierCircle']

    drone = scene.objects['drone']
    scene.frame_end = no_frames

    cur_frame = 0
    for frame_no in range(0, data_length, frame_divisor):
        scene.frame_set(cur_frame)
        current_data = state[frame_no]

        location = tuple(
            x * muliplier for x in (current_data[0], current_data[1], current_data[2]))


        # print(current_data[6], current_data[7], current_data[8])
        drone.location = location
        drone.rotation_euler = (
            current_data[6], current_data[7], current_data[8])
        # print(drone.rotation_euler)
        drone.keyframe_insert(data_path="location")
        drone.keyframe_insert(data_path="rotation_euler")

        polyline.bezier_points[cur_frame].co = location
        polyline.bezier_points[cur_frame].handle_left = location
        polyline.bezier_points[cur_frame].handle_right = location

        curvedata.bevel_factor_end = cur_frame / no_frames
        curvedata.keyframe_insert(data_path='bevel_factor_end')

        cur_frame += 1


state = np.load('/home/lonewolf/Desktop/quaddata.npy')
# generate_animation(state, 50)
generate_animation(state, 6, 75)
