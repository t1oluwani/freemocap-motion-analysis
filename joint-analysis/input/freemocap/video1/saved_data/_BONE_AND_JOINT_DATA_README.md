
# Bone data

This document describes the data that is saved for each bone in the rig.

All of this data is derived from a Blender Armature object based on a slightly tweaked version of the Rigify rig.

For multi-part data (e.g. XYZ location), each component of the data is saved in its own column: e.g. {bone_name}_x, {bone_name}_y, {bone_name}_z 

To access the bone data in Blender, use the following command (e.g. to get the Thigh.L bone) in the Blender python console (e.g. in the Scripting Tab): `bone = bpy.context.object["righ"].pose.bone["thigh.L"]`

Theses are the properties of the bone object that are saved:

# Bone Properties

## `bone.head` : location data - X, Y, Z (tuple of floats)
The location of the head of the bone in world space (e.g. (0.0, 0.0, 0.0))
https://docs.blender.org/api/current/bpy.types.PoseBone.html#bpy.types.PoseBone.head

## `bone.tail` : location data - X, Y, Z (tuple of floats)
The location of the tail of the bone in world space (e.g. (0.0, 0.0, 0.0))
https://docs.blender.org/api/current/bpy.types.PoseBone.html#bpy.types.PoseBone.tail


## `bone.rotation_quaternion['_x', '_y', '_z', '_w']` : quaternion - X, Y, Z, W (tuple of floats)
The rotation of the bone in quaternion space (e.g. (0.0, 0.0, 0.0, 1.0))
https://docs.blender.org/api/current/bpy.types.PoseBone.html#bpy.types.PoseBone.rotation_quaternion
https://en.wikipedia.org/wiki/Quaternion

## `bone.rotation_euler['_x', '_y', '_z']` : euler rotation - X, Y, Z (tuple of floats)
The rotation of the bone in euler space (e.g. (0.0, 0.0, 0.0))
https://docs.blender.org/api/current/bpy.types.PoseBone.html#bpy.types.PoseBone.rotation_euler
https://en.wikipedia.org/wiki/Euler_angles

## `bone.rotation_euler.order` : str
The order of the euler rotation (e.g. 'XYZ')
https://docs.blender.org/api/current/bpy.types.PoseBone.html#bpy.types.PoseBone.rotation_euler
https://en.wikipedia.org/wiki/Euler_angles

## `bone.matrix` : 4x4 matrix (tuple of tuples of floats)
The transformation matrix of the bone in world space after constraints and drivers are applied, in the armature object space
https://docs.blender.org/api/current/bpy.types.PoseBone.html#bpy.types.PoseBone.matrix
https://en.wikipedia.org/wiki/Transformation_matrix

