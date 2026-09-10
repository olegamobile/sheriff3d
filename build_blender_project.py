import bpy
import os
import sys
import math

def setup_blender_sheriff():
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Base workspace directory
    work_dir = os.path.abspath(os.path.dirname(__file__))
    stl_dir = os.path.join(work_dir, "stl_output")

    # Create Collections
    scene_coll = bpy.context.scene.collection
    
    col_boat = bpy.data.collections.new("Jouet_Sheriff_1to45")
    scene_coll.children.link(col_boat)

    col_studio = bpy.data.collections.new("Studio_Environment")
    scene_coll.children.link(col_studio)

    # ---------------------------------------------------------
    # Helper: Create PBR Material
    # ---------------------------------------------------------
    def make_mat(name, color, roughness=0.3, metallic=0.0, transmission=0.0):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            # Blender 4+ / 5+ Principled BSDF socket names
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = metallic
            if "Transmission Weight" in bsdf.inputs:
                bsdf.inputs["Transmission Weight"].default_value = transmission
            elif "Transmission" in bsdf.inputs:
                bsdf.inputs["Transmission"].default_value = transmission
        return mat

    mat_deck = make_mat("Gelcoat_Deck_White", (0.92, 0.94, 0.96, 1.0), roughness=0.25)
    mat_keel = make_mat("CastIron_Keel", (0.18, 0.19, 0.20, 1.0), roughness=0.6, metallic=0.7)
    mat_rudder = make_mat("Teak_Tiller_Rudder", (0.65, 0.38, 0.16, 1.0), roughness=0.45)
    mat_cradle = make_mat("Display_Cradle_Navy", (0.08, 0.14, 0.24, 1.0), roughness=0.4)
    mat_rigging = make_mat("Anodized_Aluminium", (0.75, 0.77, 0.80, 1.0), roughness=0.25, metallic=0.9)

    # ---------------------------------------------------------
    # Import Models
    # ---------------------------------------------------------
    def import_stl(filename, collection, material, name=None):
        path = os.path.join(stl_dir, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
        
        # In Blender 4.2+ / 5.2+, wm.stl_import is standard
        try:
            bpy.ops.wm.stl_import(filepath=path)
        except Exception:
            bpy.ops.import_mesh.stl(filepath=path)
            
        obj = bpy.context.selected_objects[0]
        if name:
            obj.name = name
        else:
            obj.name = filename.replace(".stl", "")
            
        if material:
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)

        # Move to correct collection
        for c in obj.users_collection:
            c.objects.unlink(obj)
        collection.objects.link(obj)
        return obj

    # Import parts
    obj_deck = import_stl("Jouet_Sheriff_Split_Deck_1to45.stl", col_boat, mat_deck, "Deck_and_Coachroof")
    obj_keel = import_stl("Jouet_Sheriff_Split_Keel_1to45.stl", col_boat, mat_keel, "Underwater_Hull_and_Keel")
    obj_rudder = import_stl("Jouet_Sheriff_Rudder_1to45.stl", col_boat, mat_rudder, "Transom_Rudder_and_Tiller")
    obj_rigging = import_stl("Jouet_Sheriff_Rigging_1to45.stl", col_boat, mat_rigging, "Mast_and_Rigging")
    obj_cradle = import_stl("Jouet_Sheriff_Display_Cradle_1to45.stl", col_boat, mat_cradle, "Display_Cradle")

    # Set up smooth shading and auto-smooth
    for obj in [obj_deck, obj_keel, obj_rudder, obj_cradle]:
        if obj:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()
            # Enable auto smooth / sharp edge retention if available
            try:
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = math.radians(35.0)
            except Exception:
                pass

    # ---------------------------------------------------------
    # Studio Lighting & Camera
    # ---------------------------------------------------------
    # Camera
    cam_data = bpy.data.cameras.new(name="Main_Camera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("Main_Camera", cam_data)
    col_studio.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Position camera looking at 1:45 boat (center around X=65, Y=0, Z=10 mm)
    cam_obj.location = (200.0, -180.0, 110.0)
    cam_obj.rotation_euler = (math.radians(65.0), 0.0, math.radians(45.0))

    # Key Light
    key_data = bpy.data.lights.new(name="Key_Light", type='AREA')
    key_data.energy = 50.0
    key_data.size = 150.0
    key_obj = bpy.data.objects.new("Key_Light", key_data)
    key_obj.location = (120.0, -120.0, 160.0)
    col_studio.objects.link(key_obj)

    # Fill Light
    fill_data = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_data.energy = 25.0
    fill_data.size = 200.0
    fill_obj = bpy.data.objects.new("Fill_Light", fill_data)
    fill_obj.location = (-60.0, 100.0, 100.0)
    col_studio.objects.link(fill_obj)

    # Rim Light
    rim_data = bpy.data.lights.new(name="Rim_Light", type='SPOT')
    rim_data.energy = 40.0
    rim_obj = bpy.data.objects.new("Rim_Light", rim_data)
    rim_obj.location = (-100.0, -60.0, 80.0)
    col_studio.objects.link(rim_obj)

    # Render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items else 'BLENDER_EEVEE'

    # Save .blend file
    blend_path = os.path.join(work_dir, "Jouet_Sheriff_600.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"SUCCESS: Saved Blender project to: {blend_path}")

if __name__ == "__main__":
    setup_blender_sheriff()
