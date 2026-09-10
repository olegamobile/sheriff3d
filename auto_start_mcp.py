import bpy
import addon_utils

# Ensure addon is enabled
addon_utils.enable('blender_mcp', default_set=True)

# Set auto start and port
if hasattr(bpy.context, 'scene') and bpy.context.scene:
    bpy.context.scene.blendermcp_port = 9876
    bpy.context.scene.blendermcp_auto_start_server = True

# Start server using timer so UI is initialized
def auto_start_mcp():
    try:
        bpy.ops.blendermcp.start_server()
        print("\n==========================================")
        print("  BLENDER MCP SERVER STARTED ON PORT 9876 ")
        print("==========================================\n")
    except Exception as e:
        print("Could not start MCP server automatically:", e)
    return None # don't repeat timer

bpy.app.timers.register(auto_start_mcp, first_interval=1.0)
