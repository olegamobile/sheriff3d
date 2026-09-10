import socket
import json
import os

DEFAULT_PORT = 9876

def send_command(cmd_type: str, params: dict = None, timeout: float = 15.0) -> dict:
    """Send JSON command to Blender MCP server on port 9876 and receive response."""
    if params is None:
        params = {}
    payload = json.dumps({"type": cmd_type, "params": params}).encode('utf-8')
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(('127.0.0.1', DEFAULT_PORT))
        s.sendall(payload)
        
        # Read JSON response
        buf = b''
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            buf += chunk
            try:
                data = json.loads(buf.decode('utf-8'))
                return data
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return {"status": "error", "message": "Connection closed before complete JSON"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        s.close()

def execute(code: str) -> str:
    """Execute Python code in active Blender instance and return output."""
    res = send_command("execute_code", {"code": code})
    if res.get("status") == "success":
        return res.get("result", {}).get("result", "")
    else:
        raise RuntimeError(f"Blender execution error: {res}")

def get_scene() -> dict:
    """Get list of objects and scene details from active Blender."""
    res = send_command("get_scene_info", {})
    if res.get("status") == "success":
        return res.get("result", {})
    return res

def set_object_color(obj_name: str, color_rgb: tuple):
    """Example helper to change material color of an object in real-time."""
    code = f"""
import bpy
obj = bpy.data.objects.get('{obj_name}')
if obj and obj.data.materials:
    mat = obj.data.materials[0]
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = ({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 1.0)
"""
    return execute(code)

if __name__ == "__main__":
    print("Testing connection to Blender MCP...")
    sc = get_scene()
    print(f"Scene Name: {sc.get('name')}")
    print(f"Object Count: {sc.get('object_count')}")
    for obj in sc.get('objects', []):
        print(f"  * {obj['name']} ({obj['type']})")
    
    # Test execution
    out = execute("import bpy; print('Connected successfully to Blender ' + bpy.app.version_string)")
    print("Execution output:", out.strip())
