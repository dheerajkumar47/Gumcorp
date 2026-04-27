import cv2
import numpy as np
import fitz
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from pathlib import Path

def generate_fov_cone(cx, cy, angle_deg, fov_angle=90, radius=300):
    points = [(cx, cy)]
    start_angle = np.radians(angle_deg - fov_angle / 2)
    end_angle = np.radians(angle_deg + fov_angle / 2)
    steps = 20
    for i in range(steps + 1):
        theta = start_angle + (end_angle - start_angle) * i / steps
        px = cx + radius * np.cos(theta)
        py = cy + radius * np.sin(theta)
        points.append((px, py))
    return Polygon(points)

def main():
    pdf_path = "docs/GC Cameras CCTV Updated April 2026 Rev 01 (2).pdf"
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # EXACT coordinates from the pink CAD vector analysis
    cameras = [
        {
            "id": "IP CAMERA 39",
            "cx": 1564,  # Exactly on left yellow wall
            "cy": 912,
            "angle": 45,  # Points Down-Right
            "bounds": Polygon([(1564, 918), (1755, 918), (1755, 1065), (1564, 1065)]),
            "radius": 400
        },
        {
            "id": "POWDER GUM",
            "cx": 1564,  # Exactly on left yellow wall
            "cy": 1220,
            "angle": -45,  # Points Up-Right
            "bounds": Polygon([(1564, 1065), (1755, 1065), (1755, 1245), (1564, 1245)]),
            "radius": 400
        },
        {
            "id": "GC PRODUCTION 2",
            "cx": 1782,  # Middle Camera as you requested!
            "cy": 1167,
            "angle": -135,  # Points Up-Left. (You can change this to 45 if it's supposed to face the room)
            "bounds": Polygon([(1755, 1065), (3000, 1065), (3000, 1500), (1755, 1500)]),
            "radius": 410 # EXACTLY 52 feet
        }
    ]
    
    fig, ax = plt.subplots(figsize=(25, 20)) # Make figure huge so full map is crisp
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    for cam in cameras:
        cx, cy = cam["cx"], cam["cy"]
        
        # Draw the Blue Square exactly on the camera icon
        rect = plt.Rectangle((cx-25, cy-25), 50, 50, linewidth=4, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        
        # Label the camera above the square
        ax.text(cx, cy - 35, cam["id"], color='blue', fontsize=18, fontweight='bold', ha='center')
        
        # Generate the FOV cone
        fov = generate_fov_cone(cx, cy, cam["angle"], fov_angle=90, radius=cam["radius"])
        
        # STRICTLY intersect with room walls so it does NOT spill out!
        if cam["bounds"]:
            fov = fov.intersection(cam["bounds"].buffer(1.0))
            
        # Fill the cone with green
        if fov.geom_type == 'Polygon':
            fx, fy = fov.exterior.xy
            ax.fill(fx, fy, alpha=0.5, color='green')
        elif fov.geom_type == 'MultiPolygon':
            for poly in fov.geoms:
                fx, fy = poly.exterior.xy
                ax.fill(fx, fy, alpha=0.5, color='green')
        elif fov.geom_type == 'GeometryCollection':
            for geom in fov.geoms:
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    if geom.geom_type == 'Polygon':
                        fx, fy = geom.exterior.xy
                        ax.fill(fx, fy, alpha=0.5, color='green')
                    else:
                        for poly in geom.geoms:
                            fx, fy = poly.exterior.xy
                            ax.fill(fx, fy, alpha=0.5, color='green')

    # DO NOT ZOOM! Show the whole factory map!
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0) # Inverted Y for image coordinates
    ax.set_title("Factory Coverage Map - targeted Fix", fontsize=25, pad=20)
    plt.axis('off')
    
    # Save to your local desktop folder so you can see it easily
    out_path = Path("outputs/factory_blind_spots_FINAL.jpg")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved perfect targeted map to {out_path}")

if __name__ == "__main__":
    main()
