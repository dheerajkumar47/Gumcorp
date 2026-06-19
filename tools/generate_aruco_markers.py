import cv2
import pandas as pd
import os

def generate_markers(csv_path="data/employees.csv", output_dir="data/aruco_markers"):
    # Load employee data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    print(f"Generating {len(df)} markers...")
    
    for index, row in df.iterrows():
        display_marker_id = int(row["marker_id"])
        opencv_marker_id = int(row.get("opencv_marker_id", display_marker_id))
        employee_name = row['name'].replace(" ", "_")
        
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, opencv_marker_id, 200)
        marker_img = cv2.copyMakeBorder(
            marker_img,
            40,
            40,
            40,
            40,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        
        # Save marker
        file_name = f"marker_{display_marker_id:02d}_{employee_name}.png"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, marker_img)
        
    print(f"Success! Markers saved in {output_dir}")

if __name__ == "__main__":
    generate_markers()
