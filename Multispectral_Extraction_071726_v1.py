import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# Ensure these match your folder names exactly
BAND_FOLDERS = {
    'Red': 'Red',
    'Green': 'Green',
    'NIR': 'NIR',
    'RedEdge': 'RedEdge' 
}

# PIXEL MASKING THRESHOLD
# Pixels with NDVI < 0.25 are treated as soil.
# Increase this (e.g., 0.3) if too much soil is being included.
# Decrease this (e.g., 0.15) if shadowed leaves are being excluded.
SOIL_THRESHOLD = 0.25 

def get_plot_id(filename):
    """
    Extracts plot number from end of filename.
    Example: 'Experiment_Green_105.tif' -> '105'
    """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    if parts:
        return parts[-1]
    return None

def load_image(path):
    """Loads image as float32 to preserve data integrity."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.astype(np.float32)

def safe_div(numerator, denominator):
    """Safe division (returns 0 instead of infinity for div-by-zero)."""
    denom_safe = denominator.copy()
    denom_safe[denom_safe == 0] = 1e-5 
    return numerator / denom_safe

def main():
    # 1. Initialize GUI
    root = tk.Tk()
    root.withdraw()

    # 2. Select Directory
    print("Waiting for user selection...")
    base_dir = filedialog.askdirectory(title="Select Root Directory containing Band Folders")
    if not base_dir:
        print("No directory selected.")
        return

    # 3. Ask User: Save Masks?
    save_masks = messagebox.askyesno("Verify Masks", "Do you want to save the crop canopy masks (black/white images) to verify soil separation?")
    
    mask_dir = os.path.join(base_dir, "_Masks_QC")
    if save_masks:
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        print(f"Mask images will be saved to: {mask_dir}")

    # 4. Map Files (Robust Matching)
    print(f"Scanning directory: {base_dir}")
    plots = {}
    
    for band_name, folder_name in BAND_FOLDERS.items():
        full_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(full_path):
            messagebox.showerror("Error", f"Missing folder: {folder_name}")
            return
        
        files = [f for f in os.listdir(full_path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
        for f in files:
            pid = get_plot_id(f)
            if pid:
                if pid not in plots: plots[pid] = {}
                plots[pid][band_name] = os.path.join(full_path, f)

    # 5. Process Data
    results = []
    total_plots = len(plots)
    print(f"Found {total_plots} plots. Processing...")
    
    count = 0
    for pid, bands_paths in plots.items():
        count += 1
        if len(bands_paths) < 4:
            continue

        try:
            # Load Bands
            R = load_image(bands_paths['Red'])
            G = load_image(bands_paths['Green'])
            NIR = load_image(bands_paths['NIR'])
            RE = load_image(bands_paths['RedEdge'])

            if not (R.shape == G.shape == NIR.shape == RE.shape):
                print(f"Skipping Plot {pid}: Dimension mismatch.")
                continue

            # --- MASK GENERATION ---
            # NDVI for Segmentation
            denom = NIR + R
            denom[denom == 0] = 1e-5
            ndvi_seg = (NIR - R) / denom
            
            # Create Boolean Mask (True=Plant, False=Soil)
            plant_mask = ndvi_seg > SOIL_THRESHOLD
            
            # --- SAVE MASK IMAGE (OPTIONAL) ---
            if save_masks:
                # Convert boolean (0/1) to image (0/255)
                # uint8 is standard for images
                mask_img = (plant_mask * 255).astype(np.uint8)
                mask_filename = f"Mask_Plot_{pid}.jpg"
                cv2.imwrite(os.path.join(mask_dir, mask_filename), mask_img)

            # Check if empty
            if np.sum(plant_mask) == 0:
                results.append({'Plot_ID': pid, 'Notes': 'No vegetation detected'})
                continue

            # --- DATA EXTRACTION ---
            # Flatten arrays to include ONLY plant pixels
            r_vec = R[plant_mask]
            g_vec = G[plant_mask]
            nir_vec = NIR[plant_mask]
            re_vec = RE[plant_mask]

            # --- INDEX CALCULATION ---
            # 1. NDVI
            ndvi = safe_div(nir_vec - r_vec, nir_vec + r_vec)
            # 2. NDRE
            ndre = safe_div(nir_vec - re_vec, nir_vec + re_vec)
            # 3. GNDVI
            gndvi = safe_div(nir_vec - g_vec, nir_vec + g_vec)
            
            # 4. SAVI (L=0.5)
            L = 0.5
            savi = safe_div(nir_vec - r_vec, nir_vec + r_vec + L) * (1 + L)
            
            # 5. OSAVI
            osavi = safe_div(nir_vec - r_vec, nir_vec + r_vec + 0.16)
            
            # 6. CIG (Green Chlorophyll)
            cig = safe_div(nir_vec, g_vec) - 1
            
            # 7. CIRE (Red Edge Chlorophyll)
            cire = safe_div(nir_vec, re_vec) - 1
            
            # 8. SR (Simple Ratio)
            sr = safe_div(nir_vec, r_vec)
            
            # 9. MSR
            msr = safe_div(sr - 1, np.sqrt(sr) + 1)
            
            # 10. EVI2 (2-band Enhanced Vegetation Index)
            evi2 = 2.5 * safe_div(nir_vec - r_vec, nir_vec + 2.4 * r_vec + 1)

            # Record Data
            results.append({
                'Plot_ID': pid,
                # Raw Means (Vegetation Only)
                'Raw_Red': r_vec.mean(),
                'Raw_Green': g_vec.mean(),
                'Raw_NIR': nir_vec.mean(),
                'Raw_RedEdge': re_vec.mean(),
                # Indices
                'NDVI': ndvi.mean(),
                'NDRE': ndre.mean(),
                'GNDVI': gndvi.mean(),
                'SAVI': savi.mean(),
                'OSAVI': osavi.mean(),
                'CIG': cig.mean(),
                'CIRE': cire.mean(),
                'SR': sr.mean(),
                'MSR': msr.mean(),
                'EVI2': evi2.mean(),
                'Pixels_Count': len(r_vec)
            })

            # Print progress every 10%
            if count % max(1, int(total_plots/10)) == 0:
                print(f"Processed {count}/{total_plots} plots...")

        except Exception as e:
            print(f"Error on Plot {pid}: {e}")

    # 6. Save CSV
    if results:
        df = pd.DataFrame(results)
        
        # Sort numerically by ID
        try:
            df['sort_helper'] = pd.to_numeric(df['Plot_ID'])
            df = df.sort_values('sort_helper').drop('sort_helper', axis=1)
        except:
            df = df.sort_values('Plot_ID')

        output_csv = os.path.join(base_dir, 'Vegetation_Indices_Full_Report.csv')
        df.to_csv(output_csv, index=False)
        
        msg = f"Processing Complete!\n\nProcessed: {len(results)} Plots\nSaved CSV: {output_csv}"
        if save_masks:
            msg += f"\nSaved Masks: {mask_dir}"
            
        messagebox.showinfo("Success", msg)
        print("Done.")
    else:
        messagebox.showwarning("Warning", "No plots were processed.")

if __name__ == "__main__":
    main()