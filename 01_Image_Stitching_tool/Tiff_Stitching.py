"""
Tiff_Stitching.py
---------------------------------
Purpose:
- Provide stitching and OME-TIFF handling utilities exported from the notebook.

Notes:
- Functions are preserved in logic, with imports consolidated.
- No self-imports: this module does not import `Tiff_Stitching`.
"""

import os
import sys
import pathlib
import re
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xml.etree.ElementTree as ET
from pyometiff import OMETIFFReader
from pyometiff.omexml import OMEXML
from tifffile import TiffFile, imwrite

# Silence noisy warnings during metadata parsing
warnings.filterwarnings('ignore')

# Helper Functions
import traceback


def extract_ome_xml(text: str) -> str:
    """
    Extract a clean OME-XML string from a mixed ImageDescription/metadata string.
    Handles cases like "ImageJ=... ome_xml=<OME ...>" by isolating the <OME> ... </OME> block.
    Supports both uppercase and lowercase variants (OME/ome).
    """
    if not text:
        return None
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='ignore')
        except Exception:
            return None
    
    # Look for explicit ome_xml= prefix first (case-insensitive)
    m = re.search(r'ome_xml=(<[Oo][Mm][Ee][\s\S]*?</[Oo][Mm][Ee]>)', text, re.IGNORECASE)
    if m:
        return m.group(1)
    
    # Fallback: find first <OME or <ome and cut to closing </OME> or </ome> (case-insensitive)
    m = re.search(r'<[Oo][Mm][Ee][\s\S]*?</[Oo][Mm][Ee]>', text, re.IGNORECASE)
    if m:
        return m.group(0)
    
    # Final fallback: find opening tag without closing (case-insensitive)
    m = re.search(r'<[Oo][Mm][Ee][\s>]', text, re.IGNORECASE)
    if m:
        start = m.start()
        return text[start:]
    
    return None




def get_channel_info_from_tiff(tiff_path: str) -> pd.DataFrame:
    """
    Extract channel information from an OME-TIFF file using multiple methods.
    
    Args:
        tiff_path: Path to the OME-TIFF file
        
    Returns:
        DataFrame with channel IDs and names
    """
    print(f"Extracting channel info from: {os.path.basename(tiff_path)}")
    channel_info = []
    
    # Method 1: Using tifffile to read OME-XML
    print("\n[Method 1] Reading with tifffile - OME-XML metadata:")
    try:
        with TiffFile(tiff_path) as tif:
            raw_meta = None
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                raw_meta = tif.ome_metadata
            elif hasattr(tif, 'OME_metadata') and tif.OME_metadata:
                raw_meta = tif.OME_metadata
            elif tif.pages and tif.pages[0].description:
                desc = tif.pages[0].description
                if isinstance(desc, bytes):
                    desc = desc.decode('utf-8', errors='ignore')
                
                # Check for PerkinElmer format
                if '<PerkinElmer-QPI-ImageDescription>' in desc:
                    print("  Detected PerkinElmer format")
                    for i, page in enumerate(tif.pages):
                        page_desc = page.description
                        if isinstance(page_desc, bytes):
                            page_desc = page_desc.decode('utf-8', errors='ignore')
                        name_match = re.search(r'<Name>([^<]+)</Name>', page_desc)
                        if name_match:
                            ch_name = name_match.group(1)
                            channel_info.append({'ID': f'Channel:{i}', 'Name': ch_name})
                    raw_meta = None
                else:
                    raw_meta = desc
            
            if raw_meta and not channel_info:
                cleaned_xml = extract_ome_xml(raw_meta)
                if cleaned_xml:
                    root = ET.fromstring(cleaned_xml)
                    namespaces = [
                        {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'},
                        {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'},
                        {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2013-06'},
                        {}
                    ]
                    for ns in namespaces:
                        channels = root.findall('.//ome:Channel', ns) if ns else root.findall('.//Channel')
                        if channels:
                            for idx, channel in enumerate(channels):
                                ch_id = channel.get('ID', f'Channel:{idx}')
                                ch_name = channel.get('Name', f'Channel_{idx}')
                                channel_info.append({'ID': ch_id, 'Name': ch_name})
                            break
    except Exception as e:
        print(f"  Method 1 failed: {e}")
    
    # Method 2: Using pyometiff
    if not channel_info:
        print("\n[Method 2] Reading with pyometiff:")
        try:
            reader = OMETIFFReader(fpath=str(tiff_path))
            try:
                img_array, metadata, xml_metadata = reader.read()
                if metadata and 'Channels' in metadata:
                    for idx, ch in enumerate(metadata['Channels']):
                        ch_name = ch.get('Name', ch.get('ID', f'Channel_{idx}'))
                        channel_info.append({'ID': f'Channel:{idx}', 'Name': ch_name})
            except Exception:
                # Try sanitized XML with OMEXML
                with TiffFile(tiff_path) as tif:
                    raw_meta = tif.ome_metadata or (tif.pages and tif.pages[0].description)
                cleaned_xml = extract_ome_xml(raw_meta) if raw_meta else None
                if cleaned_xml:
                    ox = OMEXML(cleaned_xml)
                    images = ox.get_image_count()
                    for i in range(images):
                        pixels = ox.get_image_pixels(i)
                        size_c = int(pixels.SizeC)
                        for idx in range(size_c):
                            ch = ox.get_channel(i, idx)
                            ch_name = getattr(ch, 'Name', None) or f'Channel_{idx}'
                            channel_info.append({'ID': f'Channel:{idx}', 'Name': ch_name})
        except Exception as e:
            print(f"  Method 2 failed: {e}")
    
    # Return results
    if channel_info:
        df_channels = pd.DataFrame(channel_info).drop_duplicates(subset=['Name'], keep='first').reset_index(drop=True)
        print(f"\n✓ Successfully extracted {len(df_channels)} channels")
        return df_channels
    else:
        print("\n✗ Could not extract channel information")
        return pd.DataFrame(columns=['ID', 'Name'])


def read_ometiff(img_fpath: str) -> Tuple:
    """Read an OME-TIFF file and return the array, metadata, and XML metadata.
    
    Args:
        img_fpath: Path to the OME-TIFF file
        
    Returns:
        Tuple of (img_array, metadata, xml_metadata)
    """
    reader = OMETIFFReader(fpath=str(img_fpath))
    img_array, metadata, xml_metadata = reader.read()
    return img_array, metadata, xml_metadata


def read_tif_files(path: str, file_type: str = '.tif') -> pd.DataFrame:
    """Get all TIFF files in the specified directory.
    
    Args:
        path: Directory path to search for files
        file_type: File extension to filter (default: '.tif')
        
    Returns:
        DataFrame with FileName column
    """
    files = [f for f in os.listdir(path) if f.endswith(file_type)]
    df_files = pd.DataFrame(files, columns=['FileName'])
    return df_files


def get_coordinates_from_filename(df_files: pd.DataFrame) -> pd.DataFrame:
    """Extract X, Y coordinates from filenames with format [...[x,y]...].
    
    Args:
        df_files: DataFrame with FileName column
        
    Returns:
        DataFrame with added X and Y columns
    """
    coordinates_x = []
    coordinates_y = []
    
    for file_name in df_files['FileName']:
        try:
            # Find the part of the filename that contains the coordinates
            coord_part = file_name.split('[')[-1].split(']')[0]
            # Get the x and y coordinates from the coord_part
            x, y = map(int, coord_part.split(','))
            coordinates_x.append(x)
            coordinates_y.append(y)
        except Exception as e:
            print(f"Warning: Could not extract coordinates from {file_name}: {e}")
            coordinates_x.append(0)
            coordinates_y.append(0)
    
    df_files = df_files.copy()
    df_files['X'] = coordinates_x
    df_files['Y'] = coordinates_y
    return df_files


def get_position_from_coordinates(df_files: pd.DataFrame) -> pd.DataFrame:
    """Convert coordinates to grid positions (PosY, PosX).
    
    Args:
        df_files: DataFrame with X and Y columns
        
    Returns:
        DataFrame with added Position column as (PosY, PosX) tuple
    """
    # Get sorted unique values for X and Y
    unique_x = np.sort(df_files['X'].unique())
    unique_y = np.sort(df_files['Y'].unique())
    
    # Map each (X, Y) to (PosY, PosX) based on their order
    positions = [
        (np.where(unique_y == row['Y'])[0][0] + 1, np.where(unique_x == row['X'])[0][0] + 1)
        for _, row in df_files.iterrows()
    ]
    
    df_files = df_files.copy()
    df_files['Position'] = positions
    return df_files


def combine_tiff_arrays(df_files: pd.DataFrame, path: str) -> np.ndarray:
    """Combine multiple TIFF tiles into a single array based on their positions.
    
    Args:
        df_files: DataFrame with FileName and Position columns
        path: Directory containing the TIFF files
        
    Returns:
        Combined numpy array with shape (n_channels, height, width)
    """
    # Get the number of rows and columns
    R = df_files['Position'].apply(lambda x: x[0]).max()
    C = df_files['Position'].apply(lambda x: x[1]).max()
    
    print(f"Grid size: {R} rows x {C} columns")
    
    # Check if the number of files is valid
    if len(df_files) > R * C:
        raise ValueError(f"Number of files ({len(df_files)}) exceeds grid size ({R * C})")
    
    # Read the first file to get dimensions
    first_file_path = os.path.join(path, df_files.iloc[0]['FileName'])
    ome_array, _, _ = read_ometiff(first_file_path)
    n_channels, xlength, ylength = ome_array.shape
    print(f"Tile dimensions: {n_channels} channels x {xlength} x {ylength}")
    
    # Create an empty array for the combined image
    combined_array = np.zeros((n_channels, R * xlength, C * ylength), dtype=np.uint16)
    
    # Fill in each tile
    for idx, row in df_files.iterrows():
        posX, posY = row['Position']
        print(f"Processing: {row['FileName']} at position ({posX}, {posY})")
        
        file_path = os.path.join(path, row['FileName'])
        ome_array, _, _ = read_ometiff(file_path)
        
        # Calculate starting positions
        start_x = (posX - 1) * xlength
        start_y = (posY - 1) * ylength
        
        # Place the tile in the combined array
        for channel in range(n_channels):
            combined_array[channel, start_x:start_x + xlength, start_y:start_y + ylength] = ome_array[channel]
    
    print(f"Combined array shape: {combined_array.shape}")
    return combined_array

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def OmeTiff_align(path: str, output_suffix: str = "Combined", exclude_files: list = None) -> Tuple[str, np.ndarray, Optional[pd.DataFrame]]:
    """
    Align and combine multiple OME-TIFF tiles from a directory into a single OME-TIFF file.

    Args:
        path: Directory path containing the TIFF tiles
        output_suffix: Suffix for the output filename (default: "Combined")
        exclude_files: List of text patterns to exclude. Files containing any of these
                      patterns in their filename will be excluded (case-insensitive partial match).
                      Example: ['Combined', 'ome'] will exclude 'file_Combined.tif' and 'image.ome.tif'
                      (default: None)

    Returns:
        Tuple of (output_filepath, combined_array, df_channels)

    Example:
        output_path, combined_img, channels = OmeTiff_align(
            r"C:\\path\\to\\tiles",
            exclude_files=['Combined', 'ome']
        )
    """
    print("=" * 80)
    print(f"Processing directory: {path}")
    print("=" * 80)

    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    # Step 1: Read all TIFF files
    print("\n[Step 1/7] Reading TIFF files...")
    df_files = read_tif_files(path, file_type='.tif')

    if df_files.empty:
        raise ValueError(f"No .tif files found in {path}")

    print(f"Found {len(df_files)} TIFF files")

    # Exclude files if filename contains any text from exclude_files list
    if exclude_files:
        original_count = len(df_files)
        # Filter out files whose names contain any of the exclude patterns
        df_files = df_files[~df_files['FileName'].str.contains('|'.join(exclude_files), case=False, na=False)]
        excluded_count = original_count - len(df_files)
        print(f"Excluded {excluded_count} file(s) matching patterns: {exclude_files}")
        print(f"Remaining files: {len(df_files)}")

    if df_files.empty:
        raise ValueError("No files remaining after exclusion")

    # Step 2: Extract coordinates from filenames
    print("\n[Step 2/7] Extracting coordinates from filenames...")
    df_files = get_coordinates_from_filename(df_files)

    # Step 3: Convert coordinates to grid positions
    print("\n[Step 3/7] Converting coordinates to grid positions...")
    df_files = get_position_from_coordinates(df_files)
    print(df_files[['FileName', 'Position']].to_string())

    # Step 4: Get channel information
    print("\n[Step 4/7] Extracting channel information...")
    first_file_path = os.path.join(path, df_files.iloc[0]['FileName'])
    df_channels = get_channel_info_from_tiff(first_file_path)

    if df_channels is not None:
        print(f"Found {len(df_channels)} channels:")
        print(df_channels['Name'].to_string())
    else:
        print("No channel information available")

    # Step 5: Combine all tiles
    print("\n[Step 5/7] Combining tiles...")
    combined_array = combine_tiff_arrays(df_files, path)

    # Step 6: Build correct OME metadata dict (no manual XML string)
    print("\n[Step 6/7] Building OME metadata (dict)...")

    # Generate output filename (use last folder name)
    last_folder = os.path.basename(os.path.normpath(path))
    base_name = f"{last_folder}"
    output_filename = f"{base_name}_{output_suffix}.tif"
    output_path = os.path.join(path, output_filename)

    # Prepare tifffile OME metadata
    ome_meta = {
        'axes': 'CYX',
        'SignificantBits': 16,
    }

    # Add channel names in OME metadata
    if df_channels is not None and not df_channels.empty:
        ome_meta['Channel'] = [{'Name': str(name)} for name in df_channels['Name']]
    else:
        ome_meta['Channel'] = [{'Name': f'Channel_{idx}'} for idx in range(combined_array.shape[0])]

    # Step 7: Save the combined image using tifffile.imwrite with ome=True
    print("\n[Step 7/7] Saving combined image (OME-TIFF)...")
    imwrite(
        output_path,
        combined_array,
        bigtiff=True,
        compression=None,
        photometric='minisblack',
        ome=True,
        metadata=ome_meta,
    )

    print(f"\n✓ Successfully saved combined image to: {output_path}")
    print(f"  Shape: {combined_array.shape}")
    print(f"  Size: {os.path.getsize(output_path) / (1024 ** 2):.2f} MB")
    print("=" * 80)

    return output_path, combined_array, df_channels


def batch_process_folders(root_path: str, output_suffix: str = "Combined", exclude_files: list = None) -> dict:
    """
    Process all subfolders in a root directory, aligning and combining OME-TIFF tiles.
    
    Args:
        root_path: Root directory containing subfolders with TIFF tiles
        output_suffix: Suffix for output filenames (default: "Combined")
        exclude_files: List of text patterns to exclude. Files containing any of these 
                      patterns in their filename will be excluded from all subfolders 
                      (case-insensitive partial match). (default: None)
        
    Returns:
        Dictionary mapping subfolder paths to (output_path, combined_array, df_channels)
        
    Example:
        results = batch_process_folders(
            r"C:\path\to\root",
            exclude_files=['Combined', 'ome']
        )
    """
    if not os.path.exists(root_path):
        raise ValueError(f"Root path does not exist: {root_path}")
    
    print("\n" + "#"*80)
    print("# BATCH PROCESSING STARTED")
    print(f"# Root path: {root_path}")
    print("#"*80 + "\n")
    
    # Find all subdirectories
    subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path) 
               if os.path.isdir(os.path.join(root_path, d))]
    
    if not subdirs:
        print("Warning: No subdirectories found. Processing root directory only...")
        subdirs = [root_path]
    
    print(f"Found {len(subdirs)} subfolder(s) to process:\n")
    for i, subdir in enumerate(subdirs, 1):
        print(f"  {i}. {os.path.basename(subdir)}")
    print()
    
    # Process each subfolder
    results = {}
    success_count = 0
    failed_folders = []
    
    for i, subfolder in enumerate(subdirs, 1):
        print(f"\n{'='*80}")
        print(f"Processing folder {i}/{len(subdirs)}: {os.path.basename(subfolder)}")
        print(f"{'='*80}")
        
        try:
            output_path, combined_array, df_channels = OmeTiff_align(
                subfolder, 
                output_suffix=output_suffix,
                exclude_files=exclude_files
            )
            results[subfolder] = (output_path, combined_array, df_channels)
            success_count += 1
            print(f"\n✓ Folder {i}/{len(subdirs)} completed successfully")
            
        except Exception as e:
            print(f"\n✗ Error processing {subfolder}: {e}")
            failed_folders.append((subfolder, str(e)))
            results[subfolder] = None
    
    # Summary
    print("\n" + "#"*80)
    print("# BATCH PROCESSING COMPLETED")
    print("#"*80)
    print(f"\nTotal folders: {len(subdirs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_folders)}")
    
    if failed_folders:
        print("\nFailed folders:")
        for folder, error in failed_folders:
            print(f"  - {os.path.basename(folder)}: {error}")
    
    print("\n" + "#"*80 + "\n")
    
    return results


def plot_combined_image(combined_array: np.ndarray, df_channels: Optional[pd.DataFrame] = None, 
                       figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize the combined OME-TIFF image with all channels.
    
    Args:
        combined_array: Combined image array with shape (n_channels, height, width)
        df_channels: DataFrame with channel information (optional)
        figsize: Figure size (default: (15, 10))
    """
    n_channels = combined_array.shape[0]
    
    # Plot individual channels
    print(f"Plotting {n_channels} channels...\n")
    
    for i in range(n_channels):
        plt.figure(figsize=(10, 8))
        plt.imshow(combined_array[i], cmap='gray')
        
        if df_channels is not None and 'Name' in df_channels.columns:
            channel_name = df_channels['Name'].iloc[i]
            plt.title(f'Channel {i}: {channel_name}', fontsize=14, fontweight='bold')
        else:
            plt.title(f'Channel {i}', fontsize=14, fontweight='bold')
        
        plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.show()
    
    # Plot overlay of all channels
    print("\nPlotting overlay of all channels...\n")
    
    overlay_colors = [
        (0, 0, 1, 0.4),      # Blue
        (1, 0, 0, 0.4),      # Red
        (0.5, 0, 0.5, 0.4),  # Purple
        (0, 0.5, 0, 0.4),    # Green
        (1, 0.5, 0, 0.4),    # Orange
        (0.5, 0.5, 0.5, 0.4),# Grey
        (0, 1, 1, 0.4),      # Cyan
        (1, 0, 1, 0.4),      # Magenta
        (1, 1, 0, 0.4),      # Yellow
        (0, 0, 0, 0.4),      # Black
    ]
    
    # Create RGBA overlay image
    overlay_img = np.zeros((*combined_array.shape[1:], 4), dtype=np.float32)
    
    for i in range(n_channels):
        channel = combined_array[i]
        if channel.max() > 0:
            norm_channel = (channel - channel.min()) / (channel.max() - channel.min())
        else:
            norm_channel = channel
        
        color = overlay_colors[i % len(overlay_colors)]
        for c in range(4):
            overlay_img[..., c] += norm_channel * color[c]
    
    overlay_img = np.clip(overlay_img, 0, 1)
    
    plt.figure(figsize=figsize)
    plt.imshow(overlay_img)
    
    # Create legend
    if df_channels is not None and 'Name' in df_channels.columns:
        handles = [
            mpatches.Patch(color=overlay_colors[i], label=df_channels['Name'].iloc[i])
            for i in range(n_channels)
        ]
    else:
        handles = [
            mpatches.Patch(color=overlay_colors[i], label=f'Channel {i}')
            for i in range(n_channels)
        ]
    
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.title('Overlay of All Channels', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
