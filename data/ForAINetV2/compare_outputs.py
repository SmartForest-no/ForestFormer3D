import argparse
import os
import numpy as np
import glob

def compare_files(file1, file2):
    """Compares two .npy files."""
    if not os.path.exists(file2):
        return False, "File not found", None, None

    try:
        data1 = np.load(file1)
        data2 = np.load(file2)
    except Exception as e:
        return False, f"Error loading files: {e}", None, None

    # Determine comparison method based on file name
    is_float = any(suffix in os.path.basename(file1) for suffix in 
                   ['_vert.npy', '_unaligned_bbox.npy', '_aligned_bbox.npy', 
                    '_axis_align_matrix.npy', '_offsets.npy'])

    if is_float:
        are_equal = np.allclose(data1, data2)
        method = "np.allclose"
    else:
        are_equal = np.array_equal(data1, data2)
        method = "np.array_equal"
        
    if are_equal:
        message = "Identical"
    else:
        if data1.shape != data2.shape:
            message = f"Different shapes: {data1.shape} vs {data2.shape}"
        else:
            diff = np.abs(data1 - data2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            message = f"Different content (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}) using {method}"

    return are_equal, message, data1, data2

def main():
    parser = argparse.ArgumentParser(description="Compare .npy files in two directories.")
    parser.add_argument('dir1', help="First directory")
    parser.add_argument('dir2', help="Second directory")
    parser.add_argument('--verbose', action='store_true', help="Print array contents or stats for identical files.")
    args = parser.parse_args()

    print(f"Comparing directories:\n1: {args.dir1}\n2: {args.dir2}\n")

    files_to_compare = sorted(glob.glob(os.path.join(args.dir1, '*.npy')))
    
    if not files_to_compare:
        print(f"No .npy files found in {args.dir1}")
        return

    all_identical = True
    for file1_path in files_to_compare:
        filename = os.path.basename(file1_path)
        file2_path = os.path.join(args.dir2, filename)
        
        are_equal, message, data1, data2 = compare_files(file1_path, file2_path)
        
        status = "IDENTICAL" if are_equal else "DIFFERENT"
        print(f"{filename:<40} {status:<15} {message}")
        
        if args.verbose and are_equal and data1 is not None:
            print("  " + "-"*20 + " Visual Confirmation " + "-"*20)
            if data1.size < 20:
                print(f"  Content of {filename} (from {args.dir1}):\n{data1}")
            else:
                print(f"  Stats for {filename}:")
                print(f"    Shape: {data1.shape}")
                print(f"    Dtype: {data1.dtype}")
                if np.issubdtype(data1.dtype, np.floating):
                    print(f"    Min: {np.min(data1):.6f}, Max: {np.max(data1):.6f}, Mean: {np.mean(data1):.6f}")
                else:
                    print(f"    Min: {np.min(data1)}, Max: {np.max(data1)}, Mean: {np.mean(data1):.2f}")
            print("  " + "-"*59)


        if not are_equal:
            all_identical = False

    print("\n" + "="*50)
    if all_identical:
        print("All corresponding files are identical.")
    else:
        print("Some files have differences.")
    print("="*50)

if __name__ == '__main__':
    main()
