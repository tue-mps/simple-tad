import os
import zipfile
from io import BytesIO
from PIL import Image
import concurrent.futures


def is_zip_corrupted(zip_path):
    """
    Returns True if the zip at zip_path is corrupted or contains unreadable images, False otherwise.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Quick CRC check for internal archive structure
            bad_file = z.testzip()
            if bad_file:
                print(f"CRC failed for {bad_file} in {zip_path}")
                return True

            # Try to open and verify each image
            for f in z.namelist():
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        data = z.read(f)
                        img = Image.open(BytesIO(data))
                        img.verify()  # verify image integrity
                    except Exception as e:
                        print(f"Image error in {zip_path} -> {f}: {e}")
                        return True
    except zipfile.BadZipFile as e:
        print(f"Bad zip file: {zip_path}: {e}")
        return True
    except Exception as e:
        print(f"Error processing {zip_path}: {e}")
        return True
    return False


def find_all_zip_paths(root_dir):
    """
    Walk root_dir recursively and collect all .zip file paths.
    """
    zip_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith('.zip'):
                zip_paths.append(os.path.join(dirpath, name))
    return zip_paths


def main():
    # Root directory containing the "frames" folder
    root = '/projects/0/prjs1424/sveta/RiskNetData/LOTVS-DADA/CAP-DATA/frames'

    print(f"Scanning for zip files under '{root}'...")
    all_zips = find_all_zip_paths(root)
    print(f"Found {len(all_zips)} zip files.")

    corrupted_zips = []

    # Use a process pool to parallelize zip checking
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map each zip path to a future
        futures = {executor.submit(is_zip_corrupted, zp): zp for zp in all_zips}
        for future in concurrent.futures.as_completed(futures):
            zp = futures[future]
            try:
                if future.result():
                    corrupted_zips.append(zp)
                    print(f"Corrupted: {zp}")
            except Exception as e:
                print(f"Unexpected error on {zp}: {e}")
                corrupted_zips.append(zp)

    # Report summary
    print("\n==== Summary ====")
    print(f"Total corrupted zip files: {len(corrupted_zips)}")

    # Save list to file
    out_file = 'corrupted_zips.txt'
    with open(out_file, 'w') as f:
        for zp in corrupted_zips:
            f.write(zp + '\n')
    print(f"List of corrupted zips written to {out_file}")


if __name__ == '__main__':
    main()
