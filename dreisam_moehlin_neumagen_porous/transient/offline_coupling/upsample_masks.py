from pathlib import Path
import rasterio
from rasterio.enums import Resampling

base_path = Path(__file__).parent

areas = ["wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]

for area in areas:
    input_path = base_path.parent / "input" / f"{area}_.tif"
    output_path = base_path.parent / "input" / f"{area}_25m.tif"

    # Upscale factor: 50m → 25m = factor of 2
    scale_factor = 2

    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        new_width = src.width * scale_factor
        new_height = src.height * scale_factor

        # Update the transform to reflect new resolution
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )

        # Read and resample data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear  # Choose your resampling method
        )

        # Update metadata
        new_meta = src.meta.copy()
        new_meta.update({
            "width": new_width,
            "height": new_height,
            "transform": new_transform
        })

        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(data)

    print(f"Resampled to {new_width}x{new_height} pixels at 25m resolution.")