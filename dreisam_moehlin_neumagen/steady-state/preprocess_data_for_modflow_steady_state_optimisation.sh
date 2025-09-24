#!/bin/bash

python add_masks.py
python modify_elevations.py
python modify_hydraulic_conductivity_and_specific_yield_of_layer1.py

python generate_boundary_conditions.py

cd identify_drainage_areas
python modflow6_steady-state.py --model-run 0
python write_binary_to_netcdf_steady-state.py --model-run 0
cd ..

python define_drainage_area.py

python prepare_groundwater_extraction_data.py

python make_stream_segment_routing.py
python repair_geometries_of_stream_segments.py
python write_sfr_data.py
python assign_streambed_conducivity.py