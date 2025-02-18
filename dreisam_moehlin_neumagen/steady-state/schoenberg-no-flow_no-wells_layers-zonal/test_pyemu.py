import pyemu

v = pyemu.utils.geostats.ExpVario(contribution=1.0,a=1000)
grid_gs = pyemu.utils.geostats.GeoStruct(variograms=v, transform='log')
temporal_gs = pyemu.utils.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))

grid_gs.plot()

temporal_gs.plot()

