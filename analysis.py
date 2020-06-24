import yt
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from yt.data_objects.particle_filters import add_particle_filter
from yt.analysis_modules.halo_finding.api import HaloFinder
from yt.utilities.physical_constants import \
    gravitational_constant_cgs as G
from yt.units import mp
from yt.units import kboltz
import numpy as np
import os
import sys
from numpy import linalg as LA
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
from yt.analysis_modules.level_sets.api import *
import matplotlib.cm as cm
pi = 3.14159265

argvs = sys.argv
argc = len(argvs)

nSN = int(argvs[1])
inumber_min = int(argvs[2])
inumber_max = int(argvs[3])

SLICE   = int(argvs[4])
PROJ    = int(argvs[5])
PHASE   = int(argvs[6])
PROFILE = int(argvs[7])
TRACE   = int(argvs[8])
CLUMP   = int(argvs[9])

cm.register_cmap(name='cmap_dens',
   data={'red':   [[0.0     ,  0.0, 0.0],
                   [0.333333,  0.0, 0.0],
                   [0.666667,  0.0, 0.0],
                   [1.0     ,  1.0, 1.0]],
         'green': [[0.0     ,  0.0, 0.0],
                   [0.333333,  0.0, 0.0],
                   [0.666667,  1.0, 1.0],
                   [1.0     ,  1.0, 1.0]],
         'blue':  [[0.0     ,  0.0, 0.0],
                   [0.333333,  1.0, 1.0],
                   [0.666667,  1.0, 1.0],
                   [1.0     ,  1.0, 1.0]]})

def Stars(pfilter, data):
   return data[("all", "particle_type")] == 5
add_particle_filter("PopIII", function=Stars, filtered_type='all', requires=["particle_type"])

HydrogenFractionByMass   = 0.76
DeuteriumToHydrogenRatio = 3.4e-5 * 2.0
HeliumToHydrogenRatio    = (1.0 - HydrogenFractionByMass) / HydrogenFractionByMass
SolarMetalFractionByMass = 0.01295

if nSN ==-13:
    CarbonFractionToMetalByMass    = 2.65314e-01
    OxygenFractionToMetalByMass    = 3.00982e-01
    MagnesiumFractionToMetalByMass = 3.06651e-02
    AluminiumFractionToMetalByMass = 2.47296e-04
    SiliconFractionToMetalByMass   = 6.38319e-02
    SulfurFractionToMetalByMass    = 3.40910e-02
    IronFractionToMetalByMass      = 9.62448e-02
    LifeTime = 0.932594
    inumber_rad = 77
    inumber_exp = 87
    inumber_col = 109
    inumber_fin = 137

if nSN == 13:
    CarbonFractionToMetalByMass    = 6.69235e-01
    OxygenFractionToMetalByMass    = 3.30556e-01
    MagnesiumFractionToMetalByMass = 1.86824e-04
    AluminiumFractionToMetalByMass = 1.97017e-07
    SiliconFractionToMetalByMass   = 1.30184e-05
    SulfurFractionToMetalByMass    = 0.00000e+00
    IronFractionToMetalByMass      = 8.90341e-06
    LifeTime = 0.932594
    inumber_rad = 77
    inumber_exp = 97
    inumber_col = 109
    inumber_fin = 137

if nSN==50:
    CarbonFractionToMetalByMass    = 2.79167e-01
    OxygenFractionToMetalByMass    = 7.20575e-01
    MagnesiumFractionToMetalByMass = 2.49794e-04
    AluminiumFractionToMetalByMass = 1.66468e-08
    SiliconFractionToMetalByMass   = 4.01099e-06
    SulfurFractionToMetalByMass    = 0.00000e+00
    IronFractionToMetalByMass      = 4.15804e-06
    LifeTime = 0.275524
    inumber_rad = 80
    inumber_exp = 87
    inumber_col = 142
    inumber_fin = 174


if nSN==80:
    CarbonFractionToMetalByMass    = 2.52563e-01
    OxygenFractionToMetalByMass    = 7.46061e-01
    MagnesiumFractionToMetalByMass = 1.36917e-03
    AluminiumFractionToMetalByMass = 1.55602e-08
    SiliconFractionToMetalByMass   = 3.63906e-06
    SulfurFractionToMetalByMass    = 0.00000e+00
    IronFractionToMetalByMass      = 2.43915e-06
    LifeTime = 0.239350
    inumber_rad = 80
    inumber_exp = 87
    inumber_col = 191
    inumber_fin = 230


SolarIronAbundance = 7.50

# Pop III star creation time
if abs(nSN) == 13:
    cPopIII = 28.760813
if nSN == 50:
    cPopIII = 28.760723
if nSN == 80:
    cPopIII = 28.760723

SN = '%02d' % abs(nSN)
if nSN == -13:
    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_cgrackle_HR2"
else:
    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M" + SN #+ 'b'

if TRACE:
    pdir = indir + '/time_series'
    if not os.path.exists(pdir):
        os.mkdir(pdir)
    if TRACE==1:
        fn_trace = pdir + '/time_series.dat'
    if TRACE==2:
        fn_trace = pdir + '/time_series_mass.dat'
    fp_trace = open(fn_trace, mode='w')

for inumber in range(inumber_min, inumber_max+1):
    number  = '%04d' % inumber

    if (nSN==-13 and inumber<120) or (nSN==13 and inumber<86):
      outdir = "/media/genchiaki/enzo/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_cgrackle_HR2"
    elif (nSN==-13 and inumber>=120):
      outdir = "/media/genchiaki/enzo/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_cgrackle_HR2_g"
    else:
      outdir = "/media/genchiaki/enzo/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M" + SN
#     outdir = indir
    fn = outdir + "/DD" + number + "/output_" + number
    print(fn)
    ds = yt.load(fn) # load data
    ad = ds.all_data()
    tform = ds.arr(cPopIII, 'code_time')

    if TRACE==2:  # or (PROFILE or inumber < inumber_col):
        halos = HaloFinder(ds)
        halo = halos[0]
        halo_center = halo.center_of_mass()

    if inumber < inumber_rad:
        cp = ad.argmax("density")
    elif inumber < inumber_exp:
        ds.add_particle_filter('PopIII')
        MPopIII = ad[("PopIII", "particle_mass")].in_units("Msun")
        xPopIII = ad[("PopIII", "particle_position_x")].in_units("code_length")
        yPopIII = ad[("PopIII", "particle_position_y")].in_units("code_length")
        zPopIII = ad[("PopIII", "particle_position_z")].in_units("code_length")
        print("position of PopIII = %f %f %f" % (xPopIII[0], yPopIII[0], zPopIII[0]))
        print("M_PopIII = %f Msun" % MPopIII[0])
        cp = [xPopIII[0], yPopIII[0], zPopIII[0]]
        tPopIII = (ds.current_time - tform).in_units("Myr")
        tlife = ds.arr(LifeTime, 'code_time')
        print("Time after SF %f Myr / %f Myr" % (tPopIII, tlife.in_units("Myr")))
    else:
        if inumber < inumber_exp + 1:
            cp = ad.argmax("temperature")
        elif inumber < inumber_col:
            if TRACE==2: # or (PROFILE or inumber < inumber_col):
                cp = halo_center
            else:
                halos = HaloFinder(ds)
                halo = halos[0]
                cp = halo.center_of_mass()
        else:
            cp = ad.argmax("density")
        print("center point = %f %f %f" % (cp[0], cp[1], cp[2]))
        tlife = ds.arr(LifeTime, 'code_time')
        texp = tform + tlife
        tPopIII = (ds.current_time - texp).in_units("Myr")
        print("Time after SN %f Myr" % tPopIII)

    if TRACE==2: # or (PROFILE or inumber < inumber_col):
        H0 = ds.get_parameter('CosmologyHubbleConstantNow')
        OmegaM0 = ds.get_parameter('CosmologyOmegaMatterNow')
        OmegaL0 = ds.get_parameter('CosmologyOmegaLambdaNow')
        Hubble = ds.arr(100, 'km/s/Mpc')
        atime = 1.0/(1.0+ds.current_redshift)
        Hubble_a = H0*Hubble * (OmegaM0/atime**3+OmegaL0)**0.5
        rhocr = 3.0 * Hubble_a**2 / 8.0 / pi / G
        rhoth = 200.0 * rhocr
        Rvir0 = ds.arr(0.0, 'kpc')
        Rvir1 = ds.arr(1.0, 'kpc')
        for itr in range(64):
            Rvir=0.5*(Rvir0+Rvir1)
            hr = ds.sphere(halo_center, Rvir)
            Mdm  = hr.quantities.total_quantity("particle_mass").in_units("Msun")
            Mbar = hr.quantities.total_quantity("cell_mass").in_units("Msun")
            Mvir = Mdm+Mbar
            hrho = Mvir/(4.0*pi/3.0*Rvir**3)
            if hrho > rhoth:
                Rvir0 = Rvir
            else:
                Rvir1 = Rvir
        print("redshift %13.7f" % (ds.current_redshift))
        print("halo cent [ %13.7f, %13.7f, %13.7f ]" % (halo_center[0], halo_center[1], halo_center[2]))
        print("Mvir %13.5e Msun" % Mvir)
        print("Rvir %13.5e kpc " % Rvir)
        print("halo density %13.5e g/cc" % (hrho.in_units("g/cm**3")))
        print("crit density %13.5e g/cc" % (rhocr.in_units("g/cm**3")))
        halo_sphere = ds.sphere(halo_center, Rvir)

    sphere = ds.sphere(cp, (3, "kpc"))
    enriched = ad.cut_region(["obj['Zmet'] > 1.0e-8"])
    highdens = ad.cut_region(["obj['Hydrogen_number_density'] > 1.0e16"])

    # Estimate the initial time of SN simulation
  # rs_ST = ds.arr(10.0, 'pc')
  # xi = 2.0
  # E_SN  = ds.arr(1.0e51, 'erg')
  # nH0  = ds.arr(0.01, '1/cm**3')
  # rho0 = nH0 * mp / 0.76
  # t_ST = (rs_ST * (xi * E_SN / rho0)**(-0.2))**2.5
  # print(t_ST.in_units('yr'))

#   pos_Tmin = ad.argmin("temperature")
#   print("Max Temp %f K" % ad.max("temperature"))
#   print("Min Temp %f K at %f %f %f " % (ad.min("temperature"), pos_Tmin[0], pos_Tmin[1], pos_Tmin[2]))
#   norm = ds.arr([0.0,0.0,0.0], 'dimensionless')
#   dx = pos_Tmin[0] - cp[0]
#   dy = pos_Tmin[1] - cp[1]
#   dz = pos_Tmin[2] - cp[2]
#   dr = (dx*dx + dy*dy + dz*dz)**0.5
#   norm[0] = dx/dr
#   norm[1] = dy/dr
#   norm[2] = dz/dr
#   print("%f %f %f" % (dx.in_units("pc"), dy.in_units("pc"), dz.in_units("pc")))
#   print(norm)

    def _cell_size(field, data): return data["cell_volume"]**(1.0/3.0)
    ds.add_field(("gas", "cell_size"), function=_cell_size, sampling_type="cell", units="code_length")
    def _Hydrogen_number_density(field, data): return HydrogenFractionByMass * data["density"] /mp
    ds.add_field(("gas", "Hydrogen_number_density"), function=_Hydrogen_number_density, sampling_type="cell", units="cm**(-3)")
    def _dark_matter_number_density(field, data): return HydrogenFractionByMass * data["dark_matter_density"] /mp
    ds.add_field(("gas", "dark_matter_number_density"), function=_dark_matter_number_density, sampling_type="cell", units="cm**(-3)")
    def _Compressional_heating_rate(field, data): return data["pressure"] * data["velocity_divergence_absolute"]
    ds.add_field(("gas", "Compressional_heating_rate"), function=_Compressional_heating_rate, sampling_type="cell", units="erg/cm**3/s")

    def _Zmet(field, data): return data["SN_Colour"] / data["Density"] / SolarMetalFractionByMass
    ds.add_field(("gas", "Zmet"), function=_Zmet, sampling_type="cell", units="1")
    def _y_H2I(field, data): return data["H2I_Density"] / data["Density"] /HydrogenFractionByMass/2.0
    ds.add_field(("gas", "y_H2I"), function=_y_H2I, sampling_type="cell", units="1")
    def _y_HI(field, data): return data["HI_Density"] / data["Density"] /HydrogenFractionByMass/1.0
    ds.add_field(("gas", "y_HI"), function=_y_HI, sampling_type="cell", units="1")
    def _y_HII(field, data): return data["HII_Density"] / data["Density"] /HydrogenFractionByMass/1.0 
    ds.add_field(("gas", "y_HII"), function=_y_HII, sampling_type="cell", units="1")
    def _y_CII(field, data): return data["CII_Density"] / data["Density"] /HydrogenFractionByMass/12.0 
    ds.add_field(("gas", "y_CII"), function=_y_CII, sampling_type="cell", units="1")
    def _y_OII(field, data): return data["OII_Density"] / data["Density"] /HydrogenFractionByMass/16.0 
    ds.add_field(("gas", "y_OII"), function=_y_OII, sampling_type="cell", units="1")
    def _y_SiI(field, data): return data["SiI_Density"] / data["Density"] /HydrogenFractionByMass/32.0
    ds.add_field(("gas", "y_SiI"), function=_y_SiI, sampling_type="cell", units="1")
    def _y_SiOI(field, data): return data["SiOI_Density"] / data["Density"] /HydrogenFractionByMass/48.0
    ds.add_field(("gas", "y_SiOI"), function=_y_SiOI, sampling_type="cell", units="1")
    def _y_Mg2SiO4(field, data): return data["Mg2SiO4_Density"] / data["Density"] /HydrogenFractionByMass/140.0
    ds.add_field(("gas", "y_Mg2SiO4"), function=_y_Mg2SiO4, sampling_type="cell", units="1")
    def _y_AC(field, data): return data["AC_Density"] / data["Density"] /HydrogenFractionByMass/12.0
    ds.add_field(("gas", "y_AC"), function=_y_AC, sampling_type="cell", units="1")

    def _ThermalEnergy(field, data): return data["GasEnergy"] * data["cell_mass"]
    ds.add_field(("gas", "ThermalEnergy"), function=_ThermalEnergy, sampling_type="cell", units="erg")
    def _TotEnergy(field, data): return data["TotalEnergy"] * data["cell_mass"]
    ds.add_field(("gas", "TotEnergy"), function=_TotEnergy, sampling_type="cell", units="erg")

    def _molecular_weight(field, data):
        return data["Density"] / (
               data["Electron_Density"]
             + data[      "HI_Density"]
             + data[     "HII_Density"]
             + data[     "HeI_Density"] / 4.0
             + data[    "HeII_Density"] / 4.0
             + data[   "HeIII_Density"] / 4.0
             + data[      "HM_Density"]
             + data[     "H2I_Density"] / 2.0
             + data[    "H2II_Density"] / 2.0
             ) * mp
    ds.add_field(("gas", "molecular_weight"), function=_molecular_weight, sampling_type="cell", units="g")
    def _adiabatic_index(field, data):
        gamma0 = data["y_H2I"]
        gamma0[data["y_H2I"] < 0.25] = 5.0/3.0
        gamma0[data["y_H2I"] >=0.25] = 1.4
        iteration = 0;
        while True:
            k6100 = ds.arr(8.421956e-13, 'erg')
            kT = (gamma0-1.0) * data["GasEnergy"] * data["molecular_weight"].in_units("code_mass") 
            T6100 = k6100 / kT
            exp_T6100 = np.exp(T6100)
            gamma_minus1_inv_H2 = 0.5*(5.0 + 2.0 * T6100**2 * exp_T6100 / (exp_T6100-1.0)**2)
            gamma_minus1_inv_H2[T6100 > 100.0] = 2.5
            gamma =  1.0 + (
                   data["Electron_Density"]
                 + data[      "HI_Density"]
                 + data[     "HII_Density"]
                 + data[     "HeI_Density"] / 4.0
                 + data[    "HeII_Density"] / 4.0
                 + data[   "HeIII_Density"] / 4.0
                 + data[      "HM_Density"]
                 + data[     "H2I_Density"] / 2.0
                 + data[    "H2II_Density"] / 2.0
                 ) / (
                                   1.5 * data["Electron_Density"]
                 +                 1.5 * data[      "HI_Density"]
                 +                 1.5 * data[     "HII_Density"]
                 +                 1.5 * data[     "HeI_Density"] / 4.0
                 +                 1.5 * data[    "HeII_Density"] / 4.0
                 +                 1.5 * data[   "HeIII_Density"] / 4.0
                 +                 1.5 * data[      "HM_Density"]
                 + gamma_minus1_inv_H2 * data[     "H2I_Density"] / 2.0
                 + gamma_minus1_inv_H2 * data[    "H2II_Density"] / 2.0
                 )
            red = np.abs(gamma - gamma0)/gamma0;
            if ((len(red)>0 and red.max() < 1.0e-10) or len(red)==0) or iteration > 100:
                break
            gamma0 = gamma
            iteration = iteration + 1
        return gamma
    ds.add_field(("gas", "adiabatic_index"), function=_adiabatic_index, sampling_type="cell")
    def _temperature_corr(field, data): return (data["adiabatic_index"]-1.0) * data["GasEnergy"] * data["molecular_weight"] / kboltz
    ds.add_field(("gas", "temperature_corr"), function=_temperature_corr, sampling_type="cell", units="K")
    def _sound_speed_corr(field, data): return (data["adiabatic_index"] * kboltz * data["temperature_corr"] / data["molecular_weight"])**0.5
    ds.add_field(("gas", "sound_speed_corr"), function=_sound_speed_corr, sampling_type="cell", units="code_velocity")
    def _pressure_corr(field, data): return data["Density"] * kboltz * data["temperature_corr"] / data["molecular_weight"]
    ds.add_field(("gas", "pressure_corr"), function=_pressure_corr, sampling_type="cell", units="code_pressure")

    def _MetalMass(field, data): return data["SN_Colour"] * data["cell_volume"]
    ds.add_field(("gas", "MetalMass"), function=_MetalMass, sampling_type="cell", units="code_mass")

    def _HydrogenMass(field, data): return (
        data[   "HI_Density"]/ 1.0
      + data[  "HII_Density"]/ 1.0
      + data[  "H2I_Density"]/ 1.0
      + data[   "HM_Density"]/ 1.0
      + data[ "H2II_Density"]/ 1.0
      + data["HeHII_Density"]/ 5.0
      + data[  "HDI_Density"]/ 3.0
      + data[ "HDII_Density"]/ 3.0
      + data[   "OH_Density"]/17.0
      + data[  "H2O_Density"]/ 9.0
      + data[   "CH_Density"]/13.0
      + data[  "CH2_Density"]/ 7.0
      + data[ "OHII_Density"]/17.0
      + data["H2OII_Density"]/ 9.0
      + data["H3OII_Density"]/ 6.3
      ) * 1.0 * data["cell_volume"]
    ds.add_field(("gas", "HydrogenMass"), function=_HydrogenMass, sampling_type="cell", units="code_mass")

    def _CarbonMass(field, data): return (
        data[  "CI_Density"]/12.0
      + data[ "CII_Density"]/12.0
      + data[  "CO_Density"]/28.0
      + data[ "CO2_Density"]/44.0
      + data[  "CH_Density"]/13.0
      + data[ "CH2_Density"]/14.0
      + data["COII_Density"]/28.0
      + data[  "AC_Density"]/12.0
      ) * 12.0 * data["cell_volume"]
    ds.add_field(("gas", "CarbonMass"), function=_CarbonMass, sampling_type="cell", units="code_mass")

    def _OxygenMass(field, data): return (
        data[     "CO_Density"]/28.0
      + data[    "CO2_Density"]/22.0
      + data[     "OI_Density"]/16.0
      + data[     "OH_Density"]/17.0
      + data[    "H2O_Density"]/18.0
      + data[     "O2_Density"]/16.0
      + data[   "SiOI_Density"]/44.0
      + data[  "SiO2I_Density"]/30.0
      + data[   "COII_Density"]/28.0
      + data[    "OII_Density"]/16.0
      + data[   "OHII_Density"]/17.0
      + data[  "H2OII_Density"]/18.0
      + data[  "H3OII_Density"]/19.0
      + data[   "O2II_Density"]/16.0
      + data["Mg2SiO4_Density"]/35.0
      + data[ "MgSiO3_Density"]/33.3
      + data[  "Fe3O4_Density"]/58.0
      + data[  "SiO2D_Density"]/30.0
      + data[    "MgO_Density"]/40.0
      + data[  "Al2O3_Density"]/34.0
      ) * 16.0 * data["cell_volume"]
    ds.add_field(("gas", "OxygenMass"), function=_OxygenMass, sampling_type="cell", units="code_mass")

    def _SiliconMass(field, data): return (
        data[    "SiI_Density"]/ 28.0
      + data[   "SiOI_Density"]/ 44.0
      + data[  "SiO2I_Density"]/ 60.0
      + data[    "SiM_Density"]/ 28.0
      + data["Mg2SiO4_Density"]/140.0
      + data[ "MgSiO3_Density"]/100.0
      + data[  "SiO2D_Density"]/60.0
      ) * 28.0 * data["cell_volume"]
    ds.add_field(("gas", "SiliconMass"), function=_SiliconMass, sampling_type="cell", units="code_mass")

    def _IronMass(field, data): return (
        data[   "Fe_Density"]/56.0
      + data[  "FeM_Density"]/56.0
      + data["Fe3O4_Density"]/77.3
      + data[  "FeS_Density"]/88.0
      ) * 56.0 * data["cell_volume"]
    ds.add_field(("gas", "IronMass"), function=_IronMass, sampling_type="cell", units="code_mass")

    def _HydrogenFraction(field, data): return data["HydrogenMass"] / data["cell_mass"] / HydrogenFractionByMass
    ds.add_field(("gas", "HydrogenFraction"), function=_HydrogenFraction, sampling_type="cell", units="1")
    def _CarbonFraction(field, data): return data["CarbonMass"] / data["MetalMass"] / CarbonFractionToMetalByMass
    ds.add_field(("gas", "CarbonFraction"), function=_CarbonFraction, sampling_type="cell", units="1")
    def _OxygenFraction(field, data): return data["OxygenMass"] / data["MetalMass"] / OxygenFractionToMetalByMass
    ds.add_field(("gas", "OxygenFraction"), function=_OxygenFraction, sampling_type="cell", units="1")
    def _SiliconFraction(field, data): return data["SiliconMass"] / data["MetalMass"] / SiliconFractionToMetalByMass
    ds.add_field(("gas", "SiliconFraction"), function=_SiliconFraction, sampling_type="cell", units="1")
    def _IronFraction(field, data): return data["IronMass"] / data["MetalMass"] / IronFractionToMetalByMass
    ds.add_field(("gas", "IronFraction"), function=_IronFraction, sampling_type="cell", units="1")

    def _CarbonAbundance(field, data): return 1.0e12 * (data["CarbonMass"] / 12.0 / data["HydrogenMass"])
    ds.add_field(("gas", "CarbonAbundance"), function=_CarbonAbundance, sampling_type="cell")
    def _IronAbundanceToSolar(field, data): return 1.0e12 * (data["IronMass"] / 56.0 / data["HydrogenMass"]) / 10.0**SolarIronAbundance
    ds.add_field(("gas", "IronAbundanceToSolar"), function=_IronAbundanceToSolar, sampling_type="cell")

    def _CarbonDustMass(field, data): return data["AC_Density"] * data["cell_volume"]
    ds.add_field(("gas", "CarbonDustMass"), function=_CarbonDustMass, sampling_type="cell", units="code_mass")
    def _CarbonCondensationEfficiency(field, data): return data["CarbonDustMass"] / data["CarbonMass"]
    ds.add_field(("gas", "CarbonCondensationEfficiency"), function=_CarbonCondensationEfficiency, sampling_type="cell")

    nHmax = ad.max("Hydrogen_number_density")
    posi_nHmax = ad.argmax("density")
    print("Max density %e at %f %f %f" % (
          nHmax
        , (posi_nHmax[0] - cp[0]).in_units("pc")
        , (posi_nHmax[1] - cp[1]).in_units("pc")
        , (posi_nHmax[2] - cp[2]).in_units("pc")))
    factor_nHmax = 3.0
    if TRACE:
        factor_nHmax = 3.0
    if PROJ or CLUMP:
        factor_nHmax = 100.0
    nH_thr = nHmax / factor_nHmax

##  swept = ds.sphere(cp, (1, "kpc"))
##  MetalMass_in_swept = swept.sum("MetalMass").in_units("Msun")
##  print("MetalMass     %e Msun" % (MetalMass_in_swept ))
####if inumber >= inumber_exp:
####    print("Max Temp %f K" % enriched.max("temperature"))
####    print("Min Temp %f K" % enriched.min("temperature"))

    if inumber < inumber_exp:
        width = (1.0, 'kpc')
        axes_unit='kpc'
    elif inumber < inumber_col:
        width = (2.0, 'kpc')
        axes_unit='kpc'
    else:
        cs_nHmax = ad.argmax("density", axis="sound_speed")
        l_J = (pi * cs_nHmax**2 / G / ad.max("density"))**0.5
        print("dens %e cs %e l_J %e" % (ad.max("Hydrogen_number_density"), cs_nHmax.in_units("km/s"), (l_J.in_units("pc")) ) )
     
        core_trial = ds.sphere(cp, l_J)
        core = ds.sphere(cp, 10.0*l_J).cut_region([("obj['Hydrogen_number_density'] > %e" % nH_thr)])
        # REMOVE SPHERE AROUND DM PARTICLE
        r_arDM = ds.arr(0.1, 'pc')
        xDM = core_trial[("all", "particle_position_x")]
        yDM = core_trial[("all", "particle_position_y")]
        zDM = core_trial[("all", "particle_position_z")]
        nDM = xDM.size
        if nDM > 0:
            print("%d DM particles are detected" % nDM)
            radiDM = ds.arr([0.0]*nDM)
            for iDM in range(nDM):
                radiDM[iDM] = ((xDM[iDM]-cp[0])**2+(yDM[iDM]-cp[1])**2+(zDM[iDM]-cp[2])**2)**0.5
            iDM = np.argmin(radiDM)
            print("DM particle [%23.15f, %23.15f, %23.15f]" % (xDM[iDM], yDM[iDM], zDM[iDM]))
            print("       cent [%23.15f, %23.15f, %23.15f]" % (cp[0],    cp[1],    cp[2]   ))
          # print("r_arDM      %e" % (r_arDM.in_units("code_length")))
            sphere_rem = ds.sphere([xDM[iDM], yDM[iDM], zDM[iDM]], 10.0*l_J).cut_region([("obj['radius'] > %e" % r_arDM.in_units("cm"))])
            cp = sphere_rem.argmax("density")
            nHmax     = sphere_rem.max("Hydrogen_number_density")
            nH_thr = nHmax / factor_nHmax
            rho_nHmax = sphere_rem.max("density")
            cs_nHmax  = sphere_rem.argmax("density", axis="sound_speed")
            l_J = (pi * cs_nHmax**2 / G / rho_nHmax)**0.5
            print("dens %e cs %e l_J %e" % ((HydrogenFractionByMass*rho_nHmax/mp).in_units("cm**(-3)"), cs_nHmax.in_units("km/s"), (l_J.in_units("pc")) ) )
  #####     core = ds.sphere(cp, l_J)
            core = sphere_rem.cut_region([("obj['Hydrogen_number_density'] > %e" % nH_thr)])
        ##########################################################
        width = factor_nHmax * l_J
        if PROJ:
            if nSN==13 and inumber==123: width = ds.arr(7.5  , 'pc')
            if nSN==13 and inumber==130: width = ds.arr(0.05 , 'pc')
            if nSN==13 and inumber==137: width = ds.arr( 50.0, 'au')
            if nSN==50 and inumber==174: width = ds.arr( 50.0, 'au')
            if nSN==80 and inumber==217: width = ds.arr(1.5  , 'pc')
            if nSN==80 and inumber==223: width = ds.arr(0.05 , 'pc')
            if nSN==80 and inumber==230: width = ds.arr( 50.0, 'au')

        if width > ds.arr(100, 'pc'):
          axes_unit='kpc'
        elif width > ds.arr(1.0e-3, 'pc'):
          axes_unit='pc'
        else:
          axes_unit='AU'

        if TRACE==1:
            core_dens   = core.mean("Density"  , weight="cell_volume").in_units("g/cm**3")
            core_eng    = core.mean("GasEnergy", weight="cell_mass")  .in_units("(cm/s)**2")
            core_metal  = core.mean("SN_Colour", weight="cell_volume").in_units("g/cm**3")
            core_elec   = core.mean("Electron_Density", weight="cell_volume").in_units("g/cm**3")
            core_HI     = core.mean(      "HI_Density", weight="cell_volume").in_units("g/cm**3")
            core_HII    = core.mean(     "HII_Density", weight="cell_volume").in_units("g/cm**3")
            core_H2I    = core.mean(     "H2I_Density", weight="cell_volume").in_units("g/cm**3")
            core_HM     = core.mean(      "HM_Density", weight="cell_volume").in_units("g/cm**3")
            core_H2II   = core.mean(    "H2II_Density", weight="cell_volume").in_units("g/cm**3")
            core_DI     = core.mean(      "DI_Density", weight="cell_volume").in_units("g/cm**3")
            core_DII    = core.mean(     "DII_Density", weight="cell_volume").in_units("g/cm**3")
            core_DM     = core.mean(      "DM_Density", weight="cell_volume").in_units("g/cm**3")
            core_HDI    = core.mean(     "HDI_Density", weight="cell_volume").in_units("g/cm**3")
            core_HDII   = core.mean(    "HDII_Density", weight="cell_volume").in_units("g/cm**3")
            core_HeHII  = core.mean(   "HeHII_Density", weight="cell_volume").in_units("g/cm**3")
            core_HeI    = core.mean(     "HeI_Density", weight="cell_volume").in_units("g/cm**3")
            core_HeII   = core.mean(    "HeII_Density", weight="cell_volume").in_units("g/cm**3")
            core_HeIII  = core.mean(   "HeIII_Density", weight="cell_volume").in_units("g/cm**3")
            core_CII    = core.mean(     "CII_Density", weight="cell_volume").in_units("g/cm**3")
            core_CI     = core.mean(      "CI_Density", weight="cell_volume").in_units("g/cm**3")
            core_CO     = core.mean(      "CO_Density", weight="cell_volume").in_units("g/cm**3")
            core_CO2    = core.mean(     "CO2_Density", weight="cell_volume").in_units("g/cm**3")
            core_CH     = core.mean(      "CH_Density", weight="cell_volume").in_units("g/cm**3")
            core_CH2    = core.mean(     "CH2_Density", weight="cell_volume").in_units("g/cm**3")
            core_COII   = core.mean(    "COII_Density", weight="cell_volume").in_units("g/cm**3")
            core_OI     = core.mean(      "OI_Density", weight="cell_volume").in_units("g/cm**3")
            core_OH     = core.mean(      "OH_Density", weight="cell_volume").in_units("g/cm**3")
            core_H2O    = core.mean(     "H2O_Density", weight="cell_volume").in_units("g/cm**3")
            core_O2     = core.mean(      "O2_Density", weight="cell_volume").in_units("g/cm**3")
            core_OII    = core.mean(     "OII_Density", weight="cell_volume").in_units("g/cm**3")
            core_OHII   = core.mean(    "OHII_Density", weight="cell_volume").in_units("g/cm**3")
            core_H2OII  = core.mean(   "H2OII_Density", weight="cell_volume").in_units("g/cm**3")
            core_H3OII  = core.mean(   "H3OII_Density", weight="cell_volume").in_units("g/cm**3")
            core_O2II   = core.mean(    "O2II_Density", weight="cell_volume").in_units("g/cm**3")
            core_SiI    = core.mean(     "SiI_Density", weight="cell_volume").in_units("g/cm**3")
            core_SiOI   = core.mean(    "SiOI_Density", weight="cell_volume").in_units("g/cm**3")
            core_SiO2I  = core.mean(   "SiO2I_Density", weight="cell_volume").in_units("g/cm**3")
            core_Mg     = core.mean(      "Mg_Density", weight="cell_volume").in_units("g/cm**3")
            core_Al     = core.mean(      "Al_Density", weight="cell_volume").in_units("g/cm**3")
            core_S      = core.mean(       "S_Density", weight="cell_volume").in_units("g/cm**3")
            core_Fe     = core.mean(      "Fe_Density", weight="cell_volume").in_units("g/cm**3")
            core_SiM    = core.mean(     "SiM_Density", weight="cell_volume").in_units("g/cm**3")
            core_FeM    = core.mean(     "FeM_Density", weight="cell_volume").in_units("g/cm**3")
            core_Mg2SiO4= core.mean( "Mg2SiO4_Density", weight="cell_volume").in_units("g/cm**3")
            core_MgSiO3 = core.mean(  "MgSiO3_Density", weight="cell_volume").in_units("g/cm**3")
            core_Fe3O4  = core.mean(   "Fe3O4_Density", weight="cell_volume").in_units("g/cm**3")
            core_AC     = core.mean(      "AC_Density", weight="cell_volume").in_units("g/cm**3")
            core_SiO2D  = core.mean(   "SiO2D_Density", weight="cell_volume").in_units("g/cm**3")
            core_MgO    = core.mean(     "MgO_Density", weight="cell_volume").in_units("g/cm**3")
            core_FeS    = core.mean(     "FeS_Density", weight="cell_volume").in_units("g/cm**3")
            core_Al2O3  = core.mean(   "Al2O3_Density", weight="cell_volume").in_units("g/cm**3")
            core_Gcomp  = core.mean("Compressional_heating_rate", weight="cell_volume").in_units("erg/cm**3/s")
            core_sigv   = core.std ("velocity_magnitude"        , weight="cell_mass"  )
            core_cs     = core.mean("sound_speed"               , weight="cell_mass"  ).in_units("cm/s")

            fp_trace.write("%23.15e " % tPopIII.in_units("s"))
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e "        % ( core_dens   , core_eng    , core_metal  , core_elec                ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( core_HI     , core_HII    , core_H2I    , core_HM     , core_H2II  ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( core_DI     , core_DII    , core_DM     , core_HDI    , core_HDII  ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e "        % ( core_HeHII  , core_HeI    , core_HeII   , core_HeIII               ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e "        % ( core_CII    , core_CI     , core_CO     , core_CO2                 ) )
            fp_trace.write("%13.5e %13.5e %13.5e "               % ( core_CH     , core_CH2    , core_COII                              ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e "        % ( core_OI     , core_OH     , core_H2O    , core_O2                  ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( core_OII    , core_OHII   , core_H2OII  , core_H3OII  , core_O2II  ) )
            fp_trace.write("%13.5e %13.5e %13.5e "               % ( core_SiI    , core_SiOI   , core_SiO2I                             ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e "        % ( core_Mg     , core_Al     , core_S      , core_Fe                  ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( core_SiM    , core_FeM    , core_Mg2SiO4, core_MgSiO3 , core_Fe3O4 ) )
            fp_trace.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( core_AC     , core_SiO2D  , core_MgO    , core_FeS    , core_Al2O3 ) )
            fp_trace.write("%13.5e %13.5e %13.5e " % (core_Gcomp, core_sigv, core_cs) )

        if TRACE==2:
            cloud_sphere = ds.sphere(cp, l_J)

    if TRACE==2:
        DMMass_in_halo     = halo_sphere.sum("particle_mass").in_units("Msun")
        GasMass_in_halo    = halo_sphere.sum("cell_mass").in_units("Msun")
        MetalMass_in_halo  = halo_sphere.sum("MetalMass" ).in_units("Msun")
        CarbonMass_in_halo = halo_sphere.sum("CarbonMass").in_units("Msun")
        OxygenMass_in_halo = halo_sphere.sum("OxygenMass").in_units("Msun")
        IronMass_in_halo   = halo_sphere.sum("IronMass"  ).in_units("Msun")
        if inumber >= inumber_exp:
            GasMass_in_enriched    = enriched.sum("cell_mass" ).in_units("Msun")
            MetalMass_in_enriched  = enriched.sum("MetalMass" ).in_units("Msun")
            CarbonMass_in_enriched = enriched.sum("CarbonMass").in_units("Msun")
            OxygenMass_in_enriched = enriched.sum("OxygenMass").in_units("Msun")
            IronMass_in_enriched   = enriched.sum("IronMass"  ).in_units("Msun")
        else:
            GasMass_in_enriched    = 0.0
            MetalMass_in_enriched  = 0.0
            CarbonMass_in_enriched = 0.0
            OxygenMass_in_enriched = 0.0
            IronMass_in_enriched   = 0.0
        if inumber >= inumber_col:
            GasMass_in_cloud    = cloud_sphere.sum("cell_mass" ).in_units("Msun")
            MetalMass_in_cloud  = cloud_sphere.sum("MetalMass" ).in_units("Msun")
            CarbonMass_in_cloud = cloud_sphere.sum("CarbonMass").in_units("Msun")
            OxygenMass_in_cloud = cloud_sphere.sum("OxygenMass").in_units("Msun")
            IronMass_in_cloud   = cloud_sphere.sum("IronMass"  ).in_units("Msun")
            CarbonMass_in_core     = core.sum("CarbonMass").in_units("Msun")
            CarbonDustMass_in_core = core.sum("CarbonDustMass").in_units("Msun")
        else:
            l_J = ds.arr(0.0, "code_length")
            GasMass_in_cloud    = 0.0
            MetalMass_in_cloud  = 0.0
            CarbonMass_in_cloud = 0.0
            OxygenMass_in_cloud = 0.0
            IronMass_in_cloud   = 0.0
            CarbonMass_in_core     = 0.0
            CarbonDustMass_in_core = 0.0
        fp_trace.write("%23.17f " % ( ds.current_redshift             ) ) #  0
        fp_trace.write("%23.17f " % ( ds.current_time.in_units("Myr") ) ) #  1
        fp_trace.write("%13.5e " % ( Rvir.in_units("pc") ) )              #  2
        fp_trace.write("%13.5e " % ( DMMass_in_halo     ) )               #  3
        fp_trace.write("%13.5e " % ( GasMass_in_halo    ) )               #  4
        fp_trace.write("%13.5e " % ( MetalMass_in_halo  ) )               #  5
        fp_trace.write("%13.5e " % ( CarbonMass_in_halo ) )               #  6
        fp_trace.write("%13.5e " % ( OxygenMass_in_halo ) )               #  7
        fp_trace.write("%13.5e " % ( IronMass_in_halo   ) )               #  8
        fp_trace.write("%13.5e " % ( GasMass_in_enriched    ) )           #  9
        fp_trace.write("%13.5e " % ( MetalMass_in_enriched  ) )           # 10
        fp_trace.write("%13.5e " % ( CarbonMass_in_enriched ) )           # 11
        fp_trace.write("%13.5e " % ( OxygenMass_in_enriched ) )           # 12
        fp_trace.write("%13.5e " % ( IronMass_in_enriched   ) )           # 13
        fp_trace.write("%13.5e " % ( l_J.in_units("pc") ) )               # 14
        fp_trace.write("%13.5e " % ( nHmax ) )                            # 15
        fp_trace.write("%13.5e " % ( GasMass_in_cloud    ) )              # 16
        fp_trace.write("%13.5e " % ( MetalMass_in_cloud  ) )              # 17
        fp_trace.write("%13.5e " % ( CarbonMass_in_cloud ) )              # 18
        fp_trace.write("%13.5e " % ( OxygenMass_in_cloud ) )              # 19
        fp_trace.write("%13.5e " % ( IronMass_in_cloud   ) )              # 20
        fp_trace.write("%13.5e " % ( CarbonMass_in_core     ) )           # 21
        fp_trace.write("%13.5e " % ( CarbonDustMass_in_core ) )           # 22
        fp_trace.write("\n")                                          


# ELLIPTICITY OF GAS CLOUD
    if (nSN==80 and inumber==79) or inumber >= inumber_col:
#       width = (1.2, 'kpc')
#       axes_unit = 'kpc'
        
        if nSN==80 and inumber==79:
            bigsphere = ds.sphere(cp, (0.5, "kpc"))
            filament = bigsphere.cut_region([("obj['Hydrogen_number_density'] > %f" % 0.1)])
        else:
            filament = core

#       pdir = indir + '/Projection_z'
#       plot = yt.ProjectionPlot(ds, 'z', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp, data_source=filament)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))
        
        def _moi00(field, data): return data["cell_mass"] * ((data["y"]-cp[1])**2 + (data["z"]-cp[2])**2)
        ds.add_field(("gas", "moi00"), function=_moi00, sampling_type="cell", units="code_mass*code_length**2")
        def _moi01(field, data): return data["cell_mass"] * ( -(data["x"]-cp[0]) * (data["y"]-cp[1]))
        ds.add_field(("gas", "moi01"), function=_moi01, sampling_type="cell", units="code_mass*code_length**2")
        def _moi02(field, data): return data["cell_mass"] * ( -(data["x"]-cp[0]) * (data["z"]-cp[2]))
        ds.add_field(("gas", "moi02"), function=_moi02, sampling_type="cell", units="code_mass*code_length**2")
        def _moi10(field, data): return data["cell_mass"] * ( -(data["y"]-cp[1]) * (data["x"]-cp[0]))
        ds.add_field(("gas", "moi10"), function=_moi10, sampling_type="cell", units="code_mass*code_length**2")
        def _moi11(field, data): return data["cell_mass"] * ((data["x"]-cp[0])**2 + (data["z"]-cp[2])**2)
        ds.add_field(("gas", "moi11"), function=_moi11, sampling_type="cell", units="code_mass*code_length**2")
        def _moi12(field, data): return data["cell_mass"] * ( -(data["y"]-cp[1]) * (data["z"]-cp[2]))
        ds.add_field(("gas", "moi12"), function=_moi12, sampling_type="cell", units="code_mass*code_length**2")
        def _moi20(field, data): return data["cell_mass"] * ( -(data["z"]-cp[2]) * (data["x"]-cp[0]))
        ds.add_field(("gas", "moi20"), function=_moi20, sampling_type="cell", units="code_mass*code_length**2")
        def _moi21(field, data): return data["cell_mass"] * ( -(data["z"]-cp[2]) * (data["y"]-cp[1]))
        ds.add_field(("gas", "moi21"), function=_moi21, sampling_type="cell", units="code_mass*code_length**2")
        def _moi22(field, data): return data["cell_mass"] * ((data["x"]-cp[0])**2 + (data["y"]-cp[1])**2)
        ds.add_field(("gas", "moi22"), function=_moi22, sampling_type="cell", units="code_mass*code_length**2")

        if nSN==80 and inumber==79:
            ref_unit_mass = "Msun"
            ref_unit_leng = "kpc"
        else:
            ref_unit_mass = "Msun"
            ref_unit_leng = "au"
        ref_unit_moi = ref_unit_mass + '*' + ref_unit_leng + '**2'

        moi = np.array([[0.0]*3]*3)
        moi[0, 0] = filament.sum('moi00').in_units(ref_unit_moi)
        moi[1, 0] = filament.sum('moi01').in_units(ref_unit_moi)
        moi[2, 0] = filament.sum('moi02').in_units(ref_unit_moi)
        moi[0, 1] = filament.sum('moi10').in_units(ref_unit_moi)
        moi[1, 1] = filament.sum('moi11').in_units(ref_unit_moi)
        moi[2, 1] = filament.sum('moi12').in_units(ref_unit_moi)
        moi[0, 2] = filament.sum('moi20').in_units(ref_unit_moi)
        moi[1, 2] = filament.sum('moi21').in_units(ref_unit_moi)
        moi[2, 2] = filament.sum('moi22').in_units(ref_unit_moi)
        mtot = np.array(filament.sum('cell_mass').in_units(ref_unit_mass))
   ##   print(moi)
   ##   print(mtot)
        
        moi_l, moi_lv = LA.eig(moi)
   ##   print(moi_l)
   ##   print(moi_lv)
        
        lax = np.array([0.0]*3)
        lax[0] = (2.5*(-moi_l[0]+moi_l[1]+moi_l[2])/mtot)**0.5
        lax[1] = (2.5*( moi_l[0]-moi_l[1]+moi_l[2])/mtot)**0.5   
        lax[2] = (2.5*( moi_l[0]+moi_l[1]-moi_l[2])/mtot)**0.5
    ##  print(lax)
        laxis = ds.arr(lax, ref_unit_leng)
        aaxis = ds.arr(moi_lv, 'dimensionless')
#       print(laxis)
#       print('axis0 %13.7f [%13.7f, %13.7f, %13.7f]' % (laxis[0], aaxis[0,0], aaxis[1,0], aaxis[2,0]))
#       print('axis1 %13.7f [%13.7f, %13.7f, %13.7f]' % (laxis[1], aaxis[0,1], aaxis[1,1], aaxis[2,1]))
#       print('axis2 %13.7f [%13.7f, %13.7f, %13.7f]' % (laxis[2], aaxis[0,2], aaxis[1,2], aaxis[2,2]))

        iaxis_max = np.argmax(lax)
        iaxis_min = np.argmin(lax)
        iaxis_mid = 3 - iaxis_max - iaxis_min
#       print("%d %d %d" % (iaxis_max, iaxis_mid, iaxis_min))
        axis0 = moi_lv[:, iaxis_max]
        axis1 = moi_lv[:, iaxis_mid]
        axis2 = moi_lv[:, iaxis_min]
#       print('axis0 %13.7f [%13.7f, %13.7f, %13.7f]' % (lax[iaxis_max], axis0[0], axis0[1], axis0[2]))
#       print('axis1 %13.7f [%13.7f, %13.7f, %13.7f]' % (lax[iaxis_mid], axis1[0], axis1[1], axis1[2]))
#       print('axis2 %13.7f [%13.7f, %13.7f, %13.7f]' % (lax[iaxis_min], axis2[0], axis2[1], axis2[2]))
        
        laaxis = ds.arr([[0.0]*3]*3, 'code_length')
        laaxis[0, 0] = cp[0] + laxis[0].in_units('code_length') * aaxis[0, 0]
        laaxis[1, 0] = cp[1] + laxis[0].in_units('code_length') * aaxis[1, 0]
        laaxis[2, 0] = cp[2] + laxis[0].in_units('code_length') * aaxis[2, 0]
        laaxis[0, 1] = cp[0] + laxis[1].in_units('code_length') * aaxis[0, 1]
        laaxis[1, 1] = cp[1] + laxis[1].in_units('code_length') * aaxis[1, 1]
        laaxis[2, 1] = cp[2] + laxis[1].in_units('code_length') * aaxis[2, 1]
        laaxis[0, 2] = cp[0] + laxis[2].in_units('code_length') * aaxis[0, 2]
        laaxis[1, 2] = cp[1] + laxis[2].in_units('code_length') * aaxis[1, 2]
        laaxis[2, 2] = cp[2] + laxis[2].in_units('code_length') * aaxis[2, 2]
        
        endpoint0 = np.array(laaxis[:, 0])
        endpoint1 = np.array(laaxis[:, 1])
        endpoint2 = np.array(laaxis[:, 2])
        
        pdir = indir + '/Projection_z'
#       plot = yt.OffAxisSlicePlot(ds, axis1 , "Hydrogen_number_density", center=cp, width = width, axes_unit=axes_unit, north_vector= axis2, data_source=filament)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))

#       plot = yt.ProjectionPlot(ds, 'z', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp, data_source=ad)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.annotate_arrow(endpoint0, starting_pos = cp, plot_args={'color':'yellow'})
#       plot.annotate_arrow(endpoint1, starting_pos = cp, plot_args={'color':'cyan'})
#       plot.annotate_arrow(endpoint2, starting_pos = cp, plot_args={'color':'pink'})
#       plot.annotate_particles(width = width, p_size=5.0, col='black', marker='o')
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))
        
#       pdir = indir + '/Projection_y'
#       plot = yt.ProjectionPlot(ds, 'y', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp, data_source=filament)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.annotate_arrow(endpoint0, starting_pos = cp, plot_args={'color':'yellow'})
#       plot.annotate_arrow(endpoint1, starting_pos = cp, plot_args={'color':'cyan'})
#       plot.annotate_arrow(endpoint2, starting_pos = cp, plot_args={'color':'pink'})
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))
        
#       pdir = indir + '/Projection_x'
#       plot = yt.OffAxisSlicePlot(ds, axis2 , "Hydrogen_number_density", center=cp, width = width, axes_unit=axes_unit, north_vector=-axis1, data_source=filament)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))

#       plot = yt.ProjectionPlot(ds, 'x', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp, data_source=filament)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.annotate_arrow(endpoint0, starting_pos = cp, plot_args={'color':'yellow'})
#       plot.annotate_arrow(endpoint1, starting_pos = cp, plot_args={'color':'cyan'})
#       plot.annotate_arrow(endpoint2, starting_pos = cp, plot_args={'color':'pink'})
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))
        
#       pdir = indir + '/Slice_z'
#       plot = yt.OffAxisSlicePlot(ds, axis1 , "Hydrogen_number_density", center=cp, width = width, axes_unit=axes_unit, north_vector= axis2)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
#       plot.save('%s/filament_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))
#       ofaxslc = plot.data_source.to_frb(width, 500)
#       print(ofaxslc["Hydrogen_number_density"])
    else:
        axis0 = [   -0.2781099,     0.7581359,     0.5898176] # 0.3576712 
        axis1 = [    0.3176577,    -0.5069014,     0.8013392] # 0.1382566 
        axis2 = [    0.9065034,     0.4102205,    -0.0998534] # 0.1137016 

    if TRACE==1:
        fp_trace.write("%13.5e %13.5e %13.5e" % (lax[iaxis_max], lax[iaxis_mid], lax[iaxis_min]) )
        fp_trace.write("\n")


    if SLICE:
        pdir = indir + '/Slice_z'
        if not os.path.exists(pdir):
            os.mkdir(pdir)

##      plot = yt.SlicePlot(ds, 'z', "cell_size", width = width, axes_unit=axes_unit, center=cp)
##      plot.set_unit('cell_size', 'pc')
##      plot.annotate_grids()
##      plot.save("%s/cell_size_%04d.png" % (pdir, inumber))
#      
#       plot = yt.SlicePlot(ds, 'z', "Hydrogen_number_density", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
##      plot.annotate_grids()
#       if inumber >= inumber_rad and inumber < inumber_col:
#           plot.annotate_text((0.05, 0.05), 't = %.2f Myr' % tPopIII, coord_system='axis')
#       else :
#           plot.annotate_text((0.05, 0.05), 'n$_{H,max}$ = %.2e cm$^{-3}$' % nHmax, coord_system='axis')
#       plot.save("%s/Hydrogen_number_density_%04d.png" % (pdir, inumber))
#      
#       plot = yt.SlicePlot(ds, 'z', "temperature", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='temperature', cmap="hot")
#       plot.save("%s/temperature_%04d.png" % (pdir, inumber))
#
#       if inumber >= inumber_exp:
#           plot = yt.SlicePlot(ds, 'z', "Zmet", width = width, axes_unit=axes_unit, center=cp)
##          plot.annotate_grids()
#           plot.save("%s/Zmet_%04d.png" % (pdir, inumber))
#
#           plot = yt.SlicePlot(ds, 'z', "CarbonAbundance", width = width, axes_unit=axes_unit, center=cp)
##          plot.set_log('CarbonAbundance', False)
##          plot.set_zlabel(r'$10^{A({\rm C})}$')
#           plot.save("%s/CarbonAbundance_%04d.png" % (pdir, inumber))
#
#           plot = yt.SlicePlot(ds, 'z', "IronAbundanceToSolar", width = width, axes_unit=axes_unit, center=cp)
##          plot.set_log('IronAbundanceToSolar', False)
##          plot.set_zlabel(r'$10^{[Fe/H]}$')
#           plot.save("%s/IronAbundanceToSolar_%04d.png" % (pdir, inumber))
#
##          plot = yt.SlicePlot(ds, 'z', "HydrogenFraction", width = width, axes_unit=axes_unit, center=cp)
##          plot.annotate_grids()
##          plot.save("%s/HydrogenFraction_%04d.png" % (pdir, inumber))

#       slc = ds.slice(2, cp[2])
#       frb = slc.to_frb(width, center=cp, resolution=500)
#       outfile = ('%s/%04d_Hydrogen_number_density_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['Hydrogen_number_density'])
#       outfp.write(frb['Hydrogen_number_density'])
#       outfp.close()
#       outfile = ('%s/%04d_temperature_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['temperature'])
#       outfp.write(frb['temperature'])
#       outfp.close()
#       outfile = ('%s/%04d_y_HII_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['y_HII'])
#       outfp.write(frb['y_HII'])
#       outfp.close()
#       outfile = ('%s/%04d_y_H2I_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['y_H2I'])
#       outfp.write(frb['y_H2I'])
#       outfp.close()

        plot = yt.OffAxisSlicePlot(ds, axis1 , "Hydrogen_number_density", center=cp, width = width, axes_unit=axes_unit, north_vector= axis2)
        frb = plot.data_source.to_frb(width, 500)
        fields = [
            'Hydrogen_number_density'
          , 'temperature_corr'
          , 'y_HII'
          , 'y_H2I'
          , 'CarbonAbundance'
          , 'IronAbundanceToSolar'
                 ]
        for field in fields:
            outfile = ('%s/%04d_%s.dat' % (pdir, inumber, field))
            outfp = open(outfile, 'wb')
 #          print(frb[field])
            outfp.write(frb[field])
            outfp.close()


    if PROJ:
        fields = [
            'Hydrogen_number_density'
          , 'temperature_corr'
          , 'y_HII'
          , 'y_H2I'
          , 'CarbonAbundance'
          , 'IronAbundanceToSolar'
                 ]

        # EDGE-ON
        pdir = indir + '/Projection_z'
        if not os.path.exists(pdir):
            os.mkdir(pdir)

#       plot = yt.ProjectionPlot(ds, 'z', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
##      plot.annotate_text(posi_nHmax, 'x', coord_system='data')
#       if inumber >= inumber_rad and inumber < inumber_exp:
#           plot.annotate_text((0.05, 0.05), 't = %.2f Myr' % tPopIII, coord_system='axis')
#       if inumber >= inumber_col:
#           plot.annotate_text((0.05, 0.10), 'n$_{H,max}$ = %.2e cm$^{-3}$' % nHmax, coord_system='axis')
#       plot.save('%s/Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))

##      plot = yt.ProjectionPlot(ds, 'z', "Hydrogen_number_density", weight_field="Density", width = width, center=cp)
##      plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
###     plot.set_zlim('Hydrogen_number_density', 1e2, 1e7)
##      plot.annotate_particles(width = width, p_size=50.0, col='white', marker='x')
##      plot.save('%s/Hydrogen_number_density_Density_%04d_w.png' % (pdir, inumber))

#       plot = yt.ProjectionPlot(ds, 'z', "temperature", weight_field="Density", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='temperature', cmap="hot")
#       plot.save('%s/temperature_Density_%04d.png' % (pdir, inumber))

#       plot = yt.ProjectionPlot(ds, 'z', "Zmet", weight_field="Density", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='Zmet', cmap="magma")
#       plot.save('%s/Zmet_Density_%04d.png' % (pdir, inumber))

#       pdir = indir + '/Projection_x'
#       if not os.path.exists(pdir):
#           os.mkdir(pdir)
#       plot = yt.ProjectionPlot(ds, 'x', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
##      plot.annotate_text(posi_nHmax, 'x', coord_system='data')
#       if inumber >= inumber_rad and inumber < inumber_exp:
#           plot.annotate_text((0.05, 0.05), 't = %.2f Myr' % tPopIII, coord_system='axis')
#       if inumber >= inumber_col:
#           plot.annotate_text((0.05, 0.10), 'n$_{H,max}$ = %.2e cm$^{-3}$' % nHmax, coord_system='axis')
#       plot.save('%s/Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))

#       pdir = indir + '/Projection_y'
#       if not os.path.exists(pdir):
#           os.mkdir(pdir)
#       plot = yt.ProjectionPlot(ds, 'y', "Hydrogen_number_density", weight_field="Density", width = width, axes_unit=axes_unit, center=cp)
#       plot.set_cmap(field='Hydrogen_number_density', cmap="bds_highcontrast")
##      plot.annotate_text(posi_nHmax, 'x', coord_system='data')
#       if inumber >= inumber_rad and inumber < inumber_exp:
#           plot.annotate_text((0.05, 0.05), 't = %.2f Myr' % tPopIII, coord_system='axis')
#       if inumber >= inumber_col:
#           plot.annotate_text((0.05, 0.10), 'n$_{H,max}$ = %.2e cm$^{-3}$' % nHmax, coord_system='axis')
#       plot.save('%s/Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))



##      prj = ds.proj('density', 'z', weight_field='density')
##      frb = prj.to_frb(width, center=cp, resolution=500)
###     plot = yt.ProjectionPlot(ds, 'z' , "Hydrogen_number_density", center=cp, width = width, axes_unit=axes_unit, weight_field='density')
###     frb = plot.data_source.to_frb(width, 500)
#       outfile = ('%s/%04d_temperature_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['temperature'])
#       outfp.write(frb['temperature'])
#       outfp.close()
#       outfile = ('%s/%04d_y_HII_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['y_HII'])
#       outfp.write(frb['y_HII'])
#       outfp.close()
#       outfile = ('%s/%04d_y_H2I_density.dat' % (pdir, inumber))
#       outfp = open(outfile, 'wb')
##      print(frb['y_H2I'])
#       outfp.write(frb['y_H2I'])
#       outfp.close()
#       plot = yt.OffAxisProjectionPlot(ds, axis1 , "Hydrogen_number_density", center=cp, width = width, depth = width, axes_unit=axes_unit, weight_field='density', north_vector= axis2)
#       frb = plot.frb

#       width_d = ds.arr(width).in_units('code_length')
#       print(width)
        if inumber < inumber_col:
            width_d = ds.arr(2, 'kpc').in_units('code_length')
##      elif inumber == inumber_fin:
##          width_d = ds.arr(100, 'au').in_units('code_length')
        else:
            width_d = width.in_units('code_length')
##      print("width %23.15e" % width)

        for field in fields:
            if inumber < inumber_col:
                frb = yt.off_axis_projection(ds, cp, axis1, [width_d, width_d, 0.5*width_d], 500, field, 'density', north_vector= axis2)
           #    print(frb)
                outfile = ('%s/%04d_%s_density.dat' % (pdir, inumber, field))
                outfp = open(outfile, 'wb')
                outfp.write(np.ascontiguousarray(frb.T))
                outfp.close()
            else:
                frb = yt.off_axis_projection(ds, cp, axis1, width_d, 500, field, 'density', north_vector= axis2)
           #    print(frb)
                outfile = ('%s/%04d_%s_density.dat' % (pdir, inumber, field))
                outfp = open(outfile, 'wb')
                outfp.write(np.ascontiguousarray(frb.T))
                outfp.close()

        # FACE-ON
        if inumber >= inumber_col:
            pdir = indir + '/Projection_x'
            if not os.path.exists(pdir):
                os.mkdir(pdir)
            
            for field in fields:
                if inumber < inumber_col:
                    frb = yt.off_axis_projection(ds, cp, axis2, [width_d, width_d, 0.5*width_d], 500, field, 'density', north_vector=[-axis1[0],-axis1[1],-axis1[2]])
               #    print(frb)
                    outfile = ('%s/%04d_%s_density.dat' % (pdir, inumber, field))
                    outfp = open(outfile, 'wb')
                    outfp.write(np.ascontiguousarray(frb.T))
                    outfp.close()
                else:
                    frb = yt.off_axis_projection(ds, cp, axis2, width_d, 500, field, 'density', north_vector=-axis1)
               #    print(frb)
                    outfile = ('%s/%04d_%s_density.dat' % (pdir, inumber, field))
                    outfp = open(outfile, 'wb')
                    outfp.write(np.ascontiguousarray(frb.T))
                    outfp.close()
            
            # Toomre's Q value EDGE-ON/FACE-ON
            if nSN==50 and inumber==inumber_fin:
                cv = [ core.mean("velocity_x", weight="cell_mass")
                     , core.mean("velocity_y", weight="cell_mass")
                     , core.mean("velocity_z", weight="cell_mass") ]
                def _angular_velocity(field, data):
                    dx = data["x"]-cp[0]; du = data["velocity_x"] - cv[0]
                    dy = data["y"]-cp[1]; dv = data["velocity_y"] - cv[1]
                    dz = data["z"]-cp[2]; dw = data["velocity_z"] - cv[2]
                    dr2 = dx**2 + dy**2 + dz**2
                    Omega0 = (dy * dw - dz * dv) / dr2
                    Omega1 = (dz * du - dx * dw) / dr2
                    Omega2 = (dx * dv - dy * du) / dr2
                    return (Omega0**2 + Omega1**2 + Omega2**2)**0.5
                ds.add_field(("gas", "angular_velocity"), function=_angular_velocity, sampling_type="cell", units="1/s")
            
   #            plot = yt.ProjectionPlot(ds, 'z', "angular_velocity", weight_field="Density", width = (50, 'au'), axes_unit=axes_unit, center=cp)
   #            plot.set_cmap(field='angular_velocity', cmap="bds_highcontrast")
   #            plot.set_unit('angular_velocity', '1/s')
   #            plot.save('%s/Projection_z/angular_velocity_Density_%04d.png' % (indir, inumber))
   #         
   #            plot = yt.ProjectionPlot(ds, 'z', "sound_speed_corr", weight_field="Density", width = (50, 'au'), axes_unit=axes_unit, center=cp)
   #            plot.set_cmap(field='sound_speed_corr', cmap="hot")
   #            plot.set_unit('sound_speed_corr', 'km/s')
   #            plot.save('%s/Projection_z/sound_speed_corr_Density_%04d.png' % (indir, inumber))
   #         
   #            plot = yt.ProjectionPlot(ds, 'z', "density", weight_field=None, width = (50, 'au'), axes_unit=axes_unit, center=cp)
   #            plot.set_cmap(field='density', cmap="bds_highcontrast")
   #            plot.set_unit('density', 'g/cm**2')
   #            plot.save('%s/Projection_z/density_%04d.png' % (indir, inumber))
             
                for idir in range(2):
                    if idir==0: axis_perp = axis1; axis_north = axis2
                    if idir==1: axis_perp = axis2; axis_north =-axis1
             
                    frb_Omega = yt.off_axis_projection(ds, cp, axis_perp, width_d, 500, 'angular_velocity', 'density', north_vector= axis_north)
                    frb_cs    = yt.off_axis_projection(ds, cp, axis_perp, width_d, 500, 'sound_speed_corr', 'density', north_vector= axis_north)
                    frb_Sigma = yt.off_axis_projection(ds, cp, axis_perp, width_d, 500, 'density'         , None     , north_vector= axis_north)
             
                    frb = 2.0 * frb_Omega.in_units('1/s') * frb_cs.in_units('cm/s') / pi / G / frb_Sigma.in_units('g/cm**2')
             
                    if idir==0: pdir = indir + '/Projection_z'
                    if idir==1: pdir = indir + '/Projection_x'
                    outfile = ('%s/%04d_Q.dat' % (pdir, inumber))
                    outfp = open(outfile, 'wb')
                    outfp.write(np.ascontiguousarray(frb.T))
                    outfp.close()


    if PHASE:
        pdir = indir + '/Profile-2d'
        if not os.path.exists(pdir):
            os.mkdir(pdir)

#       phpl = yt.PhasePlot(ad, "Hydrogen_number_density", "adiabatic_index" , ["cell_mass"]  , weight_field=None)
#       phpl.set_unit('cell_mass', 'Msun')
#       phpl.set_log('adiabatic_index', False)
#       phpl.save("%s/Hydrogen_number_density_adiabatic_index_cell_mass_%04d.png" % (pdir, inumber))

#       phpl = yt.PhasePlot(ad, "Hydrogen_number_density", "molecular_weight" , ["cell_mass"]  , weight_field=None)
#       phpl.set_unit('cell_mass', 'Msun')
#       phpl.set_log('molecular_weight', False)
#       phpl.save("%s/Hydrogen_number_density_molecular_weight_cell_mass_%04d.png" % (pdir, inumber))

#       phpl = yt.PhasePlot(ad, "Hydrogen_number_density", "temperature_corr" , ["cell_mass"]  , weight_field=None)
#       phpl.set_unit('cell_mass', 'Msun')
#       phpl.save("%s/Hydrogen_number_density_temperature_corr_cell_mass_%04d.png" % (pdir, inumber))

#       phpl = yt.PhasePlot(ad, "Hydrogen_number_density", "sound_speed_corr" , ["cell_mass"]  , weight_field=None)
#       phpl.set_unit('cell_mass', 'Msun')
#       phpl.set_unit('sound_speed_corr', 'km/s')
#       phpl.save("%s/Hydrogen_number_density_sound_speed_corr_cell_mass_%04d.png" % (pdir, inumber))

#       phpl = yt.PhasePlot(sphere, "Hydrogen_number_density", "temperature" , ["cell_mass"]  , weight_field=None)
#       phpl.set_unit('cell_mass', 'Msun')
#       phpl.set_xlabel('Density (cm$^{-3}$)')
#       phpl.set_ylabel('Temperature (K)')
#       phpl.set_colorbar_label('cell_mass', 'Cell mass (M$_{\odot}$)')
#       phpl.set_zlim('cell_mass', 1e-7, 1e7)
#       phpl.annotate_title('(f) %.2f Myr from SN explosion (z = %.2f)' % (tPopIII, ds.current_redshift))
#       phpl.set_font({'size':32})
#       phpl.save("%s/Hydrogen_number_density_temperature_cell_mass_%04d.png" % (pdir, inumber))

#       if nDM == 1:
#           phpl = yt.PhasePlot(sphere_rem, "Hydrogen_number_density", "temperature" , ["cell_mass"]  , weight_field=None)
#           phpl.set_unit('cell_mass', 'Msun')
#           phpl.set_xlabel('Density (cm$^{-3}$)')
#           phpl.set_ylabel('Temperature (K)')
#           phpl.set_colorbar_label('cell_mass', 'Cell mass (M$_{\odot}$)')
#           phpl.set_zlim('cell_mass', 1e-7, 1e7)
#           phpl.annotate_title('(f) %.2f Myr from SN explosion (z = %.2f)' % (tPopIII, ds.current_redshift))
#           phpl.set_font({'size':32})
#           phpl.save("%s/Hydrogen_number_density_temperature_cell_mass_%04d_rem.png" % (pdir, inumber))


        if inumber >= inumber_col:
            region_phase = ds.sphere(cp, (1, 'kpc')).cut_region(["obj['Zmet'] > 1.0e-7"])

        #   plot = yt.create_profile(region_phase, ["radius", "Zmet"], "cell_mass", weight_field=None, n_bins=(128, 128))
        #   plot.save_as_dataset("%s/%04d_radius_Zmet_cell_mass.h5" % (pdir, inumber))

            plot = yt.create_profile(region_phase, ["radius", "CarbonAbundance"], "Hydrogen_number_density", weight_field=None, n_bins=(128, 128))
            plot.save_as_dataset("%s/%04d_radius_CarbonAbundance_Hydrogen_number_density.h5" % (pdir, inumber))

            plot = yt.create_profile(region_phase, ["radius", "IronAbundanceToSolar"], "Hydrogen_number_density", weight_field=None, n_bins=(128, 128))
            plot.save_as_dataset("%s/%04d_radius_IronAbundanceToSolar_Hydrogen_number_density.h5" % (pdir, inumber))
  
        #   plot = yt.PhasePlot(sphere, "radius", "CarbonAbundance", "Hydrogen_number_density", weight_field="cell_mass")
        #   plot.set_unit('radius', 'pc')
 #      #   plot.set_log('CarbonAbundance', False)
 #      #   plot.set_ylabel(r'$10^{A({\rm C})}$')
        #   plot.save("%s/radius_CarbonAbundance_Hydrogen_number_density_%04d.png" % (pdir, inumber))

        #   plot = yt.PhasePlot(sphere, "radius", "IronAbundanceToSolar", "Hydrogen_number_density", weight_field="cell_mass")
        #   plot.set_unit('radius', 'pc')
 #      #   plot.set_log('IronAbundanceToSolar', False)
 #      #   plot.set_ylabel(r'$10^{[Fe/H]}$')
        #   plot.save("%s/radius_IronAbundanceToSolar_Hydrogen_number_density_%04d.png" % (pdir, inumber))

##      #   plot = yt.PhasePlot(sphere, "Hydrogen_number_density", "CarbonCondensationEfficiency", "cell_mass", weight_field=None)
##      #   plot.save("%s/Hydrogen_number_density_CarbonCondensationEfficiency_cell_mass_%04d.png" % (pdir, inumber))


            

    if PROFILE:
        pdir = indir + '/Profile-1d'
        if not os.path.exists(pdir):
            os.mkdir(pdir)

        fn_prof = pdir + '/' + ('%04d' % inumber) + '_rv' + '.dat'
        fp_prof = open(fn_prof, mode='w')

        rmin = 2.0*ad.min("cell_size").in_units("pc")
        rmax = 1.0e2
        log_rmin = np.log10(rmin)
        log_rmax = np.log10(rmax)
        drad = 0.1
        nrad = int((log_rmax - log_rmin) / drad) + 1
        print("%13.7f %13.7f %5d" % (log_rmin, log_rmax, nrad))

        region_prof = ds.sphere(cp, (rmax, 'pc'))
        prof_nH = yt.create_profile(region_prof, 'radius', 'Hydrogen_number_density',
                                    units = {'radius': 'pc'}, n_bins=nrad,
                                    extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
                                    weight_field='cell_volume', accumulation=False)
        prof_Tg = yt.create_profile(region_prof, 'radius', 'temperature_corr',
                                    units = {'radius': 'pc'}, n_bins=nrad,
                                    extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
                                    weight_field='cell_mass', accumulation=False)
        prof_vr = yt.create_profile(region_prof, 'radius', 'radial_velocity',
                                    units = {'radius': 'pc'}, n_bins=nrad,
                                    extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
                                    weight_field='cell_mass', accumulation=False)
        prof_Mr = yt.create_profile(region_prof, 'radius', 'cell_mass',
                                    units = {'radius': 'pc'}, n_bins=nrad,
                                    extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
                                    weight_field=None, accumulation=True)
        prof_TE = yt.create_profile(region_prof, 'radius', 'ThermalEnergy',
                                    units = {'radius': 'pc'}, n_bins=nrad,
                                    extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
                                    weight_field=None, accumulation=True)
                       #Rhc = 0.5*(Rhc0+Rhc1)
                       #hcr = ds.sphere(hcc, Rhc)
                       #Mhc = hcr.quantities.total_quantity("cell_mass").in_units("Msun")
                       #Egr = 3.0/5.0*G*Mhc**2/Rhc
                       #Eth = hcr.quantities.total_quantity("ThermalEnergy").in_units("erg")

        for irad in range(nrad):
            fp_prof.write("%13.5e %13.5e %13.5e %13.5e %13.5e\n" % (
                  prof_nH.x[irad]
                , prof_nH['Hydrogen_number_density'][irad]
                , prof_Tg['temperature_corr'][irad]
                , prof_vr['radial_velocity'][irad]
                , prof_TE['ThermalEnergy'][irad] / (3.0/5.0 * G * prof_Mr['cell_mass'][irad]**2 / prof_Mr.x[irad])
                         ))

#       rad       = [0.0] * nrad
#       nH_avr    = [0.0] * nrad
#       Tg_avr    = [0.0] * nrad
#       vrad      = [0.0] * nrad
#       omega_x   = [0.0] * nrad
#       omega_y   = [0.0] * nrad
#       omega_z   = [0.0] * nrad
#       mach_turb = [0.0] * nrad

#       region_prof = ds.sphere(cp, (200., 'pc'))
#       cv = [ region_prof.mean("velocity_x", weight="cell_mass")
#            , region_prof.mean("velocity_y", weight="cell_mass")
#            , region_prof.mean("velocity_z", weight="cell_mass") ]

#       def _angular_velocity_x(field, data):
#           dx = data["x"]-cp[0]; du = data["velocity_x"]
#           dy = data["y"]-cp[1]; dv = data["velocity_y"]
#           dz = data["z"]-cp[2]; dw = data["velocity_z"]
#           dr2 = dx**2 + dy**2 + dz**2
#           return (dy * dw - dz * dv) / dr2
#       ds.add_field(("gas", "angular_velocity_x"), function=_angular_velocity_x, sampling_type="cell", units='1/s')
#       def _angular_velocity_y(field, data):
#           dx = data["x"]-cp[0]; du = data["velocity_x"]
#           dy = data["y"]-cp[1]; dv = data["velocity_y"]
#           dz = data["z"]-cp[2]; dw = data["velocity_z"]
#           dr2 = dx**2 + dy**2 + dz**2
#           return (dz * du - dx * dw) / dr2
#       ds.add_field(("gas", "angular_velocity_y"), function=_angular_velocity_y, sampling_type="cell", units='1/s')
#       def _angular_velocity_z(field, data):
#           dx = data["x"]-cp[0]; du = data["velocity_x"]
#           dy = data["y"]-cp[1]; dv = data["velocity_y"]
#           dz = data["z"]-cp[2]; dw = data["velocity_z"]
#           dr2 = dx**2 + dy**2 + dz**2
#           return (dx * dv - dy * du) / dr2
#       ds.add_field(("gas", "angular_velocity_z"), function=_angular_velocity_z, sampling_type="cell", units='1/s')

#       for irad in range(nrad):
#           if irad==0: rad0 = ds.arr(0.0, 'pc')
#           else:       rad0 = ds.arr(10.0**(log_rmin + (irad - 0.5) * drad), 'pc')
#           rad[irad]        = ds.arr(10.0**(log_rmin +  irad        * drad), 'pc')
#           rad1             = ds.arr(10.0**(log_rmin + (irad + 0.5) * drad), 'pc')
#           shell_prof  = ds.sphere(cp, rad1).cut_region([("obj['radius'] > %e" % rad0.in_units('cm'))])
#           # CHECK ######
#         # plot = yt.ProjectionPlot(ds, 'z', "Hydrogen_number_density", weight_field="Density", width=(1.0e-5, 'pc'), center=cp, data_source=shell_prof)
#         # plot.set_cmap(field='Hydrogen_number_density', cmap="cmap_dens")
#         # plot.save('%s/Profile-1d/prof%03d_Hydrogen_number_density_Density_%04d.png' % (indir, irad, inumber))
#           ##############
#           nH_avr[irad]  = shell_prof.mean("Hydrogen_number_density", weight="cell_volume")
#           Tg_avr[irad]  = shell_prof.mean("temperature_corr"       , weight="cell_mass")
#           vbulk_x       = shell_prof.mean("x-velocity"             , weight="cell_mass")
#           vbulk_y       = shell_prof.mean("y-velocity"             , weight="cell_mass")
#           vbulk_z       = shell_prof.mean("z-velocity"             , weight="cell_mass")
#           vrad[irad]    = shell_prof.mean("radial_velocity"        , weight="cell_mass")
#           omega_x[irad] = shell_prof.mean("angular_velocity_x"     , weight="cell_mass")
#           omega_y[irad] = shell_prof.mean("angular_velocity_y"     , weight="cell_mass")
#           omega_z[irad] = shell_prof.mean("angular_velocity_z"     , weight="cell_mass")
#           def _turbulent_velocity_square(field, data):
#               dx = data["x"]-cp[0]
#               dy = data["y"]-cp[1]
#               dz = data["z"]-cp[2]
#               dr = (dx**2 + dy**2 + dz**2)**0.5
#               vrad_vec = vrad[irad] / dr * [dx, dy, dz]
#               vrot_vec = [(dy * omega_z[irad] - dz * omega_y[irad])
#                         , (dz * omega_x[irad] - dx * omega_z[irad])
#                         , (dx * omega_y[irad] - dy * omega_x[irad])]
#               du = data["velocity_x"] - vbulk_x - vrad_vec[0] - vrot_vec[0]
#               dv = data["velocity_y"] - vbulk_y - vrad_vec[1] - vrot_vec[1]
#               dw = data["velocity_z"] - vbulk_z - vrad_vec[2] - vrot_vec[2]
#               return (du**2 + dv**2 + dw**2)
#           ds.add_field(("gas", "turbulent_velocity_square"), function=_turbulent_velocity_square, sampling_type="cell", units="(cm/s)**2", force_override=True)
#           vturb2 = shell_prof.mean("turbulent_velocity_square", weight="cell_mass")
#           cs2    = shell_prof.mean("sound_speed_corr"         , weight="cell_mass")**2
#           mach_turb[irad] = (vturb2 / cs2)**0.5
#         # print("%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.7f" % (
#         #       rad[irad]
#         #     , nH_avr[irad]
#         #     , Tg_avr[irad]
#         #     , vrad[irad]
#         #     , omega_x[irad]
#         #     , omega_y[irad]
#         #     , omega_z[irad]
#         #     , mach_turb[irad]
#         #            ) )

#       for irad in range(nrad):
#           fp_prof.write("%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.7f\n" % (
#                 rad[irad]
#               , nH_avr[irad]
#               , Tg_avr[irad]
#               , vrad[irad]
#               , omega_x[irad]
#               , omega_y[irad]
#               , omega_z[irad]
#               , mach_turb[irad]
#                      ) )
    ##  prof_v = yt.create_profile(region_prof, 'radius', 'velocity_magnitude',
    ##                             units = {'radius': 'pc'}, n_bins=nrad,
    ##                             extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
    ##                             weight_field='cell_mass', accumulation=True)
    ##  vturb = prof_v.standard_deviation['gas', 'velocity_magnitude']
    ##  tturb = 2.0 * prof_v.x / vturb
    ##  prof_o = yt.create_profile(region_prof, 'radius', 'vorticity_magnitude',
    ##                             units = {'radius': 'pc'}, n_bins=nrad,
    ##                             extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
    ##                             weight_field='cell_mass', accumulation=True)
    ##  ovor = prof_o['vorticity_magnitude']
    ##  tvor = 2.0 * pi / ovor
    ##  prof_c = yt.create_profile(region_prof, 'radius', 'velocity_divergence_absolute',
    ##                             units = {'radius': 'pc'}, n_bins=nrad,
    ##                             extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
    ##                             weight_field='cell_mass', accumulation=True)
    ##  tcol = 1.0 / prof_c['velocity_divergence_absolute']

    ##  for irad in range(nrad):
    ##      fp_prof.write("%13.5e %13.5e %13.5e %13.5e\n" % (
    ##            prof_v.x[irad]
    ##          , tturb[irad].in_units('yr')
    ##          ,  tvor[irad].in_units('yr')
    ##          ,  tcol[irad].in_units('yr')
    ##                   ))

        fp_prof.close()

###     prof = yt.create_profile(sphere, 'radius', 'mach_number',
###                              units = {'radius': 'pc'}, n_bins=nrad,
###                              extrema = {'radius': ((rmin, 'pc'), (rmax, 'pc'))},
###                              weight_field='cell_mass', accumulation=False)
###     log_radius      = np.log10(prof.x)
###     log_mach_number = np.log10(prof["mach_number"])
###     plt.plot(log_radius, log_mach_number, label=SN, color = 'blue')
###     plt.xlabel('log [ Radius / pc ]')
###     plt.ylabel('log [ Mach number ]')
###     plt.legend()
###     plt.savefig("%s/radius_mach_number_%04d.png" % (pdir, inumber))

#       prof_rho_gas = yt.create_profile(sphere, ["radius"], fields=[   "Hydrogen_number_density"], n_bins=128, units = {'radius': 'pc'})
#       prof_rho_dm  = yt.create_profile(sphere, ["radius"], fields=["dark_matter_number_density"], n_bins=128, units = {'radius': 'pc'})
#       log_radius = np.log10(prof_rho_gas.x)
#       log_Hydrogen_number_density    = np.log10(prof_rho_gas[   "Hydrogen_number_density"])
#       log_dark_matter_number_density = np.log10(prof_rho_dm ["dark_matter_number_density"])
#       plt.plot(log_radius, log_Hydrogen_number_density   , label='Gas', color = 'blue')
#       plt.plot(log_radius, log_dark_matter_number_density, label='DM' , color = 'black')
#       plt.xlabel('log [ Radius / pc ]')
#       plt.ylabel('log [ Density / cm$^{-3}$ ]')
#       plt.legend()
#       plt.savefig("%s/radius_density_%04d.png" % (pdir, inumber))

####    if inumber < inumber_col:
####        vbulk = [halo_sphere.mean("x-velocity", weight="cell_mass")
####               , halo_sphere.mean("y-velocity", weight="cell_mass")
####               , halo_sphere.mean("z-velocity", weight="cell_mass")]
####    else:
####        vbulk = [core.mean("x-velocity", weight="cell_mass")
####               , core.mean("y-velocity", weight="cell_mass")
####               , core.mean("z-velocity", weight="cell_mass")]

####    for idir in range(2):
####        print(idir)
####        fn_prof = pdir + '/' + ('%04d' % inumber) + 'rad_profile' + ('%1d' % idir) + '.dat'
####        fp_prof = open(fn_prof, mode='w')
####        print(cp)
####        pos0 = cp
####        pos  = cp
####        r=ds.arr(0.0, 'pc')
####        leng=ds.arr(1000.0, 'pc')
####        if idir==0: norm = axis0
####        if idir==1: norm = axis2
####        while r < leng:
####            dx = norm * r
####            pos = pos0 + dx
####            point = ds.point(pos)
####            point_nH   = point.max('Hydrogen_number_density')
####            point_Tg   = point.max('temperature_corr')
####            point_pr   = point.max('pressure_corr').in_units('erg/cm**3')
####            du = point.max('x-velocity') - vbulk[0]
####            dv = point.max('y-velocity') - vbulk[1]
####            dw = point.max('z-velocity') - vbulk[2]
####            vr = (dx[0]*du + dx[1]*dv + dx[2]*dw) / r
####            point_vr = vr.in_units('km/s')
####            point_yHII = point.max('y_HII')
####            point_AC   = point.max('CarbonAbundance')
####            fp_prof.write("%13.5e " % (r.in_units(axes_unit)))
####            fp_prof.write("%13.5e " % (point_nH   ))
####            fp_prof.write("%13.5e " % (point_Tg   ))
####            fp_prof.write("%13.5e " % (point_pr   ))
####            fp_prof.write("%13.5e " % (point_vr   ))
####            fp_prof.write("%13.5e " % (point_yHII ))
####            fp_prof.write("%13.5e " % (point_AC   ))
####            fp_prof.write("\n")
####            dr = point.min('cell_size')
####            r = r + dr/max(norm)
####            pos0 = pos
####        fp_prof.close()


    if CLUMP:
        RUN_CLUMP_FIND = False

######  fn_clump = indir + '/clump_c20_M3e-4_n1e13.h5'
        fn_clump = indir + '/clump_c20_M1e-3_n1e13.h5'
######  fn_clump = indir + '/clump_c20_M3e-3_n1e13.h5'
        sphere_find = ds.sphere(ad.argmax("density"), (100.0, 'au'))

        if RUN_CLUMP_FIND:
            def _minimum_gas_mass(clump, min_mass):
                return (clump["gas", "cell_mass"].sum() >= min_mass)
            add_validator("minimum_gas_mass", _minimum_gas_mass)
            def _minimum_gas_density(clump, min_density):
                return (clump["gas", "Hydrogen_number_density"].min() >= min_density)
            add_validator("minimum_gas_density", _minimum_gas_density)

######      Mcl_L = ds.quan(3.0e-4, "Msun")
            Mcl_L = ds.quan(1.0e-3, "Msun")
######      Mcl_L = ds.quan(3.0e-3, "Msun")
            ncl_L = ds.quan(1.0e13, "1/cm**3")

            data_source = sphere_find
            # the field to be used for contouring
            field = ("gas", "Hydrogen_number_density")
            # This is the multiplicative interval between contours.
            step = 2.0
            # Now we set some sane min/max values between which we want to find contours.
            # This is how we tell the clump finder what to look for -- it won't look for
            # contours connected below or above these threshold values.
            c_min = 10**np.floor(np.log10(data_source[field]).min()  )
            c_max = 10**np.floor(np.log10(data_source[field]).max()+1)
            # Now find get our 'base' clump -- this one just covers the whole domain.
            master_clump = Clump(data_source, field)
            # Add a "validator" to weed out clumps with less than 20 cells.
            # As many validators can be added as you want.
            master_clump.add_validator("min_cells", 20)
            master_clump.add_validator("minimum_gas_mass", Mcl_L)
            master_clump.add_validator("minimum_gas_density", ncl_L)
        ### master_clump.add_validator("gravitationally_bound", use_particles=False)
            # Calculate center of mass for all clumps.
            master_clump.add_info_item("center_of_mass")
            # Begin clump finding.
            find_clumps(master_clump, c_min, c_max, step)
            # Save the clump tree as a reloadable dataset
            fp_clump = master_clump.save_as_dataset(filename=fn_clump, fields=["Hydrogen_number_density", "particle_mass"])
            # We can traverse the clump hierarchy to get a list of all of the 'leaf' clumps
            leaf_clumps = get_lowest_clumps(master_clump)
        else:
            ds_clumps = yt.load(fn_clump)
         #  print('clump file %s' % fn_clump)
         #  for clump in ds_clumps.tree:
         #      print('clump id %d' % clump.clump_id)
            leaf_clumps = ds_clumps.leaves
            print("# of clumps %d" % len(leaf_clumps))
            mass_cl = []
            for my_clump in leaf_clumps:
                mass_cl.append(my_clump["clump", "cell_mass"])
            ind_cl = np.flip(np.argsort(mass_cl), axis=0)

            for i_clump in ind_cl:
                my_clump = leaf_clumps[i_clump]
             #  print(my_clump)
             #  print("cell_mass %13.5e Msun" % my_clump["clump", "cell_mass"])
             #  print("density %13.5e cm-3" % my_clump["grid", "Hydrogen_number_density"].min())
           ###  cl_cen = my_clump["clump", "center_of_mass"]
                cl_icen = my_clump["grid", "Hydrogen_number_density"].argmax()
                cl_cen = [my_clump["grid", "x"][cl_icen]
                        , my_clump["grid", "y"][cl_icen]
                        , my_clump["grid", "z"][cl_icen]]
             #  print("(%25.13f, %25.13f, %25.13f)" % (cl_cen[0], cl_cen[1], cl_cen[2]))
                dx0 = [ (cl_cen[0]-cp[0]).in_units('au')
                      , (cl_cen[1]-cp[1]).in_units('au')
                      , (cl_cen[2]-cp[2]).in_units('au') ]
             #  print("center [%13.7f, %13.7f, %13.7f] au" % (dx0[0], dx0[1], dx0[2]))
             #  if n_clump==0: dx0 = -ds.arr(10.0, 'au') * axis0
             #  if n_clump==1: dx0 = -ds.arr(10.0, 'au') * axis1
             #  if n_clump==2: dx0 =  ds.arr(10.0, 'au') * axis2
                dr0 = (dx0[0]*dx0[0] + dx0[1]*dx0[1] + dx0[2]*dx0[2])**0.5
                rot = np.array([-axis0, -axis1, axis2])
                dx1 = np.dot(rot, dx0)
                print("cen_ps.append([%13.7f, %13.7f, %13.7f]) # %13.7f %13.5e" % 
                        (dx1[0], dx1[1], dx1[2]
                       , my_clump["clump", "cell_mass"]
                       , my_clump["grid", "Hydrogen_number_density"].min() ))

            for i_clump in ind_cl:
                my_clump = leaf_clumps[i_clump]
                ncen_cl = my_clump["grid", "Hydrogen_number_density"].min()
                mass_cl = my_clump["clump", "cell_mass"]
                if ncen_cl > ds.quan(1.0e15, '1/cm**3') and mass_cl > ds.quan(1.0e-2, 'Msun'):
                    icen = my_clump["grid", "Hydrogen_number_density"].argmax()
                    hcc = [my_clump["grid", "x"][icen]
                         , my_clump["grid", "y"][icen]
                         , my_clump["grid", "z"][icen]]
                    Rhc0 = ds.arr(0.0, 'au')
                    Rhc1 = ds.arr(100.0, 'au')
                    for itr in range(64):
                        Rhc = 0.5*(Rhc0+Rhc1)
                        hcr = ds.sphere(hcc, Rhc)
                        Mhc = hcr.quantities.total_quantity("cell_mass").in_units("Msun")
                        Egr = 3.0/5.0*G*Mhc**2/Rhc
                        Eth = hcr.quantities.total_quantity("ThermalEnergy").in_units("erg")
                #       Lmin = hcr.min("grid_level")
                #       Dmin = hcr.min("Hydrogen_number_density")
                #       print ("%20.10f %20.10f" % (Rhs, Egr/Eth))
                        if Egr < Eth:
                            Rhc0 = Rhc
                        else:
                            Rhc1 = Rhc
                #   Eratio = Egr/Eth
                    print(" %13.7f %13.5e" % (Rhc.in_units("au"), Mhc.in_units("Msun")))
                #   print(my_clump)
                #   print(hcc)
                #   print("radius %13.7f" % Rhc.in_units("au"))
                #   print("mass   %13.7f" % Mhc.in_units("Msun"))
    
            width = (50, 'au')
            pdir = indir + '/Projection_x'
            if not os.path.exists(pdir):
                os.mkdir(pdir)
            plot = yt.ProjectionPlot(ds, 'x', "Hydrogen_number_density", weight_field="Density", width=width, center=cp)
            plot.set_cmap(field='Hydrogen_number_density', cmap="cmap_dens")
        #   plot.annotate_sphere(cp, radius=(Rhc, 'au'), circle_args={'color':'black'})
            plot.annotate_clumps(leaf_clumps)
        #   for my_clump in leaf_clumps:
        #       icen = my_clump["grid", "Hydrogen_number_density"].argmax()
        #       cen = [my_clump["grid", "x"][icen]
        #            , my_clump["grid", "y"][icen]
        #            , my_clump["grid", "z"][icen]]
        #       plot.annotate_marker(cen, coord_system='data')
            plot.save('%s/clumpc_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))

           #axis2 = np.array([1.0, 0.0, 0.0]); axis1 = np.array([0.0, 0.0, -1.0])
            plot = yt.OffAxisSlicePlot(ds, axis2 , "Hydrogen_number_density", center=cp, width = width, axes_unit=axes_unit, north_vector=-axis1)
            plot.set_cmap(field='Hydrogen_number_density', cmap="cmap_dens")
            for my_clump in leaf_clumps:
                icen = my_clump["grid", "Hydrogen_number_density"].argmax()
                cen = [my_clump["grid", "x"][icen]
                     , my_clump["grid", "y"][icen]
                     , my_clump["grid", "z"][icen]]
                plot.annotate_marker(cen, coord_system='data')
            plot.save('%s/clumpd_Hydrogen_number_density_Density_%04d.png' % (pdir, inumber))

# CREATE FIGURES ############################################################################################## 
if TRACE:
    fp_trace.close()
