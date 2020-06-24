import yt
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from yt.data_objects.particle_filters import add_particle_filter
#from yt.analysis_modules.halo_finding.api import HaloFinder
from yt.utilities.physical_constants import \
    gravitational_constant_cgs as G
from yt.units import mp
import numpy as np
import struct
import os
import sys
from yt.visualization.api import get_multi_plot
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
pi = 3.14159265

ANALYTIC        = False
DUST_PROPERTIES = False
THERMAL_HISTORY = True
MASS_HISTORY    = False
C_FE            = False
SNAPSHOTS_INI   = False
SNAPSHOTS_RAD   = False # UNUSED
SNAPSHOTS_EXP   = False
SNAPSHOTS_COL   = False
SNAPSHOTS_FIN   = False
PROFILE_METAL   = False
PROFILE_COL     = False

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"
fontsize_suptitle = 28
fontsize_title    = 24
fontsize_boxsize  = 24
fontsize_label    = 24
fontsize_cblabel  = 24
fontsize_tick     = 24
fontsize_label_s  = 16
fontsize_cblabel_s= 16
fontsize_tick_s   = 16
fontsize_legend   = 16
fontsize_legend_s = 12


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

cm.register_cmap(name='cmap_temp',
   data={'red':   [[0.0     ,  0.0, 0.0],
                   [0.333333,  1.0, 1.0],
                   [0.666667,  1.0, 1.0],
                   [1.0     ,  1.0, 1.0]],
         'green': [[0.0     ,  0.0, 0.0],
                   [0.333333,  0.0, 0.0],
                   [0.666667,  1.0, 1.0],
                   [1.0     ,  1.0, 1.0]],
         'blue':  [[0.0     ,  0.0, 0.0],
                   [0.333333,  0.0, 0.0],
                   [0.666667,  0.0, 0.0],
                   [1.0     ,  1.0, 1.0]]})


outdir = 'fig'

HydrogenFractionByMass   = 0.76
DeuteriumToHydrogenRatio = 3.4e-5 * 2.0
HeliumToHydrogenRatio    = (1.0 - HydrogenFractionByMass) / HydrogenFractionByMass
SolarMetalFractionByMass = 0.01295
SolarIronAbundance = 7.50

progmass = [
   r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
 , r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
 , r"$M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
  ]
Fe_H = [
     -9.24989440533554    
   , -7.937489132198783 
   , -8.279791819037703 
      ]
A_C = [
     3.7951320313852737
   , 5.058492609804371
   , 4.904347239142647
      ]


if ANALYTIC:
   mu_C  = 12.0
   mu_Fe = 56.0
   # C13 ######################
#  Mmet = 1.0 # Msun
#  M_C  = 0.3 # Msun
#  M_Fe = 0.1 # Msun
#  N_cl = 3
   Mmet = 1.5 # Msun
   M_C  = 0.5 # Msun
   M_Fe = 0.2 # Msun
   N_cl = 1
   R_cl = 6   # pc
   D_cl = 50  # pc
   M_cloud = 2000.0 # Msun
   f_ret = N_cl * (pi*R_cl**2) / (4*pi*D_cl**2)
   Zmet_pred = f_ret * Mmet / M_cloud / SolarMetalFractionByMass
   A_C_pred  = 12.0 + np.log10(f_ret * M_C / mu_C / HydrogenFractionByMass / M_cloud)
   Fe_H_pred = 12.0 + np.log10(f_ret * M_Fe/ mu_Fe/ HydrogenFractionByMass / M_cloud) - SolarIronAbundance
   print("C13 Mmet %11.5f M_C %11.5f M_Fe %11.5f R_cl %11.5f D_cl %11.5f f_ret %11.5f M_cloud %11.1f A_C %7.1f Fe_H %7.1f Zmet %7.1e"
       % (Mmet, M_C, M_Fe, R_cl, D_cl, f_ret, M_cloud, A_C_pred, Fe_H_pred, Zmet_pred) )

   # F13 ######################
   M_C  = 0.08   # Msun
   M_Fe = 1.0e-6 # Msun
   N_cl = 1
   R_cl = 6   # pc
   D_cl = 50  # pc
   M_cloud = 4000.0 # Msun
   f_ret = N_cl * (pi*R_cl**2) / (4*pi*D_cl**2)
   Zmet_pred = f_ret * Mmet / M_cloud / SolarMetalFractionByMass
   A_C_pred  = 12.0 + np.log10(f_ret * M_C / mu_C / HydrogenFractionByMass / M_cloud)
   Fe_H_pred = 12.0 + np.log10(f_ret * M_Fe/ mu_Fe/ HydrogenFractionByMass / M_cloud) - SolarIronAbundance
   print("F13 Mmet %11.5f M_C %11.5f M_Fe %11.3e R_cl %11.5f D_cl %11.5f f_ret %11.5f M_cloud %11.1f A_C %7.1f Fe_H %7.1f Zmet %7.1e"
       % (Mmet, M_C, M_Fe, R_cl, D_cl, f_ret, M_cloud, A_C_pred, Fe_H_pred, Zmet_pred) )

   # F80 ######################
   M_C  = 1.0    # Msun
   M_Fe = 1.0e-5 # Msun
   N_cl = 1
   R_cl = 7    # pc
   D_cl = 100  # pc
   M_cloud = 1000.0 # Msun
   f_ret = N_cl * (pi*R_cl**2) / (4*pi*D_cl**2)
   Zmet_pred = f_ret * Mmet / M_cloud / SolarMetalFractionByMass
   A_C_pred  = 12.0 + np.log10(f_ret * M_C / mu_C / HydrogenFractionByMass / M_cloud)
   Fe_H_pred = 12.0 + np.log10(f_ret * M_Fe/ mu_Fe/ HydrogenFractionByMass / M_cloud) - SolarIronAbundance
   print("F80 Mmet %11.5f M_C %11.5f M_Fe %11.3e R_cl %11.5f D_cl %11.5f f_ret %11.5f M_cloud %11.1f A_C %7.1f Fe_H %7.1f Zmet %7.1e"
       % (Mmet, M_C, M_Fe, R_cl, D_cl, f_ret, M_cloud, A_C_pred, Fe_H_pred, Zmet_pred) )

if DUST_PROPERTIES:

    fn_abund = [
        '/home/genchiaki/grackle_md/input/metaldustPopIII_M14_mix_M13_rev2.dat'
      , '/home/genchiaki/grackle_md/input/metaldustPopIII_M14_mix_M50_rev2.dat'
      , '/home/genchiaki/grackle_md/input/metaldustPopIII_M14_mix_M80_rev2.dat'
         ]
    fn_dist = [
        '/home/genchiaki/dust/ic/data/dist_Km_13_rev2.dat'
      , '/home/genchiaki/dust/ic/data/dist_Km_50_rev2.dat'
      , '/home/genchiaki/dust/ic/data/dist_Km_80_rev2.dat'
         ]
    data_dist = [None] * 3
    i_rad    =  0
    i_SiM    =  1
    i_FeM    =  2
    i_Mg2SiO4=  3
    i_MgSiO3 =  4
    i_Fe3O4  =  5
    i_AC     =  6
    i_SiO2D  =  7
    i_MgO    =  8
    i_FeS    =  9
    i_Al2O3  = 10

    r0_dust = [0.0]*51
    dr0_dust = [0.0]*51
    for irad in range(51):
        r0_dust[irad]  = 10.0**(-8.0+irad*0.1)
        dr0_dust[irad] = 10.0**(-7.95+irad*0.1) - 10.0**(-8.05+irad*0.1)

    s_AC = 2.23 # g/cc
    n_AC = [0.0]*3
    delta_r = [0.0]*3

    for iSN in range(3):
        # FROM INITIAL DATA
        fp = open(fn_abund[iSN], 'rb')
        Mfrac_C  = struct.unpack('d', fp.read(8))[0]
        Mfrac_O  = struct.unpack('d', fp.read(8))[0]
        Mfrac_Mg = struct.unpack('d', fp.read(8))[0]
        Mfrac_Al = struct.unpack('d', fp.read(8))[0]
        Mfrac_Si = struct.unpack('d', fp.read(8))[0]
        Mfrac_S  = struct.unpack('d', fp.read(8))[0]
        Mfrac_Fe = struct.unpack('d', fp.read(8))[0]
        print("%13.5e %13.5e %13.5e " % (Mfrac_C, Mfrac_O, Mfrac_Fe))
        fcondC   = struct.unpack('d', fp.read(8))[0]
        fcondO   = struct.unpack('d', fp.read(8))[0]
        fcondMg  = struct.unpack('d', fp.read(8))[0]
        fcondAl  = struct.unpack('d', fp.read(8))[0]
        fcondSi  = struct.unpack('d', fp.read(8))[0]
        fcondS   = struct.unpack('d', fp.read(8))[0]
        fcondFe  = struct.unpack('d', fp.read(8))[0]
#       print("fcond_ini %13.5e %13.5e %13.5e %13.5e " % (fcondC, fcondO, fcondMg, fcondAl))
        fp.close()

        C_H = A_C[iSN] - 8.43
        A_O = A_C[iSN] - np.log10(Mfrac_C / 12.0) + np.log10(Mfrac_O / 16.0)
        O_H = A_O      - 8.69
        Dtrans = np.log10(10.0**C_H + 0.9 * 10.0**O_H)
        print("Dtrans %13.7f   C_H %13.7f O_H %13.7f" % (Dtrans, C_H, O_H))

        data_dist[iSN] = np.loadtxt(fn_dist[iSN])
        sum_r1df0 = 0.0
        sum_r2df0 = 0.0
        sum_r3df0 = 0.0
        for irad in range(51):
            sum_r1df0 += data_dist[iSN][irad, i_AC] * dr0_dust[irad] * r0_dust[irad]
            sum_r2df0 += data_dist[iSN][irad, i_AC] * dr0_dust[irad] * r0_dust[irad] * r0_dust[irad]
            sum_r3df0 += data_dist[iSN][irad, i_AC] * dr0_dust[irad] * r0_dust[irad] * r0_dust[irad] * r0_dust[irad]
        rho_AC = fcondC * 10.0**(A_C[iSN]-12.0) * 12.0 * mp # with nH = 1 /cc
        n_AC[iSN] = rho_AC / (4.0*pi/3.0 * s_AC * sum_r3df0)
#       print("%13.5e %13.5e %13.5e" % (10.0**(A_C[iSN]-12.0), rho_AC, n_AC[iSN]))

        # FROM SIMULATION
        if iSN==0: fcondC1 = 6.28099e-01
        if iSN==1: fcondC1 = 4.32772e-04
        if iSN==2: fcondC1 = 3.68941e-02
        print("fcondC %13.5e -> %13.5e " % (fcondC, fcondC1))
        roots = np.roots([1.0, 3.0*sum_r1df0, 3.0*sum_r2df0, (1.0 - fcondC1 / fcondC)*sum_r3df0])
        for iroot in range(3):
            if roots[iroot].imag == 0.0:
              delta_r[iSN] = roots[iroot].real
#             print(delta_r[iSN])
        rho_AC1 = fcondC1 * 10.0**(A_C[iSN]-12.0) * 12.0 * mp
        # Dust density when all C is condensed ###################
        fcondC2 = 1.0
        rho_AC2 = fcondC2 * 10.0**(A_C[iSN]-12.0) * 12.0 * mp
        log_yAC0 = np.log10(rho_AC /(12.0*mp))
        log_yAC1 = np.log10(rho_AC1/(12.0*mp))
        log_yAC2 = np.log10(rho_AC2/(12.0*mp))
        ##########################################################
        print("D(AC) %13.5e -> %13.5e" % (rho_AC/(mp/0.76), rho_AC1/(mp/0.76)))
        print("y(AC) %13.7f -> %13.7f (-> %13.7f)" % (log_yAC0, log_yAC1, log_yAC2))
        sum_r2df1 = sum_r2df0 + 2.0*sum_r1df0*delta_r[iSN] +               delta_r[iSN]**2 
        sum_r3df1 = sum_r3df0 + 3.0*sum_r2df0*delta_r[iSN] + 3.0*sum_r1df0*delta_r[iSN]**2 + delta_r[iSN]**3
        n_AC1 = rho_AC1 / (4.0*pi/3.0 * s_AC * sum_r3df1)
        print("n(AC) %13.5e -> %13.5e" % (n_AC[iSN], n_AC1))
        print("r(AC) %13.5e -> %13.5e" % (sum_r3df0/sum_r2df0*1.0e4, sum_r3df1/sum_r2df1*1.0e4))

    # PLOT
    fig, ax = plt.subplots(figsize=(6.4,4.8), constrained_layout=True)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel(r"log [ Grain radius / ${\rm \mu m}$ ]", fontsize=fontsize_label)
    ax.set_ylabel(r"log [ Number fraction / ${\rm \mu m ^{-1}}$ ]"    , fontsize=fontsize_label)
    ax.set_xlim([-3, 0])
    ax.set_ylim([-21, -13])
    ax.set_xticks(np.linspace(-3, 0, 4))
    ax.set_xticks(np.linspace(-3, 0, 16), minor=True)
    ax.set_yticks(np.linspace(-20,-14, 4))
    ax.set_yticks(np.linspace(-21,-13, 9), minor=True)
    ax.annotate(r'Amorphous carbon', xy=( 0.02, 0.98), xycoords=ax.transAxes, color='black', fontsize=fontsize_title, va='top' , ha='left')
    ax.annotate(r'before grain growth', xy=(-1.1, -14.0), xycoords='data', color='black', fontsize=fontsize_legend, va='center' , ha='left')
    ax.annotate(r'after grain growth' , xy=(-1.1, -14.5), xycoords='data', color='black', fontsize=fontsize_legend, va='center' , ha='left')
    plot = ax.plot(
        np.log10((data_dist[0][:, i_rad] + delta_r[0])*1.0e4), np.log10(n_AC[0] * data_dist[0][:, i_AC]/1.0e4)
      , np.log10((data_dist[1][:, i_rad] + delta_r[1])*1.0e4), np.log10(n_AC[1] * data_dist[1][:, i_AC]/1.0e4)
      , np.log10((data_dist[2][:, i_rad] + delta_r[2])*1.0e4), np.log10(n_AC[2] * data_dist[2][:, i_AC]/1.0e4)
      , np.log10((data_dist[0][:, i_rad]             )*1.0e4), np.log10(n_AC[0] * data_dist[0][:, i_AC]/1.0e4)
      , np.log10((data_dist[1][:, i_rad]             )*1.0e4), np.log10(n_AC[1] * data_dist[1][:, i_AC]/1.0e4)
      , np.log10((data_dist[2][:, i_rad]             )*1.0e4), np.log10(n_AC[2] * data_dist[2][:, i_AC]/1.0e4)
      , [-1.5,-1.2], [-14.0, -14.0] # legend
      , [-1.5,-1.2], [-14.5, -14.5] # legend
                  )
    plot[0].set_linestyle("-");   plot[0].set_color("orange");  plot[0].set_linewidth(2)
    plot[1].set_linestyle("-");   plot[1].set_color("purple");  plot[1].set_linewidth(2)
    plot[2].set_linestyle("-");   plot[2].set_color("green") ;  plot[2].set_linewidth(2)
    plot[3].set_linestyle("--");  plot[3].set_color("orange");  plot[3].set_linewidth(2)
    plot[4].set_linestyle("--");  plot[4].set_color("purple");  plot[4].set_linewidth(2)
    plot[5].set_linestyle("--");  plot[5].set_color("green") ;  plot[5].set_linewidth(2)
    plot[6].set_linestyle("--");  plot[6].set_color("black") ;  plot[6].set_linewidth(2) # legend
    plot[7].set_linestyle("-");   plot[7].set_color("black") ;  plot[7].set_linewidth(2) # legend
    plt.legend(labelspacing=0.0, loc='lower right')
    ax.legend(progmass, ncol=1, fontsize=fontsize_legend)
    fig.savefig("%s/dist.pdf" % outdir)

 
if THERMAL_HISTORY:
    i_time   =  0
    i_nH     =  1
    i_Tg     =  2
    i_elec   =  3
    i_HI     =  4
    i_HII    =  5
    i_HeI    =  6
    i_HeII   =  7
    i_HeIII  =  8
    i_HM     =  9
    i_H2I    = 10
    i_H2II   = 11
    i_DI     = 12
    i_DII    = 13
    i_HDI    = 14
    i_HeHII  = 15
    i_HDII   = 16
    i_DM     = 17
    i_CI     = 18
    i_CII    = 19
    i_CO     = 20
    i_CO2    = 21
    i_OI     = 22
    i_OH     = 23
    i_H2O    = 24
    i_O2     = 25
    i_SiI    = 26
    i_SiOI   = 27
    i_SiO2I  = 28
    i_CH     = 29
    i_CH2    = 30
    i_COII   = 31
    i_OII    = 32
    i_OHII   = 33
    i_H2OII  = 34
    i_H3OII  = 35
    i_O2II   = 36
    i_Mg     = 37
    i_Al     = 38
    i_S      = 39
    i_Fe     = 40
    i_SiM    = 41
    i_FeM    = 42
    i_Mg2SiO4= 43
    i_MgSiO3 = 44
    i_Fe3O4  = 45
    i_AC     = 46
    i_SiO2D  = 47
    i_MgO    = 48
    i_FeS    = 49
    i_Al2O3  = 50
    i_edot_adia = 51
    i_sigv      = 52
    i_cs        = 53
    i_axis0  = 54
    i_axis1  = 55
    i_axis2  = 56
    i_edot_comp    = 57
    i_edot_brem    = 58
    i_edot_ion_H   = 59
    i_edot_rec_H   = 60
    i_edot_line_H  = 61
    i_edot_ion_He  = 62
    i_edot_rec_He  = 63
    i_edot_line_He = 64
    i_edot_CIline  = 65
    i_edot_CIIline = 66
    i_edot_OIline  = 67
    i_edot_H2line  = 68
    i_edot_H2cont  = 69
    i_edot_H2form  = 70
    i_edot_HDline  = 71
    i_edot_H2Oline = 72
    i_edot_OHline  = 73
    i_edot_COline  = 74
    i_edot_dst_rad = 75
    i_tau_cont     = 76

    progmass = [
       r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
      ]

    trace3D = [None] * 3

    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M13"
    pdir = indir + '/time_series'
    fn_traceM13 = pdir + '/time_series_cool.dat'
    trace3D[0] = np.loadtxt(fn_traceM13)


    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M50"
    pdir = indir + '/time_series'
    fn_traceM50 = pdir + '/time_series_cool.dat'
    trace3D[1] = np.loadtxt(fn_traceM50)


    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M80"
    pdir = indir + '/time_series'
    fn_traceM80 = pdir + '/time_series_cool.dat'
    trace3D[2] = np.loadtxt(fn_traceM80)


    # Jeans mass
    nH_J   = [-1, 16]
    M_J_p6 = [0.0]*2
    M_J_p4 = [0.0]*2
    M_J_p2 = [0.0]*2
    M_J_p0 = [0.0]*2
    M_J_m2 = [0.0]*2
    M_J_m4 = [0.0]*2

    for i in range(2):
        M_J_p6[i] = np.log10( 200.0*(1.0e+6/(1.69e3*(1.23/1.23)**(-1.5)*(10.0**nH_J[i]/1.0e4)**(-0.5)))**(2.0/3.0) )
        M_J_p4[i] = np.log10( 200.0*(1.0e+4/(1.69e3*(1.23/1.23)**(-1.5)*(10.0**nH_J[i]/1.0e4)**(-0.5)))**(2.0/3.0) )
        M_J_p2[i] = np.log10( 200.0*(1.0e+2/(1.69e3*(1.23/1.23)**(-1.5)*(10.0**nH_J[i]/1.0e4)**(-0.5)))**(2.0/3.0) )
        M_J_p0[i] = np.log10( 200.0*(1.0e+0/(1.69e3*(2.31/1.23)**(-1.5)*(10.0**nH_J[i]/1.0e4)**(-0.5)))**(2.0/3.0) )
        M_J_m2[i] = np.log10( 200.0*(1.0e-2/(1.69e3*(2.31/1.23)**(-1.5)*(10.0**nH_J[i]/1.0e4)**(-0.5)))**(2.0/3.0) )
        M_J_m4[i] = np.log10( 200.0*(1.0e-4/(1.69e3*(2.31/1.23)**(-1.5)*(10.0**nH_J[i]/1.0e4)**(-0.5)))**(2.0/3.0) )

    # Density-Temperature/gamma/ellipticity
#   nvar = 1
    nvar = 3
    if nvar==1: ysize = 4.8
    if nvar==3: ysize =12.0
    fig, axs = plt.subplots(nrows=nvar, ncols=1, figsize=(6.4,ysize), constrained_layout=True)
    for ivar in range(nvar):
        if nvar==1: ax = axs
        else:       ax = axs[ivar]
        ax.tick_params(labelsize=fontsize_tick)
        ax.set_xlim([0, 16])
        if ivar==nvar-1: ax.set_xlabel(r"log [ Density / cm$^{-3}$ ]", fontsize=fontsize_label)
        if nvar==1:
            ax.set_ylabel(r"log [ Temperature / K ]", fontsize=fontsize_label)
        else:
            if ivar==0: ax.set_ylabel(r"log [ T / K ]" , fontsize=fontsize_label)
            if ivar==1: ax.set_ylabel(r"$\gamma$"      , fontsize=fontsize_label)
            if ivar==2: ax.set_ylabel(r"log [ e ]"     , fontsize=fontsize_label)
        if ivar==0: ax.set_ylim([ 1.0, 3.5])
        if ivar==1: ax.set_ylim([ 0.5, 1.5])
        if ivar==2: ax.set_ylim([-0.5, 1.1])
        if ivar<nvar-1: ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
        ax.set_xticks(np.linspace( 0, 16, 5))
        ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
        if nvar==1:
            ax.set_yticks(np.linspace( 1.0, 3.0, 3))
            ax.set_yticks(np.linspace( 1.0, 3.5, 7), minor=True)
        if ivar==0: 
            xvar0 = np.log10(trace3D[0][:, i_nH])
            yvar0 = np.log10(trace3D[0][:, i_Tg])
            xvar1 = np.log10(trace3D[1][:, i_nH])
            yvar1 = np.log10(trace3D[1][:, i_Tg])
            xvar2 = np.log10(trace3D[2][:, i_nH])
            yvar2 = np.log10(trace3D[2][:, i_Tg])
        if ivar==1:
            for iSN in range(3):
                nnc = trace3D[iSN][:, i_time].size
                xvar = [0.0]*(nnc-1); yvar = [0.0]*(nnc-1)
                for inc in range(nnc-1):
                   log_x0 = np.log10(trace3D[iSN][inc  , i_nH]); log_y0 = np.log10(trace3D[iSN][inc  , i_Tg])
                   log_x1 = np.log10(trace3D[iSN][inc+1, i_nH]); log_y1 = np.log10(trace3D[iSN][inc+1, i_Tg])
                   xvar[inc] = (log_x1 + log_x0) / 2.0
                   yvar[inc] = (log_y1 - log_y0) / (log_x1 - log_x0) + 1.0
                if iSN==0: xvar0 = xvar; yvar0 = yvar
                if iSN==1: xvar1 = xvar; yvar1 = yvar
                if iSN==2: xvar2 = xvar; yvar2 = yvar
        if ivar==2: 
            xvar0 = np.log10(trace3D[0][:, i_nH])
            yvar0 = np.log10(trace3D[0][:, i_axis0] / trace3D[0][:, i_axis2] - 1.0)
            xvar1 = np.log10(trace3D[1][:, i_nH])
            yvar1 = np.log10(trace3D[1][:, i_axis0] / trace3D[1][:, i_axis2] - 1.0)
            xvar2 = np.log10(trace3D[2][:, i_nH])
            yvar2 = np.log10(trace3D[2][:, i_axis0] / trace3D[2][:, i_axis2] - 1.0)
        plot = ax.plot(
            xvar0, yvar0
          , xvar1, yvar1
          , xvar2, yvar2
                      )
        plot[0].set_linestyle("-");   plot[0].set_color("orange");  plot[0].set_linewidth(2)
        plot[1].set_linestyle("-");   plot[1].set_color("purple");  plot[1].set_linewidth(2)
        plot[2].set_linestyle("-");   plot[2].set_color("green") ;  plot[2].set_linewidth(2)
        if ivar==0:
            plot_MJ = ax.plot(
                nH_J[:], M_J_p6[:]
              , nH_J[:], M_J_p4[:]
              , nH_J[:], M_J_p2[:]
              , nH_J[:], M_J_p0[:]
              , nH_J[:], M_J_m2[:]
              , nH_J[:], M_J_m4[:]
                        )
            plot_MJ[0].set_linestyle(":");   plot_MJ[0].set_color("black") ;  plot_MJ[0].set_linewidth(1)
            plot_MJ[1].set_linestyle(":");   plot_MJ[1].set_color("black") ;  plot_MJ[1].set_linewidth(1)
            plot_MJ[2].set_linestyle(":");   plot_MJ[2].set_color("black") ;  plot_MJ[2].set_linewidth(1)
            plot_MJ[3].set_linestyle(":");   plot_MJ[3].set_color("black") ;  plot_MJ[3].set_linewidth(1)
            plot_MJ[4].set_linestyle(":");   plot_MJ[4].set_color("black") ;  plot_MJ[4].set_linewidth(1)
            plot_MJ[5].set_linestyle(":");   plot_MJ[5].set_color("black") ;  plot_MJ[5].set_linewidth(1)
            ax.annotate(r'$M_{\rm J} = 10^6 \ {\rm M}_{\bigodot}$', xy=( 0.0, 3.48), xycoords='data', color='black', fontsize=fontsize_legend, va='top'   , ha='left'  )
            ax.annotate(            r'$10^4 \ {\rm M}_{\bigodot}$', xy=( 6.0, 3.48), xycoords='data', color='black', fontsize=fontsize_legend, va='top'   , ha='center')
            ax.annotate(            r'$10^2 \ {\rm M}_{\bigodot}$', xy=(10.0, 3.48), xycoords='data', color='black', fontsize=fontsize_legend, va='top'   , ha='center')
            ax.annotate(               r'$1 \ {\rm M}_{\bigodot}$', xy=(13.0, 3.48), xycoords='data', color='black', fontsize=fontsize_legend, va='top'   , ha='center')
            ax.annotate(         r'$10^{-2} \ {\rm M}_{\bigodot}$', xy=(16.0, 2.50), xycoords='data', color='black', fontsize=fontsize_legend, va='center', ha='right' )
        if ivar==0:
            plt.legend(labelspacing=0.0, loc='lower right')
            ax.legend(progmass, ncol=1, fontsize=fontsize_legend)
    if nvar == 1:
        fig.savefig("%s/nT.pdf" % outdir)
    if nvar == 3:
        fig.savefig("%s/nTge.pdf" % outdir)

    plt.close('all')

    # Density-Abundance
    progmass = [
       r"(a) $M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"(b) $M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"(c) $M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
      ]

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18.0,16.0), constrained_layout=True, sharex='all')
 ## fig.suptitle(progmass[iSN], fontsize=fontsize_suptitle, fontweight='bold')

    for iSN in range(3):
        # H-baring species
        ax=axs[0][iSN]
        ax.annotate(progmass[iSN], xy=(0.03, 1.10), xycoords=ax.transAxes, color='black', fontsize=fontsize_title, va='top', ha='left')
        if iSN==0: ax.set_ylabel(r"log [ Abundance ]"          , fontsize=fontsize_label)
        ax.set_xlim(0, 16)
        ax.set_ylim(-12, 2)
        ax.set_xticks(np.linspace( 0, 16, 5))
        ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
        ax.set_yticks(np.linspace(-10, 0, 3))
        ax.set_yticks(np.linspace(-12, 2, 15), minor=True)
        ax.tick_params(labelsize=fontsize_tick)
        if iSN>0: ax.tick_params(labelleft=False)
        plot = ax.plot(
            np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_elec])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_HI  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_HII ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_H2I ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_HM  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_H2II])
                      )
        plot[0].set_linestyle("-");    plot[0].set_color("orange");  plot[0].set_linewidth(2)
        plot[1].set_linestyle("-");    plot[1].set_color("black") ;  plot[1].set_linewidth(2)
        plot[2].set_linestyle("-");    plot[2].set_color("red")   ;  plot[2].set_linewidth(2)
        plot[3].set_linestyle("-");    plot[3].set_color("blue")  ;  plot[3].set_linewidth(2)
        plot[4].set_linestyle("--");   plot[4].set_color("purple");  plot[4].set_linewidth(2)
        plot[5].set_linestyle("--");   plot[5].set_color("green") ;  plot[5].set_linewidth(2)
        if iSN == 0:
            plt.legend(labelspacing=0.0, loc='lower right')
            ax.legend([
               r"e"
             , r"H"
             , r"H$^+$"
             , r"H$_2$"
             , r"H$^-$"
             , r"H$_2^+$"
                    ], ncol=2, fontsize=fontsize_legend)

        # D-baring species
        ax=axs[1][iSN]
        if iSN==0: ax.set_ylabel(r"log [ Abundance ]"          , fontsize=fontsize_label)
        ax.set_xlim(0, 16)
        ax.set_ylim(-17.5, -3.5)
        ax.set_xticks(np.linspace( 0, 16, 5))
        ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
        ax.set_yticks(np.linspace(-15, -5, 3))
        ax.set_yticks(np.linspace(-17, -4, 14), minor=True)
        ax.tick_params(labelsize=fontsize_tick)
        if iSN>0: ax.tick_params(labelleft=False)
        plot = ax.plot(
            np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_DI  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_DII ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_HDI ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_DM  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_HDII])
                      )
        plot[0].set_linestyle("-");    plot[0].set_color("black") ;  plot[0].set_linewidth(2)
        plot[1].set_linestyle("-");    plot[1].set_color("red")   ;  plot[1].set_linewidth(2)
        plot[2].set_linestyle("-");    plot[2].set_color("blue")  ;  plot[2].set_linewidth(2)
        plot[3].set_linestyle("--");   plot[3].set_color("purple");  plot[3].set_linewidth(2)
        plot[4].set_linestyle("--");   plot[4].set_color("green") ;  plot[4].set_linewidth(2)
        if iSN == 0:
            plt.legend(labelspacing=0.0, loc='lower right')
            ax.legend([
               r"D"
             , r"D$^+$"
             , r"HD"
             , r"D$^-$"
             , r"HD$^+$"
                    ], ncol=2, fontsize=fontsize_legend)

        # C-baring species
        ax=axs[2][iSN]
        if iSN==0: ax.set_ylabel(r"log [ Abundance ]"          , fontsize=fontsize_label)
        ax.set_xlim(0, 16)
        ax.set_ylim(-19, -5)
        ax.set_xticks(np.linspace( 0, 16, 5))
        ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
        ax.set_yticks(np.linspace(-15, -5, 3))
        ax.set_yticks(np.linspace(-19, -5, 15), minor=True)
        ax.tick_params(labelsize=fontsize_tick)
        if iSN>0: ax.tick_params(labelleft=False)
        plot = ax.plot(
            np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_CI  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_CII ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_CO  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_CO2 ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_AC  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_CH  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_CH2 ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_COII])
                      )
        plot[0].set_linestyle("-");    plot[0].set_color("black")  ;  plot[0].set_linewidth(2)
        plot[1].set_linestyle("-");    plot[1].set_color("red")    ;  plot[1].set_linewidth(2)
        plot[2].set_linestyle("-");    plot[2].set_color("orange") ;  plot[2].set_linewidth(2)
        plot[3].set_linestyle("-");    plot[3].set_color("green")  ;  plot[3].set_linewidth(2)
        plot[4].set_linestyle(":");    plot[4].set_color("black")  ;  plot[4].set_linewidth(2)
        plot[5].set_linestyle("--");   plot[5].set_color("blue")   ;  plot[5].set_linewidth(2)
        plot[6].set_linestyle("--");   plot[6].set_color("purple") ;  plot[6].set_linewidth(2)
        plot[7].set_linestyle("--");   plot[7].set_color("orange") ;  plot[7].set_linewidth(2)
        plt.legend(labelspacing=0.0, loc='lower right')
        if iSN == 0:
            ax.legend([
               r"C"
             , r"C$^+$"
             , r"CO"
             , r"CO$_2$"
             , r"C (dust)"
             , r"CH"
             , r"CH$_2$"
             , r"CO$^+$"
                    ], ncol=2, fontsize=fontsize_legend)

        # O-baring species
        ax=axs[3][iSN]
        ax.set_xlabel(r"log [ Density / cm$^{-3}$ ]", fontsize=fontsize_label)
        if iSN==0: ax.set_ylabel(r"log [ Abundance ]"          , fontsize=fontsize_label)
        ax.set_xlim(0, 16)
        ax.set_ylim(-19, -5)
        ax.set_xticks(np.linspace( 0, 16, 5))
        ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
        ax.set_yticks(np.linspace(-15, -5, 3))
        ax.set_yticks(np.linspace(-19, -5, 15), minor=True)
        ax.tick_params(labelsize=fontsize_tick)
        if iSN>0: ax.tick_params(labelleft=False)
        plot = ax.plot(
            np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_OI   ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_OH   ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_H2O  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_O2   ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_OII  ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_OHII ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_H2OII])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_H3OII])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(trace3D[iSN][:, i_O2II ])
                      )
        plot[0].set_linestyle("-");   plot[0].set_color("black") ;  plot[0].set_linewidth(2)
        plot[1].set_linestyle("-");   plot[1].set_color("green") ;  plot[1].set_linewidth(2)
        plot[2].set_linestyle("-");   plot[2].set_color("blue")  ;  plot[2].set_linewidth(2)
        plot[3].set_linestyle("-");   plot[3].set_color("orange");  plot[3].set_linewidth(2)
        plot[4].set_linestyle(":");   plot[4].set_color("black") ;  plot[4].set_linewidth(2)
        plot[5].set_linestyle(":");   plot[5].set_color("green") ;  plot[5].set_linewidth(2)
        plot[6].set_linestyle(":");   plot[6].set_color("blue")  ;  plot[6].set_linewidth(2)
        plot[7].set_linestyle(":");   plot[7].set_color("purple");  plot[7].set_linewidth(2)
        plot[8].set_linestyle(":");   plot[8].set_color("orange");  plot[8].set_linewidth(2)
        plt.legend(labelspacing=0.0, loc='lower right')
        if iSN == 0:
            ax.legend(["O"
                     , "OH"
                     , "H$_{2}$O"
                     , "O$_{2}$"
                     , "O$^{+}$"
                     , "OH$^{+}$"
                     , "H$_{2}$O$^{+}$"
                     , "H$_{3}$O$^{+}$"
                     , "O$_{2}$$^{+}$"
                    ], ncol=2, fontsize=fontsize_legend)

    plt.subplots_adjust(left   = 0.07
                      , right  = 0.98
                      , bottom = 0.07
                      , top    = 0.96
                      , wspace = 0.08
                      , hspace = 0.00
                       )

    fig.savefig("%s/ny.pdf" % (outdir))

    plt.close('all')


    # Cooling functions
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18.0,5.0), constrained_layout=True, sharex='all')
 ## fig.suptitle(progmass[iSN], fontsize=fontsize_suptitle, fontweight='bold')

    for iSN in range(3):
        # H-baring species
        ax=axs[iSN]
        ax.annotate(progmass[iSN], xy=(0.03, 1.10), xycoords=ax.transAxes, color='black', fontsize=fontsize_title, va='top', ha='left')
        ax.set_xlabel(r"log [ Density / cm$^{-3}$ ]", fontsize=fontsize_label)
        if iSN==0: ax.set_ylabel(r"log [ Cooling rates / erg g$^{-1}$ s$^{-1}$ ]", fontsize=fontsize_label)
        ax.set_xlim(0, 16)
        ax.set_ylim(-6, 4)
        ax.set_xticks(np.linspace( 0, 16, 5))
        ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
        ax.set_yticks(np.linspace(-6, 4, 6))
        ax.set_yticks(np.linspace(-6, 4, 11), minor=True)
        ax.tick_params(labelsize=fontsize_tick)
        if iSN>0: ax.tick_params(labelleft=False)
        plot = ax.plot(
            np.log10(trace3D[iSN][:, i_nH]), np.log10( trace3D[iSN][:, i_edot_adia   ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_H2line ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10( trace3D[iSN][:, i_edot_H2form ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_H2form ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_HDline ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_CIline ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_CIIline])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_OIline ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_H2Oline])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_OHline ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_COline ])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_dst_rad])
          , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_H2cont ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_comp   ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_brem   ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_ion_H  ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_rec_H  ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_line_H ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_ion_He ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_rec_He ])
  ####    , np.log10(trace3D[iSN][:, i_nH]), np.log10(-trace3D[iSN][:, i_edot_line_He])
                      )
        plot[ 0].set_linestyle("-");   plot[ 0].set_color("red");    plot[ 0].set_linewidth(2)
        plot[ 1].set_linestyle("-");   plot[ 1].set_color("blue") ;  plot[ 1].set_linewidth(2)
        plot[ 2].set_linestyle("-");   plot[ 2].set_color("purple"); plot[ 2].set_linewidth(2)
        plot[ 3].set_linestyle(":");   plot[ 3].set_color("purple"); plot[ 3].set_linewidth(2)
        plot[ 4].set_linestyle("-");   plot[ 4].set_color("green");  plot[ 4].set_linewidth(2)
        plot[ 5].set_linestyle(":");   plot[ 5].set_color("black");  plot[ 5].set_linewidth(2)
        plot[ 6].set_linestyle(":");   plot[ 6].set_color("red");    plot[ 6].set_linewidth(2)
        plot[ 7].set_linestyle(":");   plot[ 7].set_color("orange"); plot[ 7].set_linewidth(2)
        plot[ 8].set_linestyle("--");  plot[ 8].set_color("dodgerblue"); plot[ 8].set_linewidth(2)
        plot[ 9].set_linestyle("--");  plot[ 9].set_color("green") ; plot[ 9].set_linewidth(2)
        plot[10].set_linestyle("--");  plot[10].set_color("orange"); plot[10].set_linewidth(2)
        plot[11].set_linestyle("-");   plot[11].set_color("grey");   plot[11].set_linewidth(2)
        plot[12].set_linestyle(":");   plot[12].set_color("blue");   plot[12].set_linewidth(2)
        if iSN==0:
            ax.annotate(r"adia"   , xy=( 5.0,-1.5), xycoords='data', color="red"    , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"H$_2$"  , xy=( 2.0,-3.0), xycoords='data', color="blue"   , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"form"   , xy=(11.0, 3.0), xycoords='data', color="purple" , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"diss"   , xy=(15.0,-8.0), xycoords='data', color="purple" , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"HD"     , xy=( 4.0,-2.3), xycoords='data', color="green"  , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"CI"     , xy=( 3.0,-5.5), xycoords='data', color="black"  , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"CII"    , xy=( 3.0,-8.0), xycoords='data', color="red"    , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"OI"     , xy=( 5.5,-3.7), xycoords='data', color="orange" , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"H$_2$O" , xy=( 8.0, 0.5), xycoords='data', color="dodgerblue", fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"OH"     , xy=(11.0,-4.0), xycoords='data', color="green"  , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"CO"     , xy=(10.2,-2.8), xycoords='data', color="orange" , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"dust"   , xy=(13.5, 3.2), xycoords='data', color="grey"   , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)
            ax.annotate(r"CIE"    , xy=(15.0, 0.5), xycoords='data', color="blue"   , fontsize=fontsize_legend, va='center' , ha='center', zorder=4)

    plt.subplots_adjust(left   = 0.07
                      , right  = 0.98
                      , bottom = 0.20
                      , top    = 0.92
                      , wspace = 0.08
                      , hspace = 0.00
                       )

    fig.savefig("%s/nL.pdf" % (outdir))

    plt.close('all')


if MASS_HISTORY:
    i_redshift    =  0
    i_time        =  1
    i_Rvir        =  2
    i_Mvir_dm     =  3
    i_halo_Mgas   =  4
    i_halo_Mmet   =  5
    i_halo_MC     =  6
    i_halo_MO     =  7
    i_halo_MFe    =  8
    i_enr_Mgas    =  9
    i_enr_Mmet    = 10
    i_enr_MC      = 11
    i_enr_MO      = 12
    i_enr_MFe     = 13
    i_ljeans      = 14
    i_nHmax       = 15
    i_cloud_Mgas  = 16
    i_cloud_Mmet  = 17
    i_cloud_MC    = 18
    i_cloud_MO    = 19
    i_cloud_MFe   = 20
    i_core_MC     = 21
    i_core_MCdust = 22

    progmass = [
       r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
      ]

    trace3D = [None] * 4

    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M13"
    pdir = indir + '/time_series'
    fn_traceM13 = pdir + '/time_series_mass.dat'
    trace3D[0] = np.loadtxt(fn_traceM13)


    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M50"
    pdir = indir + '/time_series'
    fn_traceM50 = pdir + '/time_series_mass.dat'
    trace3D[1] = np.loadtxt(fn_traceM50)


    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M80"
    pdir = indir + '/time_series'
    fn_traceM80 = pdir + '/time_series_mass.dat'
    trace3D[2] = np.loadtxt(fn_traceM80)


    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_cgrackle_HR2"
    pdir = indir + '/time_series'
    fn_traceC13 = pdir + '/time_series_mass.dat'
    trace3D[3] = np.loadtxt(fn_traceC13)

#   trace3D[0][trace3D[0] <= 1.0e-8] = None
#   trace3D[1][trace3D[1] <= 1.0e-8] = None
#   trace3D[2][trace3D[2] <= 1.0e-8] = None
#   trace3D[3][trace3D[3] <= 1.0e-8] = None

    time_sf = 364.3
    time_life = [11.8, 3.49, 3.03]

    nvar = 1
    if nvar==1: ysize = 5.2
    if nvar==3: ysize =12.0
#   fig, axs = plt.subplots(nrows=nvar, ncols=1, figsize=(6.4,ysize), constrained_layout=True)
#   for ivar in range(nvar):
#       if nvar==1: ax = axs
#       else:       ax = axs[ivar]
#       ax.tick_params(labelsize=fontsize_tick)
#       if ivar==nvar-1: ax.set_xlabel(r"log [ Time from star formation / Myr ]", fontsize=fontsize_label)
#       if nvar==1:
#           ax.set_ylabel(r"log [ Mass / M$_{\bigodot}$ ]", fontsize=fontsize_label)
#       t_max = max(trace3D[2][:, i_time]) - time_sf
#       ax.set_xlim([np.log10(1.0), np.log10(t_max)])
#       # Plot data ###
#       plot_dm = ax.plot(
#           np.log10(trace3D[0][:, i_time] - time_sf), np.log10(trace3D[0][:, i_Mvir_dm])
#         , np.log10(trace3D[1][:, i_time] - time_sf), np.log10(trace3D[1][:, i_Mvir_dm])
#         , np.log10(trace3D[2][:, i_time] - time_sf), np.log10(trace3D[2][:, i_Mvir_dm])
#                     )
#       plot_dm[0].set_linestyle("-");   plot_dm[0].set_color("orange");  plot_dm[0].set_linewidth(2)
#       plot_dm[1].set_linestyle("-");   plot_dm[1].set_color("purple");  plot_dm[1].set_linewidth(2)
#       plot_dm[2].set_linestyle("-");   plot_dm[2].set_color("green" );  plot_dm[2].set_linewidth(2)
#       plot_ba = ax.plot(
#           np.log10(trace3D[0][:, i_time] - time_sf), np.log10(trace3D[0][:, i_halo_Mgas])
#         , np.log10(trace3D[1][:, i_time] - time_sf), np.log10(trace3D[1][:, i_halo_Mgas])
#         , np.log10(trace3D[2][:, i_time] - time_sf), np.log10(trace3D[2][:, i_halo_Mgas])
#                     )
#       plot_ba[0].set_linestyle("--");  plot_ba[0].set_color("orange");  plot_ba[0].set_linewidth(2)
#       plot_ba[1].set_linestyle("--");  plot_ba[1].set_color("purple");  plot_ba[1].set_linewidth(2)
#       plot_ba[2].set_linestyle("--");  plot_ba[2].set_color("green" );  plot_ba[2].set_linewidth(2)
#       X_exp0 = [time_life[0], time_life[0]]; Y_exp0 = [4.0, 7.0]
#       X_exp1 = [time_life[1], time_life[1]]; Y_exp1 = [4.0, 7.0]
#       X_exp2 = [time_life[2], time_life[2]]; Y_exp2 = [4.0, 7.0]
#       plot_ba = ax.plot(
#           np.log10(X_exp0), Y_exp0
#         , np.log10(X_exp1), Y_exp1
#         , np.log10(X_exp2), Y_exp2
#                     )
#       plot_ba[0].set_linestyle(":");   plot_ba[0].set_color("orange");  plot_ba[0].set_linewidth(2)
#       plot_ba[1].set_linestyle(":");   plot_ba[1].set_color("purple");  plot_ba[1].set_linewidth(2)
#       plot_ba[2].set_linestyle(":");   plot_ba[2].set_color("green" );  plot_ba[2].set_linewidth(2)
#       ax.set_ylim(4.0, 7.0)
#       if ivar==0:
#           plt.legend(labelspacing=0.0, loc='lower right')
#           ax.legend(progmass, ncol=1, fontsize=fontsize_legend)
#       # Redshift ticks ###
#       redshift_ticks = [12, 11, 10, 9, 8, 7]
#       time_ticks     = [0.0]*6
#       redshift_list = np.flip(trace3D[2][:, i_redshift], axis=0)
#       time_list     = np.flip(trace3D[2][:, i_time]    , axis=0)
#       for itime in range(6):
#           time_ticks[itime] = np.interp(redshift_ticks[itime], redshift_list, time_list)
#           time_ticks[itime] = np.log10(time_ticks[itime] - time_sf)
#       ax2 = ax.twiny()
#       ax2.set_xlim(ax.get_xlim())
#       ax2.set_xticks(time_ticks)
#       ax2.set_xticklabels(redshift_ticks)
#       ax2.tick_params(labelsize=fontsize_tick)
#       ax2.set_xlabel(r"Redshift", fontsize=fontsize_label)
#   fig.savefig("%s/tm.pdf" % outdir)


    # metal mass
    ### in halo
    for ivar in range(6):
        if ivar == 0: var = 'gas'
        if ivar == 1: var = 'met'
        if ivar == 2: var = 'fmet'
        if ivar == 3: var = 'Zmet'
        if ivar == 4: var = 'AC'
        if ivar == 5: var = 'FeH'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 5.2), constrained_layout=True)
        ax.tick_params(labelsize=fontsize_tick)
 #####  if iregion == 0: ax.set_xlabel(r"log [ Time from star formation / Myr ]", fontsize=fontsize_label)
 #####  if iregion == 1: ax.set_xlabel(r"log [ Density / cm$^{-3}$ ]", fontsize=fontsize_label)
        ax.set_xlabel(r"log [ Density / cm$^{-3}$ ]", fontsize=fontsize_label)
        for iregion in range(2):
            if iregion == 0: region = 'halo'
            if iregion == 1: region = 'cloud'
            if ivar == 0: ax.set_ylabel(r"Gas mass in %s [ M$_{\bigodot}$ ]"    % region, fontsize=fontsize_label)
            if ivar == 1: ax.set_ylabel(r"Metal mass in %s [ M$_{\bigodot}$ ]"  % region, fontsize=fontsize_label)
            if ivar == 2: ax.set_ylabel(r"Metal mass frac. in %s"               % region, fontsize=fontsize_label)
            if ivar == 3: ax.set_ylabel(r"Metallicity in %s [ Z$_{\bigodot}$ ]" % region, fontsize=fontsize_label)
            if ivar == 4: ax.set_ylabel(r"$A$(C) in %s"                         % region, fontsize=fontsize_label)
            if ivar == 5: ax.set_ylabel(r"[Fe/H] in %s"                         % region, fontsize=fontsize_label)
     #####  if iregion == 0:
     #####      t_max = max(trace3D[2][:, i_time]) - time_sf
     #####      ax.set_xlim([np.log10(1.0), np.log10(t_max)])
            # Plot data ###
            for iSN in range(4):
     #####      if iregion == 0: XX = np.log10(trace3D[iSN][:, i_time] - time_sf)
     #####      if iregion == 1: XX = np.log10(trace3D[iSN][:, i_nHmax])
                XX = np.log10(trace3D[iSN][:, i_nHmax])
                if ivar == 0:
                   if iregion == 0:
                       YY = trace3D[iSN][:, i_halo_Mgas ]
                   if iregion == 1:
                       YY = trace3D[iSN][:, i_cloud_Mgas]
                if ivar == 1:
                   if iregion == 0:
                       YY = trace3D[iSN][:, i_halo_Mmet ]
                   if iregion == 1:
                       YY = trace3D[iSN][:, i_cloud_Mmet]
                if ivar == 2:
                   if iregion == 0:
                       YY = trace3D[iSN][:, i_halo_Mmet ]/trace3D[iSN][:, i_enr_Mmet]
                    ## print("f_Z halo")
                    ## for iY in range (YY.size):
                    ##     print(" %11.5f %11.5f" % (XX[iY], YY[iY]))
                   if iregion == 1:
                       YY = trace3D[iSN][:, i_cloud_Mmet]/trace3D[iSN][:, i_enr_Mmet]
                    ## print("enr_MC")
                    ## if iSN == 0: print(6.69235e-01*trace3D[iSN][:, i_enr_Mmet])
                    ## if iSN == 1: print(2.79167e-01*trace3D[iSN][:, i_enr_Mmet])
                    ## if iSN == 2: print(2.52563e-01*trace3D[iSN][:, i_enr_Mmet])
                    ## if iSN == 3: print(2.65314e-01*trace3D[iSN][:, i_enr_Mmet])
                    ## print("enr_MFe")
                    ## if iSN == 0: print(8.90341e-06*trace3D[iSN][:, i_enr_Mmet])
                    ## if iSN == 1: print(4.15804e-06*trace3D[iSN][:, i_enr_Mmet])
                    ## if iSN == 2: print(2.43915e-06*trace3D[iSN][:, i_enr_Mmet])
                    ## if iSN == 3: print(9.62448e-02*trace3D[iSN][:, i_enr_Mmet])
                if ivar == 3:
                   if iregion == 0:
                       YY = trace3D[iSN][:, i_halo_Mmet] /trace3D[iSN][:, i_halo_Mgas]  /SolarMetalFractionByMass
                   if iregion == 1:                      
                       YY = trace3D[iSN][:, i_cloud_Mmet]/trace3D[iSN][:, i_cloud_Mgas] /SolarMetalFractionByMass
                       print("Z cloud")
                       for iY in range (YY.size):
                           print(" %11.3e %11.3e" % (trace3D[iSN][iY, i_cloud_Mmet]/trace3D[iSN][iY, i_cloud_Mgas], YY[iY]))
                if ivar == 4:
                   if iregion == 0:
                       if iSN == 0: YY = 12.0 + np.log10(6.69235e-01*trace3D[iSN][:, i_halo_Mmet ]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])
                       if iSN == 1: YY = 12.0 + np.log10(2.79167e-01*trace3D[iSN][:, i_halo_Mmet ]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])
                       if iSN == 2: YY = 12.0 + np.log10(2.52563e-01*trace3D[iSN][:, i_halo_Mmet ]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])
                       if iSN == 3: YY = 12.0 + np.log10(2.65314e-01*trace3D[iSN][:, i_halo_Mmet ]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])
                       YY = 12.0 + np.log10(trace3D[iSN][:, i_halo_MC ]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas])
                   if iregion == 1:
                       if iSN == 0: YY = 12.0 + np.log10(6.69235e-01*trace3D[iSN][:, i_cloud_Mmet]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])
                       if iSN == 1: YY = 12.0 + np.log10(2.79167e-01*trace3D[iSN][:, i_cloud_Mmet]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])
                       if iSN == 2: YY = 12.0 + np.log10(2.52563e-01*trace3D[iSN][:, i_cloud_Mmet]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])
                       if iSN == 3: YY = 12.0 + np.log10(2.65314e-01*trace3D[iSN][:, i_cloud_Mmet]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])
                    ## YY2 = 12.0 + np.log10(trace3D[iSN][:, i_cloud_MC]/12.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])
                    ## print("A(C)")
                    ## for iY in range (YY.size):
                    ##     print(" %7.2f %7.2f" % (YY[iY], YY2[iY]))
                if ivar == 5:
                   if iregion == 0:
                       if iSN == 0: YY = 12.0 + np.log10(8.90341e-06*trace3D[iSN][:, i_halo_Mmet ] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])  - SolarIronAbundance
                       if iSN == 1: YY = 12.0 + np.log10(4.15804e-06*trace3D[iSN][:, i_halo_Mmet ] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])  - SolarIronAbundance
                       if iSN == 2: YY = 12.0 + np.log10(2.43915e-06*trace3D[iSN][:, i_halo_Mmet ] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])  - SolarIronAbundance
                       if iSN == 3: YY = 12.0 + np.log10(9.62448e-02*trace3D[iSN][:, i_halo_Mmet ] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas ])  - SolarIronAbundance
                       YY2 = 12.0 + np.log10(trace3D[iSN][:, i_halo_MFe] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_halo_Mgas])  - SolarIronAbundance
                   if iregion == 1:
                       if iSN == 0: YY = 12.0 + np.log10(8.90341e-06*trace3D[iSN][:, i_cloud_Mmet] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])  - SolarIronAbundance
                       if iSN == 1: YY = 12.0 + np.log10(4.15804e-06*trace3D[iSN][:, i_cloud_Mmet] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])  - SolarIronAbundance
                       if iSN == 2: YY = 12.0 + np.log10(2.43915e-06*trace3D[iSN][:, i_cloud_Mmet] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])  - SolarIronAbundance
                       if iSN == 3: YY = 12.0 + np.log10(9.62448e-02*trace3D[iSN][:, i_cloud_Mmet] /56.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas])  - SolarIronAbundance
                    ## YY2 = 12.0 + np.log10(trace3D[iSN][:, i_cloud_MFe]/56.0/HydrogenFractionByMass/trace3D[iSN][:, i_cloud_Mgas]) - SolarIronAbundance
                    ## print("[Fe/H]")
                    ## for iY in range (YY.size):
                    ##     print(" %7.2f %7.2f" % (YY[iY], YY2[iY]))
                if ivar < 4:
                    YY[YY <= 1.0e-8] = None
                    plot_tm = ax.semilogy(XX[:], YY[:])
                else:
                    if ivar == 4: YY[YY <=   0.0] = None
                    if ivar == 5: YY[YY <= -12.0] = None
                    plot_tm = ax.plot(XX[:], YY[:])
                if iregion == 0:
                    if iSN == 0: plot_tm[0].set_linestyle("--"); plot_tm[0].set_color("orange");  plot_tm[0].set_linewidth(2)
                    if iSN == 1: plot_tm[0].set_linestyle("--"); plot_tm[0].set_color("purple");  plot_tm[0].set_linewidth(2)
                    if iSN == 2: plot_tm[0].set_linestyle("--"); plot_tm[0].set_color("green" );  plot_tm[0].set_linewidth(2)
                    if iSN == 3: plot_tm[0].set_linestyle("--"); plot_tm[0].set_color("black" );  plot_tm[0].set_linewidth(2)
                if iregion == 1:
                    if iSN == 0: plot_tm[0].set_linestyle("-");  plot_tm[0].set_color("orange");  plot_tm[0].set_linewidth(2)
                    if iSN == 1: plot_tm[0].set_linestyle("-");  plot_tm[0].set_color("purple");  plot_tm[0].set_linewidth(2)
                    if iSN == 2: plot_tm[0].set_linestyle("-");  plot_tm[0].set_color("green" );  plot_tm[0].set_linewidth(2)
                    if iSN == 3: plot_tm[0].set_linestyle("-");  plot_tm[0].set_color("black" );  plot_tm[0].set_linewidth(2)
        plt.legend(labelspacing=0.0, loc='lower right')
        ax.legend(progmass, ncol=1, fontsize=fontsize_legend)
        # Redshift ticks ###
######  if iregion == 0:
######      redshift_ticks = [12, 11, 10, 9, 8, 7]
######      time_ticks     = [0.0]*6
######      redshift_list = np.flip(trace3D[2][:, i_redshift], axis=0)
######      time_list     = np.flip(trace3D[2][:, i_time]    , axis=0)
######      for itime in range(6):
######          time_ticks[itime] = np.interp(redshift_ticks[itime], redshift_list, time_list)
######          time_ticks[itime] = np.log10(time_ticks[itime] - time_sf)
######      ax2 = ax.twiny()
######      ax2.set_xlim(ax.get_xlim())
######      ax2.set_xticks(time_ticks)
######      ax2.set_xticklabels(redshift_ticks)
######      ax2.tick_params(labelsize=fontsize_tick)
######      ax2.set_xlabel(r"Redshift", fontsize=fontsize_label)
######  fig.savefig("%s/tm_%s_%s.pdf" % (outdir, region, var))
        fig.savefig("%s/tm_%s.pdf" % (outdir, var))
        plt.close('all')



if C_FE:
    # OBSERVATIONAL DATA
    i_Fe_H   = 0
    i_A_C    = 1
    i_dmFe_H = 2
    i_dpFe_H = 3
    i_dmA_C  = 4
    i_dpA_C  = 5
    abdata = [None] * 11
    abdir = "/home/genchiaki/dust/dil/MergingHalo/Fe_C/cgisess_32fdbf0214126b4a9c3a66534bcb3a09"
    abfile = abdir + '_MP.dat'         ; abdata[ 0] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_EMP_RGB.dat'    ; abdata[ 1] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_EMP_MS.dat'     ; abdata[ 2] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_Crich_RGB.dat'  ; abdata[ 3] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_Crich_MS.dat'   ; abdata[ 4] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_CEMP_RGB.dat'   ; abdata[ 5] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_CEMP_MS.dat'    ; abdata[ 6] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_CEMP-s_RGB.dat' ; abdata[ 7] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_CEMP-s_MS.dat'  ; abdata[ 8] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_CEMP-no_RGB.dat'; abdata[ 9] = np.loadtxt(abfile, comments = '#')
    abfile = abdir + '_CEMP-no_MS.dat' ; abdata[10] = np.loadtxt(abfile, comments = '#')
    dx = -0.5; dy = -0.7
    narrow = 122
    startpoint = [[0.0]*2] * narrow
    endpoint   = [[0.0]*2] * narrow
    iarrow = 0
    startpoint[iarrow] = [-2.06, 7.36];  endpoint[iarrow] = [-2.06, 7.36+dy];  iarrow += 1
    startpoint[iarrow] = [-2.03, 7.39];  endpoint[iarrow] = [-2.03, 7.39+dy];  iarrow += 1
    startpoint[iarrow] = [-2.25, 6.79];  endpoint[iarrow] = [-2.25, 6.79+dy];  iarrow += 1
    startpoint[iarrow] = [-2.26, 6.48];  endpoint[iarrow] = [-2.26, 6.48+dy];  iarrow += 1
    startpoint[iarrow] = [-2.21, 6.68];  endpoint[iarrow] = [-2.21, 6.68+dy];  iarrow += 1
    startpoint[iarrow] = [-1.85, 6.94];  endpoint[iarrow] = [-1.85, 6.94+dy];  iarrow += 1
    startpoint[iarrow] = [-2.08, 6.81];  endpoint[iarrow] = [-2.08, 6.81+dy];  iarrow += 1
    startpoint[iarrow] = [-2.57, 6.00];  endpoint[iarrow] = [-2.57, 6.00+dy];  iarrow += 1
    startpoint[iarrow] = [-2.14, 7.00];  endpoint[iarrow] = [-2.14, 7.00+dy];  iarrow += 1
    startpoint[iarrow] = [-2.71, 7.26];  endpoint[iarrow] = [-2.71, 7.26+dy];  iarrow += 1 
    startpoint[iarrow] = [-1.87, 6.85];  endpoint[iarrow] = [-1.87, 6.85+dy];  iarrow += 1
    startpoint[iarrow] = [-2.20, 7.05];  endpoint[iarrow] = [-2.20, 7.05+dy];  iarrow += 1
    startpoint[iarrow] = [-2.47, 6.35];  endpoint[iarrow] = [-2.47, 6.35+dy];  iarrow += 1 
    startpoint[iarrow] = [-2.01, 6.90];  endpoint[iarrow] = [-2.01, 6.90+dy];  iarrow += 1
    startpoint[iarrow] = [-2.54, 6.80];  endpoint[iarrow] = [-2.54, 6.80+dy];  iarrow += 1
    startpoint[iarrow] = [-1.43, 7.10];  endpoint[iarrow] = [-1.43, 7.10+dy];  iarrow += 1
    startpoint[iarrow] = [-2.38, 9.05];  endpoint[iarrow] = [-2.38, 9.05+dy];  iarrow += 1 
    startpoint[iarrow] = [-1.93, 6.62];  endpoint[iarrow] = [-1.93, 6.62+dy];  iarrow += 1
    startpoint[iarrow] = [-2.90, 5.13];  endpoint[iarrow] = [-2.90, 5.13+dy];  iarrow += 1
    startpoint[iarrow] = [-2.64, 5.60];  endpoint[iarrow] = [-2.64, 5.60+dy];  iarrow += 1
    startpoint[iarrow] = [-2.45, 5.69];  endpoint[iarrow] = [-2.45, 5.69+dy];  iarrow += 1
    startpoint[iarrow] = [-3.19, 4.70];  endpoint[iarrow] = [-3.19, 4.70+dy];  iarrow += 1
    startpoint[iarrow] = [-2.85, 6.46];  endpoint[iarrow] = [-2.85, 6.46+dy];  iarrow += 1
    startpoint[iarrow] = [-2.89, 5.18];  endpoint[iarrow] = [-2.89, 5.18+dy];  iarrow += 1
    startpoint[iarrow] = [-3.78, 5.36];  endpoint[iarrow] = [-3.78, 5.36+dy];  iarrow += 1
    startpoint[iarrow] = [-3.25, 4.96];  endpoint[iarrow] = [-3.25, 4.96+dy];  iarrow += 1
    startpoint[iarrow] = [-4.14, 5.16];  endpoint[iarrow] = [-4.14, 5.16+dy];  iarrow += 1
    startpoint[iarrow] = [-3.35, 5.63];  endpoint[iarrow] = [-3.35, 5.63+dy];  iarrow += 1
    startpoint[iarrow] = [-2.70, 5.46];  endpoint[iarrow] = [-2.70, 5.46+dy];  iarrow += 1
    startpoint[iarrow] = [-3.69, 5.26];  endpoint[iarrow] = [-3.69, 5.26+dy];  iarrow += 1
    startpoint[iarrow] = [-4.59, 3.65];  endpoint[iarrow] = [-4.59, 3.65+dy];  iarrow += 1
    startpoint[iarrow] = [-2.94, 6.45];  endpoint[iarrow] = [-2.94, 6.45+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.71, 6.70];  endpoint[iarrow] = [-2.71, 6.70+dy];  iarrow += 1
    startpoint[iarrow] = [-2.66, 6.30];  endpoint[iarrow] = [-2.66, 6.30+dy];  iarrow += 1
    startpoint[iarrow] = [-3.55, 5.65];  endpoint[iarrow] = [-3.55, 5.65+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.62, 6.50];  endpoint[iarrow] = [-2.62, 6.50+dy];  iarrow += 1
    startpoint[iarrow] = [-3.47, 5.50];  endpoint[iarrow] = [-3.47, 5.50+dy];  iarrow += 1
    startpoint[iarrow] = [-2.68, 6.10];  endpoint[iarrow] = [-2.68, 6.10+dy];  iarrow += 1
    startpoint[iarrow] = [-2.52, 6.80];  endpoint[iarrow] = [-2.52, 6.80+dy];  iarrow += 1
    startpoint[iarrow] = [-3.10, 5.65];  endpoint[iarrow] = [-3.10, 5.65+dy];  iarrow += 1
    startpoint[iarrow] = [-3.14, 5.17];  endpoint[iarrow] = [-3.14, 5.17+dy];  iarrow += 1 
    startpoint[iarrow] = [-2.82, 7.20];  endpoint[iarrow] = [-2.82, 7.20+dy];  iarrow += 1
    startpoint[iarrow] = [-2.84, 6.50];  endpoint[iarrow] = [-2.84, 6.50+dy];  iarrow += 1
    startpoint[iarrow] = [-2.51, 6.25];  endpoint[iarrow] = [-2.51, 6.25+dy];  iarrow += 1
    startpoint[iarrow] = [-3.43, 6.27];  endpoint[iarrow] = [-3.43, 6.27+dy];  iarrow += 1
    startpoint[iarrow] = [-4.34, 5.54];  endpoint[iarrow] = [-4.34, 5.54+dy];  iarrow += 1
    startpoint[iarrow] = [-2.68, 6.75];  endpoint[iarrow] = [-2.68, 6.75+dy];  iarrow += 1
    startpoint[iarrow] = [-3.36, 4.67];  endpoint[iarrow] = [-3.36, 4.67+dy];  iarrow += 1
    startpoint[iarrow] = [-2.90, 5.68];  endpoint[iarrow] = [-2.90, 5.68+dy];  iarrow += 1
    startpoint[iarrow] = [-3.89, 4.25];  endpoint[iarrow] = [-3.89, 4.25+dy];  iarrow += 1
    startpoint[iarrow] = [-3.39, 4.78];  endpoint[iarrow] = [-3.39, 4.78+dy];  iarrow += 1
    startpoint[iarrow] = [-3.12, 5.08];  endpoint[iarrow] = [-3.12, 5.08+dy];  iarrow += 1
    startpoint[iarrow] = [-3.84, 6.47];  endpoint[iarrow] = [-3.84, 6.47+dy];  iarrow += 1
    startpoint[iarrow] = [-4.07, 5.30];  endpoint[iarrow] = [-4.07, 5.30+dy];  iarrow += 1
    startpoint[iarrow] = [-2.43, 6.26];  endpoint[iarrow] = [-2.43, 6.26+dy];  iarrow += 1
    startpoint[iarrow] = [-3.21, 7.60];  endpoint[iarrow] = [-3.21, 7.60+dy];  iarrow += 1
    startpoint[iarrow] = [-3.44, 6.70];  endpoint[iarrow] = [-3.44, 6.70+dy];  iarrow += 1
    startpoint[iarrow] = [-4.71, 4.70];  endpoint[iarrow] = [-4.71, 4.70+dy];  iarrow += 1
    startpoint[iarrow] = [-2.98, 5.77];  endpoint[iarrow] = [-2.98, 5.77+dy];  iarrow += 1
    startpoint[iarrow] = [-2.73, 6.96];  endpoint[iarrow] = [-2.73, 6.96+dy];  iarrow += 1
    startpoint[iarrow] = [-3.68, 6.56];  endpoint[iarrow] = [-3.68, 6.56+dy];  iarrow += 1
    startpoint[iarrow] = [-3.28, 6.46];  endpoint[iarrow] = [-3.28, 6.46+dy];  iarrow += 1
    startpoint[iarrow] = [-3.75, 6.56];  endpoint[iarrow] = [-3.75, 6.56+dy];  iarrow += 1
    startpoint[iarrow] = [-3.10, 6.96];  endpoint[iarrow] = [-3.10, 6.96+dy];  iarrow += 1
    startpoint[iarrow] = [-3.01, 7.06];  endpoint[iarrow] = [-3.01, 7.06+dy];  iarrow += 1
    startpoint[iarrow] = [-2.86, 7.06];  endpoint[iarrow] = [-2.86, 7.06+dy];  iarrow += 1
    startpoint[iarrow] = [-3.41, 7.16];  endpoint[iarrow] = [-3.41, 7.16+dy];  iarrow += 1
    startpoint[iarrow] = [-2.87, 6.76];  endpoint[iarrow] = [-2.87, 6.76+dy];  iarrow += 1
    startpoint[iarrow] = [-3.49, 6.66];  endpoint[iarrow] = [-3.49, 6.66+dy];  iarrow += 1
    startpoint[iarrow] = [-2.84, 7.16];  endpoint[iarrow] = [-2.84, 7.16+dy];  iarrow += 1
    startpoint[iarrow] = [-3.57, 5.96];  endpoint[iarrow] = [-3.57, 5.96+dy];  iarrow += 1
    startpoint[iarrow] = [-3.17, 6.76];  endpoint[iarrow] = [-3.17, 6.76+dy];  iarrow += 1
    startpoint[iarrow] = [-2.93, 7.16];  endpoint[iarrow] = [-2.93, 7.16+dy];  iarrow += 1
    startpoint[iarrow] = [-3.63, 7.16];  endpoint[iarrow] = [-3.63, 7.16+dy];  iarrow += 1
    startpoint[iarrow] = [-3.14, 6.66];  endpoint[iarrow] = [-3.14, 6.66+dy];  iarrow += 1
    startpoint[iarrow] = [-3.31, 6.86];  endpoint[iarrow] = [-3.31, 6.86+dy];  iarrow += 1
    startpoint[iarrow] = [-2.88, 7.26];  endpoint[iarrow] = [-2.88, 7.26+dy];  iarrow += 1
    startpoint[iarrow] = [-2.72, 7.26];  endpoint[iarrow] = [-2.72, 7.26+dy];  iarrow += 1
    startpoint[iarrow] = [-2.96, 6.86];  endpoint[iarrow] = [-2.96, 6.86+dy];  iarrow += 1
    startpoint[iarrow] = [-3.63, 6.10];  endpoint[iarrow] = [-3.63, 6.10+dy];  iarrow += 1
    startpoint[iarrow] = [-3.29, 5.90];  endpoint[iarrow] = [-3.29, 5.90+dy];  iarrow += 1
    startpoint[iarrow] = [-3.00, 6.15];  endpoint[iarrow] = [-3.00, 6.15+dy];  iarrow += 1
    startpoint[iarrow] = [-3.03, 6.00];  endpoint[iarrow] = [-3.03, 6.00+dy];  iarrow += 1
    startpoint[iarrow] = [-3.02, 6.35];  endpoint[iarrow] = [-3.02, 6.35+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.79, 7.10];  endpoint[iarrow] = [-2.79, 7.10+dy];  iarrow += 1
    startpoint[iarrow] = [-2.96, 6.10];  endpoint[iarrow] = [-2.96, 6.10+dy];  iarrow += 1
    startpoint[iarrow] = [-3.10, 6.70];  endpoint[iarrow] = [-3.10, 6.70+dy];  iarrow += 1
    startpoint[iarrow] = [-2.58, 7.40];  endpoint[iarrow] = [-2.58, 7.40+dy];  iarrow += 1
    startpoint[iarrow] = [-2.73, 6.40];  endpoint[iarrow] = [-2.73, 6.40+dy];  iarrow += 1
    startpoint[iarrow] = [-3.09, 6.10];  endpoint[iarrow] = [-3.09, 6.10+dy];  iarrow += 1
    startpoint[iarrow] = [-2.68, 6.20];  endpoint[iarrow] = [-2.68, 6.20+dy];  iarrow += 1
    startpoint[iarrow] = [-3.12, 6.15];  endpoint[iarrow] = [-3.12, 6.15+dy];  iarrow += 1  
    startpoint[iarrow] = [-3.14, 6.00];  endpoint[iarrow] = [-3.14, 6.00+dy];  iarrow += 1
    startpoint[iarrow] = [-2.83, 6.80];  endpoint[iarrow] = [-2.83, 6.80+dy];  iarrow += 1
    startpoint[iarrow] = [-2.71, 6.95];  endpoint[iarrow] = [-2.71, 6.95+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.98, 6.40];  endpoint[iarrow] = [-2.98, 6.40+dy];  iarrow += 1
    startpoint[iarrow] = [-3.32, 6.75];  endpoint[iarrow] = [-3.32, 6.75+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.71, 6.80];  endpoint[iarrow] = [-2.71, 6.80+dy];  iarrow += 1
    startpoint[iarrow] = [-2.69, 7.00];  endpoint[iarrow] = [-2.69, 7.00+dy];  iarrow += 1
    startpoint[iarrow] = [-3.15, 6.95];  endpoint[iarrow] = [-3.15, 6.95+dy];  iarrow += 1  
    startpoint[iarrow] = [-3.10, 6.40];  endpoint[iarrow] = [-3.10, 6.40+dy];  iarrow += 1
    startpoint[iarrow] = [-2.67, 7.08];  endpoint[iarrow] = [-2.67, 7.08+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.64, 7.50];  endpoint[iarrow] = [-2.64, 7.50+dy];  iarrow += 1
    startpoint[iarrow] = [-2.71, 8.35];  endpoint[iarrow] = [-2.71, 8.35+dy];  iarrow += 1  
    startpoint[iarrow] = [-2.78, 6.60];  endpoint[iarrow] = [-2.78, 6.60+dy];  iarrow += 1
    startpoint[iarrow] = [-3.38, 6.20];  endpoint[iarrow] = [-3.38, 6.20+dy];  iarrow += 1
    startpoint[iarrow] = [-2.61, 6.75];  endpoint[iarrow] = [-2.61, 6.75+dy];  iarrow += 1
    startpoint[iarrow] = [-2.81, 6.32];  endpoint[iarrow] = [-2.81, 6.32+dy];  iarrow += 1  
    startpoint[iarrow] = [-3.87, 6.56];  endpoint[iarrow] = [-3.87, 6.56+dy];  iarrow += 1  
    startpoint[iarrow] = [-3.00, 7.43];  endpoint[iarrow] = [-3.00, 7.43+dy];  iarrow += 1
    startpoint[iarrow] = [-4.15, 5.98];  endpoint[iarrow] = [-4.15, 5.98+dy];  iarrow += 1
    startpoint[iarrow] = [-3.94, 5.30];  endpoint[iarrow] = [-3.94, 5.30+dy];  iarrow += 1
    startpoint[iarrow] = [-3.54, 6.30];  endpoint[iarrow] = [-3.54, 6.30+dy];  iarrow += 1
    startpoint[iarrow] = [-3.86, 7.34];  endpoint[iarrow] = [-3.86, 7.34+dy];  iarrow += 1
    startpoint[iarrow] = [-3.96, 6.47];  endpoint[iarrow] = [-3.96, 6.47+dy];  iarrow += 1
    startpoint[iarrow] = [-3.73, 6.70];  endpoint[iarrow] = [-3.73, 6.70+dy];  iarrow += 1
    startpoint[iarrow] = [-2.93, 6.80];  endpoint[iarrow] = [-2.93, 6.80+dy];  iarrow += 1
    startpoint[iarrow] = [-3.50, 5.60];  endpoint[iarrow] = [-3.50, 5.60+dy];  iarrow += 1
    startpoint[iarrow] = [-3.45, 5.80];  endpoint[iarrow] = [-3.45, 5.80+dy];  iarrow += 1
    startpoint[iarrow] = [-7.30, 6.03];  endpoint[iarrow] = [-7.30+dx, 6.03];  iarrow += 1
    startpoint[iarrow] = [-3.79, 7.70];  endpoint[iarrow] = [-3.79+dx, 7.70];  iarrow += 1
    startpoint[iarrow] = [-5.05, 6.90];  endpoint[iarrow] = [-5.05+dx, 6.90];  iarrow += 1

    # THEORETICAL MODEL
    Fe_C_cr_file = "/home/genchiaki/dust/A_C/data_Fe_C_cr.dat"
    Fe_C_cr_data = np.loadtxt(Fe_C_cr_file)

    # PLOT
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.tick_params(labelsize=fontsize_tick_s)
    ax.set_xlabel(r"[Fe/H]"      , fontsize=fontsize_label_s)
    ax.set_ylabel(r"$A({\rm C})$", fontsize=fontsize_label_s)
    ax.set_xlim([-10, -2])
    ax.set_ylim([2.5, 9.5])
    ax.set_xticks(np.linspace(-10, -2, 5))
    ax.set_xticks(np.linspace(-10, -2, 9), minor=True)
    ax.set_yticks(np.linspace(  4,  8, 3))
    ax.set_yticks(np.linspace(  3,  9, 7), minor=True)
    ax.annotate(r'Silicate dominant'       , xy=(-3.3, 3.2), xycoords='data', color='#0000ff', fontsize=fontsize_legend_s, va='center' , ha='center', zorder=4)
    ax.annotate(r'Carbon grain dominant'   , xy=(-7.5, 7.0), xycoords='data', color='#cc0000', fontsize=fontsize_legend_s, va='center' , ha='center', zorder=4)
    ax.annotate(r'Ineffective dust cooling', xy=(-7.5, 4.2), xycoords='data', color='#000000', fontsize=fontsize_legend_s, va='center' , ha='center', zorder=4)
    ax.annotate(r'[C/Fe] = 0.7', xy=(-5.9, 3.5), xycoords='data', rotation=40, color='#000000', fontsize=fontsize_legend_s, va='center' , ha='center', zorder=4)
    ax.annotate(r'[C/Fe] = 2.3', xy=(-4.8, 6.2), xycoords='data', rotation=40, color='#000000', fontsize=fontsize_legend_s, va='center' , ha='center', zorder=4)
    ax.annotate(r'$10^{\rm [C/H]-2.30} + 10^{\rm [Fe/H]} = 10^{-5.07}$', xy=(-7.8, 5.4), xycoords='data', color='#000000', fontsize=fontsize_legend_s, va='center' , ha='center', zorder=4)
    # OBSERVATIONAL DATA
    for iarrow in range(0,119):
        ax.annotate('', xy=endpoint[iarrow], xytext=startpoint[iarrow],
                    arrowprops=dict(shrink=0, width=0.5, headwidth=5, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='#8888ff', edgecolor='#8888ff')
                   , zorder = 2)
    for iarrow in range(119,narrow):
        ax.annotate('', xy=endpoint[iarrow], xytext=startpoint[iarrow],
                    arrowprops=dict(shrink=0, width=0.5, headwidth=5, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='#ff8888', edgecolor='#ff8888')
                   , zorder = 2)
    errb = ax.errorbar(abdata[ 0][:, i_Fe_H], abdata[ 0][:, i_A_C], xerr = abdata[ 0][:, i_dmFe_H], yerr = abdata[ 0][:, i_dmA_C], capsize=1, fmt='o', markersize=2, ecolor='#8888ff', markeredgecolor = "#8888ff", color='#8888ff', zorder = 1)
    errb = ax.errorbar(abdata[ 1][:, i_Fe_H], abdata[ 1][:, i_A_C], xerr = abdata[ 1][:, i_dmFe_H], yerr = abdata[ 1][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#8888ff', markeredgecolor = "#8888ff", color='#8888ff', zorder = 1)
    errb = ax.errorbar(abdata[ 2][:, i_Fe_H], abdata[ 2][:, i_A_C], xerr = abdata[ 2][:, i_dmFe_H], yerr = abdata[ 2][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#8888ff', markeredgecolor = "#8888ff", color='#8888ff', zorder = 1)
    errb = ax.errorbar(abdata[ 3][:, i_Fe_H], abdata[ 3][:, i_A_C], xerr = abdata[ 3][:, i_dmFe_H], yerr = abdata[ 3][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2)
    errb = ax.errorbar(abdata[ 4][:, i_Fe_H], abdata[ 4][:, i_A_C], xerr = abdata[ 4][:, i_dmFe_H], yerr = abdata[ 4][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2)
    errb = ax.errorbar(abdata[ 5][:, i_Fe_H], abdata[ 5][:, i_A_C], xerr = abdata[ 5][:, i_dmFe_H], yerr = abdata[ 5][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2)
    errb = ax.errorbar(abdata[ 6][:, i_Fe_H], abdata[ 6][:, i_A_C], xerr = abdata[ 6][:, i_dmFe_H], yerr = abdata[ 6][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2)
    errb = ax.errorbar(abdata[ 7][:, i_Fe_H], abdata[ 7][:, i_A_C], xerr = abdata[ 7][:, i_dmFe_H], yerr = abdata[ 7][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#88cc88', markeredgecolor = "#88cc88", color='#88cc88', zorder = 2)
    errb = ax.errorbar(abdata[ 8][:, i_Fe_H], abdata[ 8][:, i_A_C], xerr = abdata[ 8][:, i_dmFe_H], yerr = abdata[ 8][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#88cc88', markeredgecolor = "#88cc88", color='#88cc88', zorder = 2)
    errb = ax.errorbar(abdata[ 9][:, i_Fe_H], abdata[ 9][:, i_A_C], xerr = abdata[ 9][:, i_dmFe_H], yerr = abdata[ 9][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2)
    errb = ax.errorbar(abdata[10][:, i_Fe_H], abdata[10][:, i_A_C], xerr = abdata[10][:, i_dmFe_H], yerr = abdata[10][:, i_dmA_C], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2)
##  errb = ax.errorbar([-11.0], [0.0], xerr = [0.0], yerr = [0.0], capsize=1, fmt='o', markersize=2, ecolor='#8888ff', markeredgecolor = "#8888ff", color='#8888ff', zorder = 1, label='C-normal')
##  errb = ax.errorbar([-11.0], [0.0], xerr = [0.0], yerr = [0.0], capsize=2, fmt='o', markersize=3, ecolor='#888888', markeredgecolor = "#888888", color='#888888', zorder = 2, label='CEMP')
##  errb = ax.errorbar([-11.0], [0.0], xerr = [0.0], yerr = [0.0], capsize=2, fmt='o', markersize=3, ecolor='#88cc88', markeredgecolor = "#88cc88", color='#88cc88', zorder = 2, label='CEMP-s')
##  errb = ax.errorbar([-11.0], [0.0], xerr = [0.0], yerr = [0.0], capsize=2, fmt='o', markersize=3, ecolor='#ff8888', markeredgecolor = "#ff8888", color='#ff8888', zorder = 2, label='CEMP-no')
    # THEORETICAL MODEL (Chiaki et al. 2017)
    fill = ax.fill_between([-10, -2],[9.5, 9.5], [8.43-10.0+2.3, 8.43-2.0+2.3],facecolor='#ffeeee',alpha=1.0, zorder=0)
    fill = ax.fill_between([-10, -2], [8.43-10.0+2.3, 8.43-2.0+2.3],[2.5, 2.5],facecolor='#eeeeff',alpha=1.0, zorder=0)
    fill = ax.fill_between(Fe_C_cr_data[:,1], Fe_C_cr_data[:,2], [2.5]*500,facecolor='#eeeeee',alpha=1.0, zorder=1)
    plot = ax.plot(
       [-10.0, -2.0], [8.43-10.0+0.7, 8.43-2.0+0.7]
     , [-5.38, -3.0], [8.43-5.38+2.3, 8.43-3.0+2.3]
     , Fe_C_cr_data[:,1], Fe_C_cr_data[:,2]
     , zorder = 3 )
    # SIMULATION RESULTS
##  Fe_H_no = -3.42
##  C_H_no = Fe_H_no + 0.18
##  A_C_no = C_H_no + 8.43
    A_C_no  = 12.0 + np.log10(2.65314e-01*2.6e-4*SolarMetalFractionByMass/12.0/HydrogenFractionByMass)
    Fe_H_no = 12.0 + np.log10(9.62448e-02*2.6e-4*SolarMetalFractionByMass/56.0/HydrogenFractionByMass) - SolarIronAbundance
    print("C13 A(C) %7.2f [Fe/H] %7.2f" % (A_C_no, Fe_H_no))
    scat = ax.scatter(Fe_H_no, A_C_no, s=100.0, facecolor='black' , edgecolors='black' , marker='s', label=r'Normal $13$ M$_{\bigodot}$ (Paper I)', zorder=9)
    scat = ax.scatter(Fe_H[0], A_C[0], s=100.0, facecolor='orange', edgecolors='orange', marker='D', label=r'Faint $13$ M$_{\bigodot}$ (This work)', zorder=9)
    scat = ax.scatter(Fe_H[1], A_C[1], s=100.0, facecolor='None'  , edgecolors='purple', marker='^', label=r'Faint $50$ M$_{\bigodot}$ (This work)', zorder=9)
    scat = ax.scatter(Fe_H[2], A_C[2], s=100.0, facecolor='None'  , edgecolors='green' , marker='o', label=r'Faint $80$ M$_{\bigodot}$ (This work)', zorder=9)
    plot[0].set_linestyle("--");  plot[0].set_color("black");  plot[0].set_linewidth(2)
    plot[1].set_linestyle("-");   plot[1].set_color("black");  plot[1].set_linewidth(3)
    plot[2].set_linestyle("-");   plot[2].set_color("black");  plot[2].set_linewidth(3)
    plt.legend(fontsize=fontsize_legend_s, labelspacing=0.2, loc='upper left')
    plotfile = outdir + '/C_Fe'
    fig.savefig("%s.pdf" % plotfile)


if SNAPSHOTS_INI:
    reso = 500

    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M80"
    pdir = indir + '/Projection_z'
    infile = pdir + '/0079_Hydrogen_number_density_density.dat'
    infp = open(infile, 'rb')
    dens = np.fromfile(infp, dtype='d',sep='')
##  print(dens)
    infile = pdir + '/0079_temperature_density.dat'
    infp = open(infile, 'rb')
    temp = np.fromfile(infp, dtype='d',sep='')
##  print(temp)
    infile = pdir + '/0079_y_HII_density.dat'
    infp = open(infile, 'rb')
    yHII = np.fromfile(infp, dtype='d',sep='')
##  print(yHII)
    infile = pdir + '/0079_y_H2I_density.dat'
    infp = open(infile, 'rb')
    yH2I = np.fromfile(infp, dtype='d',sep='')
##  print(yH2I)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6.0))
    #fig.suptitle(eng, fontsize=fontsize_suptitle, fontweight='bold')

    ibox = '0'; s_box = 2.0; boxsize = '2 kpc'

    Y, X = np.mgrid[0:reso+1, 0:reso+1]
    X = X / float(reso) * s_box
    Y = Y / float(reso) * s_box
    X = X - 0.5 * s_box
    Y = Y - 0.5 * s_box
    #print(X)

    for ivar in range(3):

        if ivar==0: Z = np.log10(dens).reshape(reso,reso)
        if ivar==1: Z = np.log10(temp).reshape(reso,reso)
#       if ivar==2: Z = np.log10(yHII).reshape(reso,reso)
        if ivar==2: Z = np.log10(yH2I).reshape(reso,reso)

        ax = axs[ivar]
        if ivar==0: cmap = 'bds_highcontrast'
        if ivar==1: cmap = 'hot'
        if ivar==2: cmap = 'BLUE'
        pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelbottom=False,
                       labelleft=False,
                       labelright=False,
                       labeltop=False)
        ax.tick_params(bottom=False,
                       left=False,
                       right=False,
                       top=False)
        if ivar==2:
            ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                      , color='white', fontsize=fontsize_boxsize, rotation=90
                      , va='center', ha='center'
        #             , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                          )
            ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                          , xytext=(0.93, 0.4), textcoords=ax.transAxes
                          , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                           headlength=10, connectionstyle='arc3',
                                           facecolor='white', edgecolor='white')
                          )
            ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                          , xytext=(0.93, 0.6), textcoords=ax.transAxes
                          , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                           headlength=10, connectionstyle='arc3',
                                           facecolor='white', edgecolor='white')
                          )

        if ivar==0:
            cmin = min(np.log10(dens)); cmax = max(np.log10(dens)); 
        #   if cmin < 1.5: cmin = 1.5 
            pcolor.set_clim(cmin, cmax)
            ticks=np.linspace(-3, 3, 7)
        if ivar==1:
            cmin = min(np.log10(temp)); cmax = max(np.log10(temp)); 
            if cmin < 1.5: cmin = 1.5 
            pcolor.set_clim(cmin, cmax)
            ticks=np.linspace( 1, 6, 11)
        if ivar==2:
            cmin = min(np.log10(yH2I)); cmax = max(np.log10(yH2I)); 
        #   if cmin < 1.5: cmin = 1.5 
            pcolor.set_clim(cmin, cmax)
            ticks=np.linspace(-5,-2, 7)
#       if ivar==0: ticks=np.linspace(-2, 3, 6)
#       if ivar==1: ticks=np.linspace(1, 3, 3)
##      if ivar==2: ticks=np.linspace(-4.5, -3.5, 3)
#       if ivar==2: ticks=np.linspace(-4.0, -3.0, 3)
        if ivar==0: xccolor='white'
        if ivar==1: xccolor='white'
#       if ivar==2: xccolor='white'
        if ivar==2: xccolor='white'
        axins = inset_axes(ax,
                           width="50%",  # width = 5% of parent_bbox width
                           height="5%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(0.03, 0.15, 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                          )
        colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#              ,aspect=10,pad=-0.22 ,shrink=0.50
               ,ticks=ticks
                  )
        colbar.outline.set_edgecolor(xccolor)
        if ivar==0: xclabel = r'log [ Density / cm$^{-3}$ ]'
        if ivar==1: xclabel = r'log [ Temperature / K ]'
#       if ivar==2: xclabel = r'log [ $y({\rm H^{+}})$ ]'
        if ivar==2: xclabel = r'log [ $y({\rm H_{2}})$ ]'
        colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
        colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)

    plt.subplots_adjust(left   = 0.00
                      , right  = 1.00
                      , bottom = 0.00
                      , top    = 1.00
                      , wspace = 0.0
                      , hspace = 0.0
                       )


    fig.savefig("%s/snapshots_ini.png" % outdir)
##  print(("%s/snapshots_ini.png" % outdir))

    plt.close('all')
 

if SNAPSHOTS_RAD:
    reso = 500
    progmass = [
       r"(a) $M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"(b) $M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"(c) $M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
      ]

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18.0, 19.5))
    #fig.suptitle(eng, fontsize=fontsize_suptitle, fontweight='bold')

    for iSN in range(3):
        if iSN==0: SN = 'M13'; inc = 96; xtime = r'$t_{\rm UV} = 11.78$ Myr'
        if iSN==1: SN = 'M50'; inc = 86; xtime = r'$t_{\rm UV} = 1.87$ Myr'
        if iSN==2: SN = 'M80'; inc = 86; xtime = r'$t_{\rm UV} = 2.76$ Myr'
       
        indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_" + SN
        pdir = indir + '/Slice_z'
        infile = pdir + ('/%04d_Hydrogen_number_density.dat' % inc)
        infp = open(infile, 'rb')
        dens = np.fromfile(infp, dtype='d',sep='')
#       print(dens)
        infile = pdir + ('/%04d_temperature.dat' % inc)
        infp = open(infile, 'rb')
        temp = np.fromfile(infp, dtype='d',sep='')
#       print(temp)
        infile = pdir + ('/%04d_y_HII.dat' % inc)
        infp = open(infile, 'rb')
        yHII = np.fromfile(infp, dtype='d',sep='')
#       print(yHII)
        infile = pdir + ('/%04d_y_H2I.dat' % inc)
        infp = open(infile, 'rb')
        yH2I = np.fromfile(infp, dtype='d',sep='')
#       print(yH2I)
      
        ibox = '0'; s_box = 1.0; boxsize = '1 kpc'
      
        Y, X = np.mgrid[0:reso+1, 0:reso+1]
        X = X / float(reso) * s_box
        Y = Y / float(reso) * s_box
        X = X - 0.5 * s_box
        Y = Y - 0.5 * s_box
        #print(X)
      
        for ivar in range(3):
      
            if ivar==0: Z = np.log10(dens).reshape(reso,reso)
            if ivar==1: Z = np.log10(temp).reshape(reso,reso)
            if ivar==2: Z = np.log10(yHII).reshape(reso,reso)
      
            ax = axs[iSN][ivar]
            if ivar==0: cmap = 'bds_highcontrast'
            if ivar==1: cmap = 'hot'
            if ivar==2: cmap = 'BLUE'
            pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelbottom=False,
                           labelleft=False,
                           labelright=False,
                           labeltop=False)
            ax.tick_params(bottom=False,
                           left=False,
                           right=False,
                           top=False)
            if ivar==0:
                ax.set_title(progmass[iSN], fontsize=fontsize_title, loc='left')
                ax.annotate(xtime, xy=(0.03, 0.97), xycoords=ax.transAxes
                          , color='white', fontsize=fontsize_title
                          , va='top', ha='left'
                            )
            if ivar==2:
                ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                          , color='white', fontsize=fontsize_boxsize, rotation=90
                          , va='center', ha='center'
            #             , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                              )
                ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                              , xytext=(0.93, 0.4), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                               headlength=10, connectionstyle='arc3',
                                               facecolor='white', edgecolor='white')
                              )
                ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                              , xytext=(0.93, 0.6), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                               headlength=10, connectionstyle='arc3',
                                               facecolor='white', edgecolor='white')
                              )
            if ivar==0: pcolor.set_clim(-3.0, 2.0); ticks=np.linspace(-3, 2, 6)
            if ivar==1: pcolor.set_clim( 1.0, 4.5); ticks=np.linspace( 1, 4, 4)
            if ivar==2: pcolor.set_clim(-4.0, 0.0); ticks=np.linspace(-4, 0, 5)
            if iSN==0:
              if ivar==0: xccolor='white'
              if ivar==1: xccolor='white'
              if ivar==2: xccolor='white'
            if iSN==1:
              if ivar==0: xccolor='white'
              if ivar==1: xccolor='black'
              if ivar==2: xccolor='white'
            if iSN==2:
              if ivar==0: xccolor='white'
              if ivar==1: xccolor='black'
              if ivar==2: xccolor='white'
            axins = inset_axes(ax,
                               width="50%",  # width = 5% of parent_bbox width
                               height="5%",  # height : 50%
                               loc='lower left',
                               bbox_to_anchor=(0.03, 0.15, 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,
                              )
            colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                  ,aspect=10,pad=-0.22 ,shrink=0.50
                   ,ticks=ticks
                      )
            colbar.outline.set_edgecolor(xccolor)
            if ivar==0: xclabel = r'log [ Density / cm$^{-3}$ ]'
            if ivar==1: xclabel = r'log [ Temperature / K ]'
            if ivar==2: xclabel = r'log [ $y({\rm H^{+}})$ ]'
            colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
            colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
     
    plt.subplots_adjust(left   = 0.00
                      , right  = 1.00
                      , bottom = 0.00
                      , top    = 0.98
                      , wspace = 0.0
                      , hspace = 0.03
                       )
    
    
    fig.savefig("%s/snapshots_rad.png" % outdir)
##  print(("%s/snapshots_rad.png" % outdir))
    
    plt.close('all')
 

if SNAPSHOTS_EXP:
    SLICE = True
    PROJ  = False
    PAPER = True

    if PAPER:
      ninc = 3

    reso = 500
    progmass = [
       r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$ (CCSN)"
      ]

    for iSN in range(4):
      if iSN==0: SN = 'M13';
      if iSN==1: SN = 'M50';
      if iSN==2: SN = 'M80';
      if iSN==3: SN = 'C13';
      if PAPER:
         fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18.0, 18.4))
         fig.suptitle(progmass[iSN], fontsize=fontsize_suptitle, fontweight='bold', y=0.99)

      if not PAPER:
        if iSN==0:
          ninc = 8
        if iSN==1:
          ninc = 1
        if iSN==2:
          ninc = 18
        if iSN==3:
          ninc = 14

      for iinc in range(ninc):
        if PAPER:
          if iSN==0:
            if iinc==0: inc = 97; xtime = r'(a) $t_{\rm SN} = 0.41$ Myr'
            if iinc==1: inc =101; xtime = r'(b) $t_{\rm SN} =15.22$ Myr'
            if iinc==2: inc =104; xtime = r'(c) $t_{\rm SN} =29.16$ Myr'
          if iSN==1:
            if iinc==0: inc = 87; xtime = r'(a) $t_{\rm SN} = 0.32$ Myr'
            if iinc==1: inc = 91; xtime = r'(b) $t_{\rm SN} =18.90$ Myr'
            if iinc==2: inc =102; xtime = r'(c) $t_{\rm SN} =70.02$ Myr'
          if iSN==2:
            if iinc==0: inc = 87; xtime = r'(a) $t_{\rm SN} = 0.77$ Myr'
            if iinc==1: inc = 91; xtime = r'(b) $t_{\rm SN} =19.36$ Myr'
            if iinc==2: inc =104; xtime = r'(c) $t_{\rm SN} =79.77$ Myr'
          if iSN==3:
            if iinc==0: inc = 87; xtime = r'(a) $t_{\rm SN} = 0.20$ Myr'
            if iinc==1: inc = 97; xtime = r'(b) $t_{\rm SN} =15.22$ Myr'
            if iinc==2: inc =100; xtime = r'(c) $t_{\rm SN} =29.16$ Myr'
        else:
          fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18.0, 6.2))
          if iSN < 0:
             progmass0 = progmass[0]
          else:
             progmass0 = progmass[iSN]
          fig.suptitle(progmass0, fontsize=fontsize_suptitle, fontweight='bold', y=0.99)
          if iSN==0:
            inc = 97 + iinc
          if iSN==1 or iSN==2:
            inc = 87 + iinc
          if iSN==3:
            inc = 87 + iinc
          xtime = ''
       
        if iSN < 3:
            indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_" + SN
        else:
            indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_cgrackle_HR2"
        if PROJ:  pdir = indir + '/Projection_z'
        if SLICE: pdir = indir + '/Slice_z'
        if PROJ:  extension = '_density.dat'
        if SLICE: extension = '.dat'

        fields = [
            'Hydrogen_number_density'
          , 'temperature_corr'
          , 'CarbonAbundance'
                 ]
       
        ibox = '0'; s_box = 2.0; boxsize = '2 kpc'
      
        Y, X = np.mgrid[0:reso+1, 0:reso+1]
        X = X / float(reso) * s_box
        Y = Y / float(reso) * s_box
        X = X - 0.5 * s_box
        Y = Y - 0.5 * s_box
        #print(X)
      
        for ivar in range(3):
      
#           if ivar==0: Z = np.log10(dens).reshape(reso,reso)
#           if ivar==1: Z = np.log10(temp).reshape(reso,reso)
#           if ivar==2: Z = np.log10(AC  ).reshape(reso,reso)
            infile = pdir + ('/%04d_%s%s' % (inc, fields[ivar], extension))
            infp = open(infile, 'rb')
            outdata = np.fromfile(infp, dtype='d',sep='')
#           print(outdata)
            Z = np.log10(outdata).reshape(reso,reso)

            if PAPER:
                ax = axs[ivar][iinc]
            else:
                ax = axs[ivar]
            if ivar==0: cmap = 'bds_highcontrast'
            if ivar==1: cmap = 'hot'
            if ivar==2: cmap = 'BLUE'
            pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelbottom=False,
                           labelleft=False,
                           labelright=False,
                           labeltop=False)
            ax.tick_params(bottom=False,
                           left=False,
                           right=False,
                           top=False)
            if ivar==0:
                ax.annotate(xtime, xy=(0.03, 0.97), xycoords=ax.transAxes
                          , color='white', fontsize=fontsize_title
                          , va='top', ha='left'
                            )
            if iSN==0 or iSN==3:
                if ivar==0: xccolor1='black'; xccolor2='white'; xccolor='white'
                if ivar==1: xccolor1='black'; xccolor2='white'; xccolor='white'
                if ivar==2: xccolor1='white'; xccolor2='white'; xccolor='white'
            if iSN==1 or iSN==2:
                if ivar==0: xccolor1='white'; xccolor2='white'; xccolor='white'
                if ivar==1: xccolor1='black'; xccolor2='black'; xccolor='black'
                if ivar==2: xccolor1='white'; xccolor2='white'; xccolor='white'
            if (PAPER and iinc==2) or (not PAPER):
                ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                          , color=xccolor1, fontsize=fontsize_boxsize, rotation=90
                          , va='center', ha='center'
            #             , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                              )
                ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                              , xytext=(0.93, 0.4), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                               headlength=10, connectionstyle='arc3',
                                               facecolor=xccolor2, edgecolor=xccolor2)
                              )
                ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                              , xytext=(0.93, 0.6), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                               headlength=10, connectionstyle='arc3',
                                               facecolor=xccolor2, edgecolor=xccolor2)
                              )
      ####  if iinc > 0:
      ####      ax.annotate('', xy=(0.5,1.00), xycoords=ax.transAxes, xytext=(0.5,1.04), textcoords=ax.transAxes
      ####                    , arrowprops=dict(shrink=0, width=30, headwidth=50, headlength=10, connectionstyle='arc3', facecolor='black', edgecolor='black') )
            if iinc < 2:
                ax.annotate('', xy=(1.03,0.5), xycoords=ax.transAxes, xytext=(1.00,0.5), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=20, headwidth=40, headlength=8, connectionstyle='arc3', facecolor='black', edgecolor='black') )
#           if iinc < 2:
#               ax.annotate('', xy=(0.5,-0.02), xycoords=ax.transAxes, xytext=(0.5,0.02), textcoords=ax.transAxes
#                             , arrowprops=dict(shrink=0, width=15, headwidth=30, headlength=15, connectionstyle='arc3', facecolor='pink', edgecolor='pink') )
####        cmin = min(np.log10(outdata)); cmax = max(np.log10(outdata)); 
            cmin = Z.min(); cmax = Z.max(); 
            if ivar==0: 
                if cmin < -4.0: cmin = -4.0 
                if not PAPER:
                   cmin = -3.5; cmax = 2.0
                pcolor.set_clim(cmin, cmax)
                ticks=np.linspace(-4, 2, 7)
            if ivar==1:
                if cmin < 0.5: cmin = 0.5 
                pcolor.set_clim(cmin, cmax)
                ticks=np.linspace( 1, 6, 6)
            if ivar==2:
                if cmin < 0.0: cmin = 0.0 
                pcolor.set_clim(cmin, cmax)
                ticks=np.linspace( 0, 10, 6)
            axins = inset_axes(ax,
                               width="50%",  # width = 5% of parent_bbox width
                               height="5%",  # height : 50%
                               loc='lower left',
                               bbox_to_anchor=(0.03, 0.15, 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,
                              )
            colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                  ,aspect=10,pad=-0.22 ,shrink=0.50
                   ,ticks=ticks
                      )
            colbar.outline.set_edgecolor(xccolor)
            if ivar==0: xclabel = r'log [ Density / cm$^{-3}$ ]'
            if ivar==1: xclabel = r'log [ Temperature / K ]'
            if ivar==2: xclabel = r'$A({\rm C})$'
            colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
            colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
     
        if not PAPER:
          plt.subplots_adjust(left   = 0.00
                            , right  = 1.00
                            , bottom = 0.00
                            , top    = 0.94
                            , wspace = 0.03
                            , hspace = 0.00
                             )
          fig.savefig("%s/snapshots_exp_%04d.png" % (pdir, inc))

      if PAPER:
          plt.subplots_adjust(left   = 0.00
                            , right  = 1.00
                            , bottom = 0.00
                            , top    = 0.96
                            , wspace = 0.03
                            , hspace = 0.00
                             )
      
          fig.savefig("%s/snapshots_exp_%s.png" % (outdir, SN))
      
      plt.close('all')
 


if SNAPSHOTS_COL:
    reso = 500
    progmass = [
       r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"$M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
      ]


    for iSN in range(3):
        if iSN==0: SN = 'M13';
        if iSN==1: SN = 'M50';
        if iSN==2: SN = 'M80';

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18.0, 12.5))
        fig.suptitle(progmass[iSN], fontsize=fontsize_suptitle, fontweight='bold')

        for iinc in range(3):
            if iSN==0:
                if iinc==0: inc = 123; xtime=r'(a) $t_{\rm SN}= 86.608585$ Myr'; xnHmax=r'$n_{\rm H,max}=2.69 \times 10^{ 7}$ cm$^{-3}$'; s_box=7.5  ;  boxsize = '7.5 pc' 
                if iinc==1: inc = 130; xtime=r'(b) $t_{\rm SN}= 86.628300$ Myr'; xnHmax=r'$n_{\rm H,max}=1.25 \times 10^{12}$ cm$^{-3}$'; s_box=0.05 ;  boxsize = '0.05 pc'
                if iinc==2: inc = 137; xtime=r'(c) $t_{\rm SN}= 86.628362$ Myr'; xnHmax=r'$n_{\rm H,max}=2.22 \times 10^{16}$ cm$^{-3}$'; s_box= 50.0;  boxsize = '50 au'
            if iSN==1:
                if iinc==0: inc = 161; xtime=r'(a) $t_{\rm SN}=303.685840$ Myr'; xnHmax=r'$n_{\rm H,max}=3.39 \times 10^{ 7}$ cm$^{-3}$'; s_box=1.5  ;  boxsize = '1.5 pc'
                if iinc==1: inc = 167; xtime=r'(b) $t_{\rm SN}=303.724321$ Myr'; xnHmax=r'$n_{\rm H,max}=1.35 \times 10^{12}$ cm$^{-3}$'; s_box=0.05 ;  boxsize = '0.05 pc'
                if iinc==2: inc = 174; xtime=r'(c) $t_{\rm SN}=303.724593$ Myr'; xnHmax=r'$n_{\rm H,max}=3.49 \times 10^{16}$ cm$^{-3}$'; s_box=50.0 ;  boxsize = '50 au'
            if iSN==2:
                if iinc==0: inc = 217; xtime=r'(a) $t_{\rm SN}=564.408802$ Myr'; xnHmax=r'$n_{\rm H,max}=1.64 \times 10^{ 7}$ cm$^{-3}$'; s_box=1.5  ;  boxsize = '1.5 pc'
                if iinc==1: inc = 223; xtime=r'(b) $t_{\rm SN}=564.445656$ Myr'; xnHmax=r'$n_{\rm H,max}=9.15 \times 10^{11}$ cm$^{-3}$'; s_box=0.05 ;  boxsize = '0.05 pc'
                if iinc==2: inc = 230; xtime=r'(c) $t_{\rm SN}=564.445741$ Myr'; xnHmax=r'$n_{\rm H,max}=1.79 \times 10^{16}$ cm$^{-3}$'; s_box=50.0 ;  boxsize = '50 au'

            if iinc==2:
                if iSN==0:
                    # c20_M1e-3_n1e13
                    cen_ps = [[0.0]*3]*5; rad_ps = [0.0]*5
                    cen_ps[ 0] = [    0.0000000,     0.0000000,     0.0000000] #     0.3551108   4.09601e+13
                    cen_ps[ 1] = [    4.2889306,   -17.9492171,     2.3386352] #     0.0105663   2.04820e+13
                    cen_ps[ 2] = [   -8.1535228,    -9.6583518,    12.1273509] #     0.0023478   2.04845e+13
                    cen_ps[ 3] = [    1.8179823,    16.6711809,    -1.9564724] #     0.0015482   2.04958e+13
                    cen_ps[ 4] = [    4.4804749,     8.7195604,     4.7824769] #     0.0012171   4.09717e+13
                    rad_ps[ 0] = 1.1656430 #     4.20708e-02
                    rad_ps[ 1] = 0.0000000 #     0.0000000
                    rad_ps[ 2] = 0.0000000 #     0.0000000
                    rad_ps[ 3] = 0.0000000 #     0.0000000
                    rad_ps[ 4] = 0.0000000 #     0.0000000
                if iSN==1:
                    cen_ps = [[0.0]*3]*1; rad_ps = [0.0]*1
                    cen_ps[ 0] = [    0.0000000,     0.0000000,     0.0000000]
                    rad_ps[ 0] = 1.6199056 # au 0.0294296 Msun
                if iSN==2:
                    cen_ps = [[0.0]*3]*1; rad_ps = [0.0]*1
                    cen_ps[ 0] = [    0.0000000,     0.0000000,     0.0000000]
                    rad_ps[ 0] = 0.8681980 # au 0.0137835 Msun
       
            for ivar in range(2):
           
                indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_" + SN
                pdir = indir + '/Projection_x'
                if ivar==0: var = "Hydrogen_number_density"; cmap = 'bds_highcontrast'
                if ivar==1: var = "temperature_corr"       ; cmap = 'hot'
                infile = pdir + ('/%04d_%s_density.dat' % (inc, var))
                infp = open(infile, 'rb')
                dens = np.fromfile(infp, dtype='d',sep='')
#               print(dens)

                Y, X = np.mgrid[0:reso+1, 0:reso+1]
                X = X / float(reso) * s_box
                Y = Y / float(reso) * s_box
                X = X - 0.5 * s_box
                Y = Y - 0.5 * s_box
                #print(X)
          
                if ivar==0:
                    Z = np.log10(dens).reshape(reso,reso)
                if ivar==1:
                    Z =          dens .reshape(reso,reso)
          
                ax = axs[ivar][iinc]
                pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)
                ax.set_aspect('equal', adjustable='box')
                # CLUMP
                if iinc==2 and ivar==0:
                    for i_ps in range(len(rad_ps)):
                        cen0 = cen_ps[i_ps][0]; cen1 = cen_ps[i_ps][1]
                        rad = rad_ps[i_ps]
                        if rad > 0.0:
                            ps1_X = []; ps1_Y = []
                            for j in np.linspace(0, 2 * np.pi, 1000):
                                  ps1_X.append(cen0 + rad * math.sin(j))
                                  ps1_Y.append(cen1 + rad * math.cos(j))
                            pplot  = ax.plot(ps1_X, ps1_Y, color='black' , lw=2)
                        else:
                            scat = ax.scatter(cen0, cen1, s=150.0, color='white'  , marker='+', linewidths=2)
                ax.tick_params(labelbottom=False,
                               labelleft=False,
                               labelright=False,
                               labeltop=False)
                ax.tick_params(bottom=False,
                               left=False,
                               right=False,
                               top=False)
                if ivar==0: xccolor = 'white'
                if ivar==1:
                     if iSN==1:
                        if iinc>=1: xccolor = 'white'
                        else:       xccolor = 'black'
                     else:
                        if iinc< 2: xccolor = 'white'
                        else:       xccolor = 'black'
                if ivar==0:
                    ax.annotate(xtime , xy=(0.03, 0.97), xycoords=ax.transAxes
                              , color=xccolor, fontsize=fontsize_title
                              , va='top', ha='left'
                                )
                    ax.annotate(xnHmax, xy=(0.03, 0.90), xycoords=ax.transAxes
                              , color=xccolor, fontsize=fontsize_title
                              , va='top', ha='left'
                                )
                ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                          , color=xccolor, fontsize=fontsize_boxsize, rotation=90
                          , va='center', ha='center'
                #         , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                              )
                ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                              , xytext=(0.93, 0.4), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                               headlength=10, connectionstyle='arc3',
                                               facecolor=xccolor, edgecolor=xccolor)
                              )
                ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                              , xytext=(0.93, 0.6), textcoords=ax.transAxes
                              , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                               headlength=10, connectionstyle='arc3',
                                               facecolor=xccolor, edgecolor=xccolor)
                              )
                if iinc < 2:
                    ax.annotate('', xy=(1.03,0.5), xycoords=ax.transAxes, xytext=(1.00,0.5), textcoords=ax.transAxes
                                  , arrowprops=dict(shrink=0, width=20, headwidth=40, headlength=8, connectionstyle='arc3', facecolor='black', edgecolor='black') )
                if ivar==0: ticks=np.linspace(0,16,17)
                if ivar==1: 
           ##       if   iSN>=1 and iinc==1: ticks=np.linspace(0.0,4.0,  9)
           ##       elif iSN==1 and iinc==2: ticks=np.linspace(0.0,4.0, 41)
           ##       else                   : ticks=np.linspace(0.0,4.0, 21)
                    if iSN==0:
                        if iinc==0: ticks=np.linspace( 200.0, 600.0, 3)
                        if iinc==1: ticks=np.linspace( 500.0,1500.0, 3)
                        if iinc==2: ticks=np.linspace(1000.0,2000.0, 3)
                    else:
                        if iinc==0: ticks=np.linspace( 50.0, 150.0, 3)
                        if iinc==1: ticks=np.linspace(200.0, 800.0, 4)
                        if iinc==2: ticks=np.linspace(600.0,1200.0, 4)
                axins = inset_axes(ax,
                                   width="50%",  # width = 5% of parent_bbox width
                                   height="5%",  # height : 50%
                                   loc='lower left',
                                   bbox_to_anchor=(0.03, 0.15, 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                  )
                colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                      ,aspect=10,pad=-0.22 ,shrink=0.50
                       ,ticks=ticks
                          )
                colbar.outline.set_edgecolor(xccolor)
                if ivar==0: xclabel = r'log [ Density / cm$^{-3}$ ]'
          ##    if ivar==1: xclabel = r'log [ Temperature / K ]'
                if ivar==1: xclabel = r'Temperature [ K ]'
             #  if ivar==1 and idir==1: xclabel = r'log [ $y({\rm AC})$ ]'
                colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
                colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
         
        plt.subplots_adjust(left   = 0.00
                          , right  = 1.00
                          , bottom = 0.00
                          , top    = 0.94
                          , wspace = 0.03
                          , hspace = 0.00
                           )
        
        
        fig.savefig("%s/snapshots_col_%s.png" % (outdir, SN))
        
        plt.close('all')


if SNAPSHOTS_FIN:
    reso = 500
    progmass = [
       r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
      ]

    for iSN in range(1):
        if iSN==0: SN = 'M50';

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18.0, 12.5))
        fig.suptitle(progmass[iSN], fontsize=fontsize_suptitle, fontweight='bold')

        indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_" + SN
        inc = 174; xtime=r'$t_{\rm SN}= 303.724593$ Myr'; xnHmax=r'$n_{\rm H,max}=3.49 \times 10^{16}$ cm$^{-3}$'; s_box=50.0; boxsize = '50 au'

        ps1_cen = [0.0, 0.0, 0.0]
        ps1_rad = 1.6199056 # au 0.0294296 Msun
        ps1_X = []; ps1_Y = []
        for j in np.linspace(0, 2 * np.pi, 1000):
              ps1_X.append(ps1_cen[0] + ps1_rad * math.sin(j))
              ps1_Y.append(ps1_cen[1] + ps1_rad * math.cos(j))

        for idir in range(2):

            if idir==0: pdir = indir + '/Projection_x'; direc = 'Face-on'
            if idir==1: pdir = indir + '/Projection_z'; direc = 'Edge-on'
       
            for ivar in range(3):
           
                if ivar==0: var = "Hydrogen_number_density_density"; cmap = 'bds_highcontrast'
                if ivar==1: var = "temperature_corr_density"       ; cmap = 'hot'
                if ivar==2: var = "Q"                              ; cmap = 'coolwarm'
                infile = pdir + ('/%04d_%s.dat' % (inc, var))
                infp = open(infile, 'rb')
                dens = np.fromfile(infp, dtype='d',sep='')
#               print(dens)

                Y, X = np.mgrid[0:reso+1, 0:reso+1]
                X = X / float(reso) * s_box
                Y = Y / float(reso) * s_box
                X = X - 0.5 * s_box
                Y = Y - 0.5 * s_box
                #print(X)
          
                if ivar < 1:
                    Z = np.log10(dens).reshape(reso,reso)
                else:
                    Z = dens.reshape(reso,reso)
          
                ax = axs[idir][ivar]
                if ivar < 2:
                    pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)
                else:
                    pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=LogNorm(vmin=0.06, vmax=6.0))
                # CLUMP
                if ivar==0:
                    pplot  = ax.plot(ps1_X, ps1_Y, color='black' , lw=2)
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(labelbottom=False,
                               labelleft=False,
                               labelright=False,
                               labeltop=False)
                ax.tick_params(bottom=False,
                               left=False,
                               right=False,
                               top=False)
     #          if idir==0:
     #              ax.set_title(progmass[iSN], fontsize=fontsize_title, loc='left')
                xccolor = 'white'
              # if ivar==1 and idir==1: xccolor = 'white'
                ax.annotate(direc, xy=(0.03, 0.97), xycoords=ax.transAxes
                          , color=xccolor, fontsize=fontsize_title
                          , va='top', ha='left'
                            )
                if ivar==0 and idir==0:
                    ax.annotate(xtime , xy=(0.03, 0.90), xycoords=ax.transAxes
                              , color=xccolor, fontsize=fontsize_title
                              , va='top', ha='left'
                                )
                    ax.annotate(xnHmax, xy=(0.03, 0.83), xycoords=ax.transAxes
                              , color=xccolor, fontsize=fontsize_title
                              , va='top', ha='left'
                                )
                if ivar==2:
                    ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                              , color=xccolor, fontsize=fontsize_boxsize, rotation=90
                              , va='center', ha='center'
                    #         , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                                  )
                    ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                                  , xytext=(0.93, 0.4), textcoords=ax.transAxes
                                  , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                                   headlength=10, connectionstyle='arc3',
                                                   facecolor=xccolor, edgecolor=xccolor)
                                  )
                    ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                                  , xytext=(0.93, 0.6), textcoords=ax.transAxes
                                  , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                                   headlength=10, connectionstyle='arc3',
                                                   facecolor=xccolor, edgecolor=xccolor)
                                  )
                if ivar==0: ticks=np.linspace(11,16, 6)
                if ivar==1: ticks=np.linspace(800.0,1600.0, 5)
                if ivar==2:
                    ticks = np.array([0.06, 0.6, 6.0])
                axins = inset_axes(ax,
                                   width="50%",  # width = 5% of parent_bbox width
                                   height="5%",  # height : 50%
                                   loc='lower left',
                                   bbox_to_anchor=(0.06, 0.15, 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                  )
                colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                      ,aspect=10,pad=-0.22 ,shrink=0.50
                       ,ticks=ticks
                          )
                colbar.outline.set_edgecolor(xccolor)
                if ivar==0: xclabel = r'log [ Density / cm$^{-3}$ ]'
                if ivar==1: xclabel = r'Temperature [ K ]'
                if ivar==2: xclabel = r'$Q$'
                colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
                if ivar==2: colbar.ax.set_xticklabels(['0.06', '0.6', '6'])
                colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
         
        plt.subplots_adjust(left   = 0.00
                          , right  = 1.00
                          , bottom = 0.00
                          , top    = 0.94
                          , wspace = 0.03
                          , hspace = 0.00
                           )
        
        
        fig.savefig("%s/snapshots_fin_%s.png" % (outdir, SN))
        
        plt.close('all')
 

 
if PROFILE_METAL:
    reso = 128
    progmass = [
       r"(a) $M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$"
     , r"(b) $M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$"
     , r"(c) $M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$"
      ]

    nvar = 1

    if nvar==1: ysize =  6.5
    if nvar==2: ysize = 12.0

    fig, axs = plt.subplots(nrows=nvar, ncols=3, figsize=(18.0, ysize), sharex='all')
    #fig.suptitle(eng, fontsize=fontsize_suptitle, fontweight='bold')

    for iSN in range(3):
        if iSN==0: SN = 'M13'; inc =137;
        if iSN==1: SN = 'M50'; inc =174;
        if iSN==2: SN = 'M80'; inc =230;

        indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_" + SN
        pdir = indir + '/Profile-2d'

        for ivar in range(nvar):
            if ivar==0: var = "CarbonAbundance"
            if ivar==1: var = "IronAbundanceToSolar"
           
            infile = pdir + ('/%04d_radius_%s_Hydrogen_number_density.h5' % (inc, var))
            prof_2d_ds = yt.load(infile)

            X = np.log10(prof_2d_ds.data['radius'] / 3.0856e18)
            Y = np.log10(prof_2d_ds.data[var])
            Z = np.log10(prof_2d_ds.data['Hydrogen_number_density'])

            if nvar==1: ax = axs[iSN]
            else:       ax = axs[ivar][iSN]
            cmap = "bds_highcontrast"
            xccolor = 'black'
            pcolor = ax.pcolormesh(X, Y, Z.T, cmap=cmap)
         #  ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-7.0, 3.0)
            if ivar==0: ax.set_ylim(2.0, 6.2)
            if ivar==1: ax.set_ylim(-11.0, -6.2)
            if iSN > 0:
                ax.tick_params(labelleft=False)
            else:
               if ivar==0: ax.set_ylabel(r'$A({\rm C})$', fontsize=fontsize_tick)
               if ivar==1: ax.set_ylabel(r'[ Fe / H ]'  , fontsize=fontsize_tick)
            if ivar==nvar-1:
                ax.set_xlabel(r'log [ Distance / pc ]', fontsize=fontsize_tick)
            else:
                ax.tick_params(labelbottom=False)
            if ivar==0:
                ax.set_title(progmass[iSN], fontsize=fontsize_title, loc='left')
            ax.tick_params(labelsize=fontsize_tick)
            axins = inset_axes(ax,
                               width="50%",  # width = 5% of parent_bbox width
                               height="5%",  # height : 50%
                               loc='lower left',
                               bbox_to_anchor=(0.03, 0.20, 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,
                              )
            cmin = Z.min(); cmax = Z.max(); 
            cmin = -2.0
            if cmax > 16.0:
                cmax = 16.0
            pcolor.set_clim(cmin, cmax)
            ticks = np.linspace(0, 20, 5)
            colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                  ,aspect=10,pad=-0.22 ,shrink=0.50
                   ,ticks=ticks
                      )
          # colbar.outline.set_edgecolor(xccolor)
            xclabel = r'log [ n$_{\rm H}$ / cm$^{-3}$ ]'
            colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
            colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
    if nvar==1:
        plt.subplots_adjust(left   = 0.05
                          , right  = 0.98
                          , bottom = 0.12
                          , top    = 0.94
                          , wspace = 0.03
                          , hspace = 0.00
                           )
    if nvar==2:
        plt.subplots_adjust(left   = 0.05
                          , right  = 0.98
                          , bottom = 0.10
                          , top    = 0.96
                          , wspace = 0.03
                          , hspace = 0.00
                           )
    
    
    fig.savefig("%s/rZ.png" % (outdir))
    
    plt.close('all')
 

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18.0, ysize), sharex='all')
    #fig.suptitle(eng, fontsize=fontsize_suptitle, fontweight='bold')

    for iSN in range(3):
        if iSN==0: SN = 'M13'; inc =137;
        if iSN==1: SN = 'M50'; inc =174;
        if iSN==2: SN = 'M80'; inc =230;

        indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_" + SN
        pdir = indir + '/Profile-1d'

        infile = pdir + ('/%04d_rv.dat' % (inc))
        rtt = np.loadtxt(infile)

        ax=axs[iSN]
     #  ax.annotate(progmass[iSN], xy=(0.03, 1.10), xycoords=ax.transAxes, color='black', fontsize=fontsize_title, va='top', ha='left')
        ax.set_xlabel(r"log [ Distance / pc ]", fontsize=fontsize_label)
        if iSN==0: ax.set_ylabel(r"log [ Time / yr ]", fontsize=fontsize_label)
        ax.set_xlim(-7, 3)
     #  ax.set_ylim(-12, 2)
     #  ax.set_xticks(np.linspace( 0, 16, 5))
     #  ax.set_xticks(np.linspace( 0, 16, 17), minor=True)
     #  ax.set_yticks(np.linspace(-10, 0, 3))
     #  ax.set_yticks(np.linspace(-12, 2, 15), minor=True)
        if iSN > 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=fontsize_tick)
     #  if iSN>0: ax.tick_params(labelleft=False)
        plot = ax.plot(
            np.log10(rtt[:, 0]), np.log10(rtt[:, 2])
          , np.log10(rtt[:, 0]), np.log10(rtt[:, 3])
                      )
        plot[0].set_linestyle("-");    plot[0].set_color("blue");  plot[0].set_linewidth(2)
        plot[1].set_linestyle("-");    plot[1].set_color("red" );  plot[1].set_linewidth(2)
        if iSN == 0:
            plt.legend(labelspacing=0.0, loc='lower right')
            ax.legend([
               r"$t_{\rm vor}$"
             , r"$t_{\rm col}$"
                    ], ncol=1, fontsize=fontsize_tick)

    plt.subplots_adjust(left   = 0.05
                      , right  = 0.98
                      , bottom = 0.12
                      , top    = 0.94
                      , wspace = 0.03
                      , hspace = 0.00
                       )
    
    fig.savefig("%s/rt.png" % (outdir))
    
    plt.close('all')



if PROFILE_COL:
    i_rad    =  0
    i_nH     =  1
    i_Tg     =  2
    i_vrad   =  3
    i_ee     =  4

  # profM13 = [None] * 3
  # profM50 = [None] * 3
  # profM80 = [None] * 3

    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M13"
    pdir = indir + '/Profile-1d'
  # fn_profM13 = pdir + '/0123_rv.dat'
  # profM13[0] = np.loadtxt(fn_profM13)
  # fn_profM13 = pdir + '/0130_rv.dat'
  # profM13[1] = np.loadtxt(fn_profM13)
    fn_profM13 = pdir + '/0137_rv.dat'
    profM13    = np.loadtxt(fn_profM13)

    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M50"
    pdir = indir + '/Profile-1d'
  # fn_profM50 = pdir + '/0161_rv.dat'
  # profM50[0] = np.loadtxt(fn_profM50)
  # fn_profM50 = pdir + '/0167_rv.dat'
  # profM50[1] = np.loadtxt(fn_profM50)
    fn_profM50 = pdir + '/0174_rv.dat'
    profM50    = np.loadtxt(fn_profM50)

    indir = "/home/genchiaki/scratch/enzo-dev/run/CosmologySimulation/TestPop3-L2_M14_M80"
    pdir = indir + '/Profile-1d'
  # fn_profM80 = pdir + '/0217_rv.dat'
  # profM80[0] = np.loadtxt(fn_profM80)
  # fn_profM80 = pdir + '/0223_rv.dat'
  # profM80[1] = np.loadtxt(fn_profM80)
    fn_profM80 = pdir + '/0230_rv.dat'
    profM80    = np.loadtxt(fn_profM80)

    nvar = 3
    fig, axs = plt.subplots(nrows=nvar, ncols=1, figsize=(6.0, 15.0))
    for ivar in range(nvar):
        ax = axs[ivar]
        ax.tick_params(labelsize=fontsize_tick)
        if ivar==nvar-1: ax.set_xlabel(r"log [ Radius / pc ]", fontsize=fontsize_label)
        if ivar==0: ax.set_ylabel(r"log [ $n_{\rm H}$ / cm$^{-3}$ ]"    , fontsize=fontsize_label)
        if ivar==1: ax.set_ylabel(r"log [ $T$ / K ]"                    , fontsize=fontsize_label)
        if ivar==2: ax.set_ylabel(r"$v_{\rm rad}$ [ km s$^{-1} ]$"      , fontsize=fontsize_label)
        if ivar==3: ax.set_ylabel(r"log [ $E_{\rm th}$ / $E_{\rm gr}$ ]", fontsize=fontsize_label)
        ax.set_xticks(np.linspace( -6, 2,  5))
        ax.set_xticks(np.linspace( -7, 2, 10), minor=True)
        ax.set_xlim([-7, 2])
     #  if ivar==0: ylim = [-1  , 17]
     #  if ivar==1: ylim = [ 1.6,  3.4]
     #  if ivar==2: ylim = [-5.2,  1]
     #  ax.set_ylim(ylim)
        if ivar==3: ax.set_ylim([0, 2])
        if ivar==0:
            ax.set_yticks(np.linspace(   0,  16,  5))
            ax.set_yticks(np.linspace(  -1,  17, 19), minor=True)
        if ivar==2:
            ax.set_yticks(np.linspace(  -4,   0,  3))
            ax.set_yticks(np.linspace(  -5,   1,  7), minor=True)
        if ivar<nvar-1: ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
        if ivar == 0:
            xvar0 = np.log10(profM13[:, i_rad])
            yvar0 = np.log10(profM13[:, i_nH ])
            xvar1 = np.log10(profM50[:, i_rad])
            yvar1 = np.log10(profM50[:, i_nH ])
            xvar2 = np.log10(profM80[:, i_rad])
            yvar2 = np.log10(profM80[:, i_nH ])
        if ivar == 1:
            xvar0 = np.log10(profM13[:, i_rad])
            yvar0 = np.log10(profM13[:, i_Tg ])
            xvar1 = np.log10(profM50[:, i_rad])
            yvar1 = np.log10(profM50[:, i_Tg ])
            xvar2 = np.log10(profM80[:, i_rad])
            yvar2 = np.log10(profM80[:, i_Tg ])
        if ivar == 2:
            xvar0 = np.log10(profM13[:, i_rad])
            yvar0 =          profM13[:, i_vrad] / 1.0e5
            xvar1 = np.log10(profM50[:, i_rad])
            yvar1 =          profM50[:, i_vrad] / 1.0e5
            xvar2 = np.log10(profM80[:, i_rad])
            yvar2 =          profM80[:, i_vrad] / 1.0e5
        if ivar == 3:
            xvar0 = np.log10(profM13[:, i_rad])
            yvar0 =          profM13[:, i_ee ] 
            xvar1 = np.log10(profM50[:, i_rad])
            yvar1 =          profM50[:, i_ee ] 
            xvar2 = np.log10(profM80[:, i_rad])
            yvar2 =          profM80[:, i_ee ] 

     #  ax.plot([-1.8, -1.8], ylim, color = 'blue', linestyle=':', linewidth = 1)
     #  ax.plot([-5.3, -5.3], ylim, color = 'red' , linestyle=':', linewidth = 1)

        ax.plot(xvar0, yvar0, color = 'orange', linestyle='-', linewidth = 2, label = r"$M_{\rm PopIII} = 13 \ {\rm M}_{\bigodot}$")
        ax.plot(xvar1, yvar1, color = 'purple', linestyle='-', linewidth = 2, label = r"$M_{\rm PopIII} = 50 \ {\rm M}_{\bigodot}$")
        ax.plot(xvar2, yvar2, color = 'green' , linestyle='-', linewidth = 2, label = r"$M_{\rm PopIII} = 80 \ {\rm M}_{\bigodot}$")

        if ivar == 0:
          ax.legend(labelspacing=0.0, ncol=1, fontsize=fontsize_legend_s)

    plt.subplots_adjust(left   = 0.20
                      , right  = 0.96
                      , bottom = 0.07
                      , top    = 0.98
                      , wspace = 0.00
                      , hspace = 0.00
                       )
    fig.align_labels(axs)
    fig.savefig("%s/rv.pdf" % outdir)

    plt.close('all')
