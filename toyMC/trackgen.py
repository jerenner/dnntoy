"""
trackgen.py

Generates tracks with multiple scattering.

Usage:

python trackgen.py <start_event_number>

Track output format:
    
    x0 y0 z0 deltaE
    
    (x0,y0,z0): "hit" location
    deltaE: the energy deposited in this hit
    
    All distances are in mm

"""
import sys
import numpy as np
import scipy.integrate as integrate
import random as rd
import os
import h5py
from math import *
from trackdefs import *
from scipy.interpolate import interp1d

from abc import ABCMeta, abstractmethod

debug = 0

# -----------------------------------------------------------------------------
# Get the arguments, if any.
# -----------------------------------------------------------------------------
args = sys.argv;
if(len(args) < 1):
    print "Usage:\n\n python trackgen.py <start_event>";
    exit();
trk_startnum = int(args[1]);

# Create directories
if(not os.path.isdir(trk_outdir)): os.mkdir(trk_outdir);
if(not os.path.isdir("{0}/{1}".format(trk_outdir,trk_name))): 
    os.mkdir("{0}/{1}".format(trk_outdir,trk_name));
if(not os.path.isdir("{0}/{1}".format(trk_outdir,trk_name))): os.mkdir("{0}/{1}".format(trk_outdir,trk_name));

# Read in the stopping power and interpolate.
xesp_tbl = np.loadtxt("{0}/data/xe_estopping_power_NIST.dat".format(os.path.dirname(os.path.abspath(__file__))));
rho = pc_rho0*(Pgas/(Tgas/273.15))*(pc_m_Xe/pc_NA);
e_vals = xesp_tbl[:,0];
dEdx_vals = xesp_tbl[:,1];
e_vals = np.insert(e_vals,0,0.0);
dEdx_vals = np.insert(dEdx_vals,0,dEdx_vals[0]);
xesp = interp1d(e_vals,dEdx_vals*rho,kind='cubic');
print "xenon density is rho = {0} g/cm^3; Lr = {1} cm".format(rho,Lr);

# Calculate the cyclotron frequency.
wcyc = -1.0*(pc_eC/pc_me)*Bfield;
print "Cyclotron frequency wcyc = {0}".format(wcyc);

# Open the hdf5 file.
h5f = h5py.File("{0}/{1}/{2}_{3}.h5".format(trk_outdir,trk_name,trk_name,trk_startnum));

# Create num_tracks tracks.
for ntrk in range(num_tracks):
    
    print "Processing track {0}".format(ntrk + trk_startnum);
        
    # Declare arrays for the track.
    # x0 y0 zi zf ux uy uz E deltaE deltaX
    trk_x = []; trk_y = []; trk_zi = []; trk_zf = [];
    trk_ux = []; trk_uy = []; trk_uz = [];
    trk_E = []; trk_deltaE = []; trk_deltaX = [];

    nelecs = 1;
    if(trk_bb):
        nelecs = 2;

    # Initialize the track.    
    te_0 = 1.0*(E_0-pc_me_MeV)/nelecs;
    tx_0 = 0.; ty_0= 0.; tz_0 = 0.;
    costheta_i = rd.uniform(-1,1); sintheta_i = sqrt(1 - costheta_i**2);
    phi_i = rd.uniform(0,2*pi); sinphi_i = sin(phi_i); cosphi_i = cos(phi_i);
    ux_0 = sintheta_i*cosphi_i;
    uy_0 = sintheta_i*sinphi_i;
    uz_0 = costheta_i;
    #print "Initial direction is ({0},{1},{2}); perpendicular velocity is: {3}".format(ux_0,uy_0,uz_0,sqrt(ux_0**2 + uy_0**2));

    for nelec in range(nelecs):
        
        # Lists to hold temporary track quantities.
        trk_tx = []; trk_ty = []; trk_tzi = []; trk_tzf = [];
        trk_tux = []; trk_tuy = []; trk_tuz = [];
        trk_tE = []; trk_tdeltaE = []; trk_tdeltaX = [];
        
        # Set the initial position and energy.
        te = te_0;
        tx = tx_0; ty = ty_0; tz = tz_0;
        if(nelec == 1):
            ux = -ux_0; uy = -uy_0; uz = -uz_0;
        else:
            ux = ux_0; uy = uy_0; uz = uz_0;
    
        # Continue until 0 energy.
        while(te > E_tol):

            # Compute the current momentum of the track.
            ptrk = sqrt((te + 0.511)**2 - 0.511**2);
            #print "Momentum is {0}".format(ptrk);
        
            # Determine the energy loss for this step.        
            if(te < eslice):
                deltaE = te;
            else:
                deltaE = eslice;
            te -= deltaE;
            if(te < 0.): te = 0.;
        
            if(debug > 0): print "-> Energy = {0}, energy loss: {1}".format(te,deltaE);

            # Determine the distance of this step.
            deltaX = 10.0*integrate.quad(lambda ee: 1./xesp(ee),te,te+deltaE,limit=1000)[0]
        
            # Make the step.
            if(Bfield < 1.0e-10):
            
                # No B-field case.
                dx = deltaX*ux; dy = deltaX*uy; dz = deltaX*uz;

            else:
            
                # Compute the velocity.
                vel = (ptrk/(te + deltaE + 0.511))*pc_clight*1000.;  # in m/s*1000 = mm/s
            
                velX = vel*ux;
                velY = vel*uy;
                velZ = vel*uz;
                #print "Velocity = {0}".format(vel);
            
                # Convert deltaX to a time.
                dt = deltaX/vel;
                wt = dt*wcyc;
                #print "dt = {0}".format(dt);
            
                # Compute the changes in velocity.
                velX_B = velX*cos(wt) + velY*sin(wt); # velX + wcyc*velY*dt; #
                velY_B = velY*cos(wt) - velX*sin(wt); # velY - wcyc*velX*dt;
                velZ_B = velZ;
                #print "wcyc*dt = {0}, velX = {1}, velZ = {2}".format(wcyc*dt,velX,velZ);
            
                # Compute the changes in position.
                dx = velX*dt; #(velX/wcyc)*sin(wt) + (velY/wcyc)*(1 - cos(wt));
                dy = velY*dt; #(velY/wcyc)*sin(wt) - (velX/wcyc)*(1 - cos(wt));
                dz = deltaX*uz;
                
                
            if(debug > 0): print "-> Step: deltaX = {0}; dx = {1}, dy = {2}, dz = {3}".format(deltaX,dx,dy,dz);
        
            # Record the variables for the step.
            trk_tx.append(tx + dx);  #dx/2
            trk_ty.append(ty + dy);  #dy/2
            trk_tzi.append(tz);
            trk_tzf.append(tz + dz);
            trk_tux.append(ux);
            trk_tuy.append(uy);
            trk_tuz.append(uz);
            trk_tE.append(te + deltaE);
            trk_tdeltaE.append(deltaE);
            trk_tdeltaX.append(deltaX);
        
            # Update the positions and direction vectors.
            tx += dx; ty += dy; tz += dz;
            if(Bfield > 1.0e-10):
                vel_B = sqrt(velX_B**2 + velY_B**2 + velZ_B**2);
                ux = velX_B/vel_B;
                uy = velY_B/vel_B;
                uz = velZ_B/vel_B;
        
            # Determine the scattering angles in the frame in which the track
            #  direction is aligned with the z-axis.
            sigma_theta = SigmaThetaMs(ptrk,deltaX/Lr);
            #if(deltaX/Lr < 0.001 or deltaX/Lr > 100.):
            #    print "WARNING: L/Lr = {0} out of range of validity of the formula.".format(deltaX/Lr);
            thetaX = rd.gauss(0,sigma_theta);
            thetaY = rd.gauss(0,sigma_theta);
            tanX = tan(thetaX);
            tanY = tan(thetaY);
        
            if(debug > 0): print "-> sigma(theta) = {0}; tanX = {1}, tanY = {2}".format(sigma_theta,tanX,tanY);
        
            # Compute the direction cosines of the rotation matrix to move to the lab frame.
            nxy = sqrt(ux**2 + uy**2);
            if(nxy > 0.):
                alpha1 = uy/nxy; alpha2 = -ux*uz/nxy; alpha3 = ux;
                beta1 = -ux/nxy; beta2 = -uy*uz/nxy; beta3 = uy;
                gamma1 = 0.; gamma2 = nxy; gamma3 = uz;
            else:
                # Special case; the direction vector is the x-axis.  Choose
                #  the orthonormal basis as the normal unit vectors.
                alpha1 = 1.; alpha2 = 0.; alpha3 = 0.;
                beta1 = 0.; beta2 = 1.; beta3 = 0.;
                gamma1 = 0.; gamma2 = 0.; gamma3 = 1.;
        
            if(debug > 1): print "-> alpha1 = {0}, alpha2 = {1}, alpha3 = {2}".format(alpha1,alpha2,alpha3);
            if(debug > 1): print "-> alpha1 = {0}, alpha2 = {1}, alpha3 = {2}".format(alpha1,alpha2,alpha3);
            if(debug > 1): print "-> beta1 = {0}, beta2 = {1}, beta3 = {2}".format(beta1,beta2,beta3);
            if(debug > 1): print "-> gamma1 = {0}, gamma2 = {1}, gamma3 = {2}".format(gamma1,gamma2,gamma3);

            # Determine direction vector components in the reference (lab) frame.
            nrm = sqrt(tanX**2 + tanY**2 + 1);
            xp = (alpha1*tanX + alpha2*tanY + alpha3)/nrm;
            yp = (beta1*tanX + beta2*tanY + beta3)/nrm;
            zp = (gamma1*tanX + gamma2*tanY + gamma3)/nrm;
        
            # Set the new direction vector.
            nrm = sqrt(xp**2 + yp**2 + zp**2);
            ux = xp/nrm;
            uy = yp/nrm;
            uz = zp/nrm;
        
        # Add the current track to the main track.
        if(nelecs == 2 and nelec == 0):            
            # For two-electron tracks, reverse the first one.
            trk_tx.reverse(); trk_ty.reverse();
            trk_tzi.reverse(); trk_tzf.reverse();
            trk_tux.reverse(); trk_tuy.reverse(); trk_tuz.reverse();
            trk_tE.reverse(); trk_tdeltaE.reverse(); trk_tdeltaX.reverse();
            
        for txv in trk_tx: trk_x.append(txv);
        for tyv in trk_ty: trk_y.append(tyv);
        for tziv in trk_tzi: trk_zi.append(tziv);
        for tzfv in trk_tzf: trk_zf.append(tzfv);
        for tuxv in trk_tux: trk_ux.append(tuxv);
        for tuyv in trk_tuy: trk_uy.append(tuyv);
        for tuzv in trk_tuz: trk_uz.append(tuzv);
        for tEv in trk_tE: trk_E.append(tEv);
        for tdeltaEv in trk_tdeltaE: trk_deltaE.append(tdeltaEv);
        for tdeltaXv in trk_tdeltaX: trk_deltaX.append(tdeltaXv);
    
    # Make sure we don't have a step over the max step size.
    max_stepl = max(trk_deltaX)
    if(max_stepl > vox_size / 2.): 
        print "ERROR: A step exceeds {0} mm (half intended voxel size), invalid track".format(vox_size / 2.);
        exit();
    
    # Save the track as an entry in the hdf5 file.
    carr = np.array([trk_x, trk_y, trk_zf, trk_deltaE]);               # "combined" array
    h5f.create_dataset("trk{0}".format(ntrk+trk_startnum),data=carr);

# Close the hdf5 file.
h5f.close();
