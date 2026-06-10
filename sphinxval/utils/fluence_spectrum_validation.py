from . import config
from . import plotting_tools
import argparse
import numpy as np
import datetime
import math
from itertools import zip_longest
import matplotlib.pylab as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from scipy.stats import pearsonr
from scipy.optimize import curve_fit 
from lmfit import minimize, Parameters, fit_report
import os
import sys
import re
import csv
import seaborn as sns
from pathlib import Path

"""
The first part of this code (most of it) is all commented out until a later date. It involves fitting
the fluence spectrum to a Band Function or an Ellison Ramaty (powerlaw with an exponential rollover)
and comparing the fit parameters for the observations and the model. The idea is that the physics can
more easily be seen in the fit parameters (rollover energy, low energy spectral slope, high energy spectral slope)
compared to just looking at the error in individual energy bins. Looking at how these fit parameters
vary across the event set can tell you if the physics contained within a fluence spectrum is being
appropriately modeled (rollover energy and spectral slope).

Second part of handles a simpler error calculation between the model energy bins individual fluence
and the observed fluence for the same energy bin, interpolated if required.

"""

def make_diff_from_int(integral_fluxes, energy_bins):
    """ Make differential fluxes from integral channels.
        Subtract each integral flux channel from the consecutive
        one to create differential bins.
    """

    new_bins = []
    nchan = len(integral_fluxes)
    nflx = len(integral_fluxes[0])
    diff_fluxes = np.zeros((nchan-1,nflx))
    diff_bins = np.zeros((nchan-1,2))
    
    for i in range(nchan-1):
        diff_bins[i][0] = energy_bins[i][0]
        diff_bins[i][1] = energy_bins[i+1][0]
        bin_width = diff_bins[i][1] - diff_bins[i][0]
        for j in range(nflx):
            diff_fluxes[i][j] = \
                (integral_fluxes[i][j] - integral_fluxes[i+1][j])/bin_width
            
    
    print("make_diff_from_int converted to energy bins " + str(diff_bins))
    
    return diff_bins, diff_fluxes




#################### Ellison Ramaty Function Subroutines ##################################################
def fit_ellisonramaty_function(energy_bins, fluence_array, energy):
    """
    Fitting the Ellison-Ramaty Function to energy spectra
    Should work with observations or models 
    Inputs:
    energy_bin : array of the same size as the fluence array - contains energy bins (x axis of spectra)
    fluence_array: array of the same size as energy_bin -  contains the fluences corresponding to the
        energy bins (y axis of spectra)
    energy: array to fit the final parameters to, creating a line that'll be plotted in the main function

    Outputs:
    minimize_er - a MinimizerResult from lmfit that contains all of the best fit parameters and their
        uncertanties - use minimize_er.params.valuesdict() after the function call in the main function
        to extract these values
    fit_er -  the final fit corresponding to the energy array - used for final plotting
    transition_er - the roll_over energy of the ellison-ramaty fit
    """
    from lmfit import minimize, Parameters
    import numpy as np

    params_er = Parameters()
    params_er.add('norm', value = 101, min = 1, max = 10**15)
    params_er.add('gamma_er', value = 1.0, min = 0.0000001, max =10)
    params_er.add('roll_over', value = 15.0, min = 1.0, max = 2000)
    minimize_er = minimize(residual_er, params_er, args = [energy_bins, fluence_array], nan_policy= 'propagate', max_nfev= 10**8)
    values_dict = minimize_er.params.valuesdict()
    roll_over = values_dict['roll_over']
    # if roll_over > 1000:
    #     minimize_er, fit_er = fit_single_power_law(energy_bins, fluence_array, energy)
    #     rolloverenergy_er = None
    # else: 
    fit_er, rolloverenergy_er = er_func(minimize_er.params, energy)

    return minimize_er, fit_er, rolloverenergy_er

def residual_er(first_guess, *args):
    """
    Defining the Ramaty-Ellison function - for minimization
    """
  
    parvals = first_guess.valuesdict()
    norm = parvals['norm']
    gamma = parvals['gamma_er']
    E_r = parvals['roll_over']
    
    fluence = args[1]
    energy = args[0]

    
    fit_error = []
    total_fit = []
    e = 0

    for e in range(len(energy)):
        total_fit.append(norm*energy[e]**(-gamma)*math.exp(-energy[e]/E_r))
        fit_error.append(np.abs((fluence[e] / total_fit[e])-1))
        e = e+1
   
    return fit_error

def er_func(first_guess, energy):
    """
    Defining the Ellison-Ramaty function for plotting the fit
    """
    # print(args)
    parvals = first_guess.valuesdict()
    norm = parvals['norm']
    gamma = parvals['gamma_er']
    E_r = parvals['roll_over']
   
    fit = []
    e = 0

    for e in range(len(energy)):
        
        fit.append(norm*energy[e]**(-gamma)*math.exp(-energy[e]/E_r))
        # fit = band[e]*math.exp(-energy[e]/E_r)
        e = e+1

    return fit, E_r



###################### Band Function Subroutines ##############################################
def fit_band_function(energy_bins, fluence_array, energy):
    """ Fitting the Band function to spectral data - see
    description of fit_ellisonramaty_function for full details of 
    inputs/outputs
    """
    from lmfit import minimize, Parameters
    import numpy as np
    params_band = Parameters()
    params_band.add('norm', value = 10000.0, min = 1.0, max = 10**13)
    params_band.add('gamma_b', value = 5.0, min = 0.1, max =20)
    params_band.add('break_energy', value = 7.0, min = 5.0, max = 1000)
    params_band.add('gamma_a', value = 0.70, min = 0.1, max = 4.0)
    minimize_band = minimize(residual_band, params_band, args= [energy_bins, fluence_array], nan_policy= 'propagate', max_nfev= 100000000)
    
    fit_band, transition_band = band_func(minimize_band.params, energy) #Band
    return minimize_band, fit_band, transition_band

def residual_band(first_guess, *args):
    """
    Defining the Band function for minimization

    Inputs:
    First Guess. Array of length 5, first guess for the values
        the free parameters will take.
        Norm - normalization factor (A)
        Gamma_a - low energy spectral index
        Gamma_b - high energy spectral index
        E_0 -  break energy
       

    args. Array of length 2, contains any other arguments used in the function
        since least_squares requires functions of the form func(x, *args)
        args[0] - energy bins
        args[1] - fluence values       
    """
    parvals = first_guess.valuesdict()
    norm = parvals['norm']
    gamma_a = parvals['gamma_a']
    gamma_b = parvals['gamma_b']
    E_0 = parvals['break_energy']


    
    fluence = args[1]
    energy = args[0]

    band = []
    fit_error = []
    total_fit = []
    e = 0

    E_t = (gamma_b - gamma_a)*E_0

    for e in range(len(energy)-1):
        
        if energy[e] <= E_t:

            band.append(norm*energy[e]**(-gamma_a)*math.exp(-energy[e] / E_0))
            
        else:
            band.append(norm*energy[e]**(-gamma_b)*(((gamma_b-gamma_a)*E_0)**(gamma_b-gamma_a))*math.exp(gamma_a-gamma_b))   
        total_fit.append((band[e]))
        fit_error.append(np.abs(fluence[e] / total_fit[e])-1)
        e = e+1

    return fit_error

def band_func(first_guess, energy):
    """
    Defining the Band function for plotting the fit
    """
    

    parvals = first_guess.valuesdict()
    norm = parvals['norm']
    gamma_a = parvals['gamma_a']
    gamma_b = parvals['gamma_b']
    E_0 = parvals['break_energy']
        # print(norm, gamma_a, gamma_b, E_0, E_t, E_r)
    
    band = []
    e = 0
    E_transition = (gamma_b - gamma_a)*E_0
    # print('Transition Energy ', E_transition)

    for e in range(len(energy)):
        
        if energy[e] <= (gamma_b - gamma_a)*E_0:
            band.append(norm*energy[e]**(-gamma_a)*math.exp(-energy[e] / E_0))
        else:
            band.append(norm*energy[e]**(-gamma_b)*((gamma_b-gamma_a)*E_0)**(gamma_b-gamma_a)*math.exp(gamma_a-gamma_b))
        band[e] = band[e]
        e = e+1

    return band, E_transition

def fluence_fitting_stuff(obs, pred, events, model_name, threshold, energy_key):
    """
    Doing stuff with the fluences that are found - plotting and fitting
    the spectra via a Band/ER function which is then fit via the least_squares
    package



    Inputs:
    obs - 
    pred - 
    event_label - string labeling the event being investigated
    model_name - string, model's name
    """

    if len(obs) == len(pred) == len(events):
        model_band_gamma_a = []
        model_band_gamma_b = []
        model_band_rollover = []
        model_band_norm = []

        model_er_gamma = []
        model_er_rollover = []
        model_er_norm = []


        obs_band_gamma_a = []
        obs_band_gamma_b = []
        obs_band_rollover = []
        obs_band_norm = []

        obs_er_gamma = []
        obs_er_rollover = []
        obs_er_norm = []

        for m in range(len(pred)):
            model_fluence_array = []
            model_energy_bin = []
            model_bin_min = []
            model_bin_max = []
            model_bin_width = []
            for n in range(len(pred[m])):
                # Checking that we are comparing the same kind of spectrum for both obs and pred - both need to be integral or both are differential
                if pred[m][n]['energy_max'] == pred[m][n]['energy_min'] and obs[m][0]['energy_max'] == -1 or pred[m][n]['energy_max'] == -1 and obs[m][0]['energy_max'] == -1 :
                    model_energy_bin.append(pred[m][n]['energy_min'])
                    model_fluence_array.append(pred[m][n]['fluence'])
                elif pred[m][n]['energy_max'] != -1 and obs[m][n]['energy_max'] != -1:
                    model_energy_bin.append((pred[m][n]['energy_max']+pred[m][n]['energy_min'])/2)
                    model_bin_min.append(model_energy_bin[m][n] - pred[m][n]['energy_min'])
                    model_bin_max.append(pred[m][n]['energy_max'] - model_energy_bin[m][n])
                    model_fluence_array.append(pred[m][n]['fluence'])
            

            
            obs_fluence_array = []
            obs_energy_bin = []
            obs_bin_min = []
            obs_bin_max = []
            obs_bin_width = []
            n = 0
            for n in range(len(obs[m])):
                # if integral flux turn to differential flux (rudimentary for now)
                if obs[m][n]['energy_max'] == obs[m][n]['energy_min'] or obs[m][n]['energy_max'] == -1:
                    obs_energy_bin.append(obs[m][n]['energy_min'])
                    obs_fluence_array.append(obs[m][n]['fluence']) # -obs[m][n+1]['fluence']
                else:
                    obs_energy_bin.append((obs[m][n]['energy_max']+obs[m][n]['energy_min'])/2)
                    # obs_bin_min.append(obs_energy_bin[m][n] - obs[m][n]['energy_min'])
                    # obs_bin_max.append(obs[m][n]['energy_max'] - obs_energy_bin[m][n])
                    obs_fluence_array.append(obs[m][n]['fluence'])
            if len(obs_fluence_array) != len(obs[m]):
                obs_fluence_array.append(obs[m][n+1]['fluence'])
                obs_energy_bin.append(obs[m][n]['energy_min'])
    
            new_obs_fluence, new_bins = interpolate_fluence(obs_energy_bin, obs_fluence_array, model_energy_bin, model_fluence_array)
            energy = np.arange(0,2000,1) # Creating an array of energies across all bins (5 - 2000 MeV) to use for plotting the end fit
            

            
            # # input()
            min_obs_er_params, fit_obs_er, transition_obs_er = fit_ellisonramaty_function(obs_energy_bin, obs_fluence_array, energy)
            print('Obs Fit ER Parameters', min_obs_er_params.params, min_obs_er_params.chisqr)#, min_obs_er.message)
            obs_powerlaw = False
            try:
                obs_er_params = min_obs_er_params.params.valuesdict()
                obs_norm_er = obs_er_params['norm']
                obs_gamma_er = obs_er_params['gamma_er']
                obs_er_er = obs_er_params['roll_over']
                print('Now printing fit report for Ellison-Ramaty function:')
                print(fit_report(min_obs_er_params))
            except:
                obs_er_params = min_obs_er_params.params.valuesdict()
                obs_norm_er = obs_er_params['norm']
                obs_gamma_er = obs_er_params['spectral_index']
                obs_er_er = obs_er_params['coefficient']
                print('Now printing fit report for Powerlaw function:')
                print(fit_report(min_obs_er_params))
                obs_powerlaw = True

            min_model_er_params, fit_model_er, transition_model_er = fit_ellisonramaty_function(model_energy_bin, model_fluence_array, energy)
            print('Model Fit ER Parameters', min_model_er_params.params, min_model_er_params.chisqr)#, min_model_er.message)
            model_powerlaw = False
            try:
                model_er_params = min_model_er_params.params.valuesdict()
                model_norm_er = model_er_params['norm']
                model_gamma_er = model_er_params['gamma_er']
                model_er_er = model_er_params['roll_over']
                print(fit_report(min_model_er_params))
            except:
                model_er_params = min_model_er_params.params.valuesdict()
                model_norm_er = model_er_params['norm']
                model_gamma_er = model_er_params['spectral_index']
                model_er_er = model_er_params['coefficient']
                print(fit_report(min_model_er_params))
                model_powerlaw = True


            # input()
            min_obs_band_params, fit_obs_band, transition_obs_band = fit_band_function(obs_energy_bin, obs_fluence_array, energy)
            print('Best Fit Band Parameters Obs: ', min_obs_band_params.params, min_obs_band_params.chisqr)#, min_obs_band.message)
            obs_band_params = min_obs_band_params.params.valuesdict() 
            obs_norm_band = obs_band_params['norm']
            obs_gamma_a_band = obs_band_params['gamma_a']
            obs_gamma_b_band = obs_band_params['gamma_b']
            obs_e_0_band = obs_band_params['break_energy']
            print(fit_report(min_obs_band_params))

            min_model_band_params, fit_model_band, transition_model_band = fit_band_function(model_energy_bin, model_fluence_array, energy)
            print('Best Fit Band Parameters Model: ', min_model_band_params.params, min_model_band_params.chisqr)#, min_model_band.message)
            model_band_params = min_model_band_params.params.valuesdict() 
            model_norm_band = model_band_params['norm']
            model_gamma_a_band = model_band_params['gamma_a']
            model_gamma_b_band = model_band_params['gamma_b']
            model_e_0_band = model_band_params['break_energy']

        
            # print(new_bins, new_obs, model_fluence_array)
            # print(model_name, new_bins)
            # input()
            # print(fluence_error)


            cwd = os.getcwd()

            fig = plt.figure(figsize=(20,10))
            ax = plt.subplot(111)
            ax.errorbar(obs_energy_bin, obs_fluence_array, fmt = 'o', label = 'Observations', color = 'Red') 
            ax.plot(model_energy_bin, model_fluence_array, 'P', label = 'Model', color = 'Blue')
            ax.plot(new_bins, new_obs_fluence, 'o', label = 'Interpolated Obs Fluence')
            plt.yscale("log")
            plt.xscale('log')
            
            
            plt.ylim((10**(-2),10**12))

            ax.plot(energy, fit_obs_er, color = 'Red', linestyle = 'dashdot', label = 'Ellison-Ramaty')
            ax.plot(energy, fit_model_er, color = 'Blue', linestyle = 'dashdot')
            ax.plot(energy, fit_obs_band, color = 'Red', linestyle = 'dashed', label = 'Band')
            ax.plot(energy, fit_model_band, color = 'Blue', linestyle = 'dashed')
            
            
            if obs_powerlaw == False:
                ax.text(100,10**7,'E-R Function Parameters: ' + str(obs_gamma_er) + ', ' + str(obs_er_er), color = 'Red')
            else:
                ax.text(100,10**7,'Powerlaw Parameters: ' + '\n Spectral Index ' + str(obs_gamma_er),  color = 'Red')
            ax.text(100, 10**8, 'Band Function Parameters: ' + 'Gamma_a= ' + str(obs_gamma_a_band) + '\nGamma_b=' + str(obs_gamma_b_band) + '\nE_break=  '\
                + str(obs_e_0_band), color = 'Red')
            if model_powerlaw == False:
                ax.text(10**1,0.65,'E-R Function Parameters: ' + str(model_gamma_er) + ', ' + str(model_er_er), color = 'Blue')
            else:
                ax.text(10**1,0.65,'Powerlaw Parameters: ' + '\n Spectral Index ' + str(model_gamma_er),  color = 'Blue')
            ax.text(10, 10**-1, 'Band Function Parameters: Gamma_a= ' + str(model_gamma_a_band) + '\nGamma_b= ' + str(model_gamma_b_band) + '\nE_break= '\
                + str(model_e_0_band), color = 'Blue')
            # ax.axvline(x=transition_obs_er, color ='Red', linestyle = 'dotted', label = 'roll-over energy obs: Ellison-Ramaty')
            # ax.axvline(x=transition_model_er, color ='Blue', linestyle = 'dotted')
            plt.yscale("log")
            plt.xscale('log')
            plt.xlabel("Energy (MeV)", fontsize = '12')
            plt.ylabel('Fluence (' + str(config.t2_units) + ')', fontsize = '12')
            plt.title('Observed and Model Fluence Spectra for ' + str(events[m]).rsplit(' ')[0])
            ax.legend(loc = 'lower left', fontsize='10', framealpha=1.0) 
            savename = config.outpath + '/plots/band_fitting_' + model_name + "_" +  str(events[m]).rsplit(' ')[0] + '_' + energy_key + '_' + threshold + '.pdf'
            fig.savefig(savename)

            plt.close(fig)

            model_band_gamma_a.append(model_gamma_a_band)
            model_band_gamma_b.append(model_gamma_b_band)
            model_band_rollover.append(model_e_0_band)
            model_band_norm.append(model_norm_band)

            model_er_gamma.append(model_gamma_er)
            model_er_rollover.append(model_er_er)
            model_er_norm.append(model_norm_er)

            obs_band_gamma_a.append(obs_gamma_a_band)
            obs_band_gamma_b.append(obs_gamma_b_band)
            obs_band_rollover.append(obs_e_0_band)
            obs_band_norm.append(obs_norm_band)

            obs_er_gamma.append(obs_gamma_er)
            obs_er_rollover.append(obs_er_er)
            obs_er_norm.append(obs_norm_er)
        
        
        
        
        
        
        corr_plot = plotting_tools.correlation_plot(obs_er_gamma, model_er_gamma, 'Spectral Slopes of Ellison-Ramaty Function', xlabel="Observations", ylabel="Model",  value="Value", use_log = False, use_logx = False, use_logy = False)
        figname = config.outpath + '/plots/Correlation_er_gamma_' + model_name + "_" + energy_key + '_' + threshold
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()

        corr_plot = plotting_tools.correlation_plot(obs_er_rollover, model_er_rollover, 'Rollover Energy of Ellison-Ramaty Function', xlabel="Observations", ylabel="Model",  value="Value", use_log = True, use_logx = True, use_logy = True)
        figname = config.outpath + '/plots/Correlation_er_rollover_' + model_name + "_" + energy_key + '_' + threshold
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()

        corr_plot = plotting_tools.correlation_plot(obs_band_gamma_a, model_band_gamma_a, 'Low Energy Spectral Slopes of Band Function', xlabel="Observations", ylabel="Model",  value="Value", use_log = False, use_logx = False, use_logy = False)
        figname = config.outpath + '/plots/Correlation_band_gamma_a_' + model_name + "_" + energy_key + '_' + threshold
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()

        corr_plot = plotting_tools.correlation_plot(obs_band_gamma_b, model_band_gamma_b, 'High Energy Spectral Slopes of Band Function', xlabel="Observations", ylabel="Model",  value="Value", use_log = False, use_logx = False, use_logy = False)
        figname = config.outpath + '/plots/Correlation_band_gamma_b_' + model_name + "_" + energy_key + '_' + threshold
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()

        corr_plot = plotting_tools.correlation_plot(obs_band_rollover, model_band_rollover, 'Rollover Energy of Band Function', xlabel="Observations", ylabel="Model",  value="Value", use_log = True, use_logx = True, use_logy = True)
        figname = config.outpath + '/plots/Correlation_band_rollover_' + model_name + "_" + energy_key + '_' + threshold
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()



def fluence_histograms(model_energy_bins, fluence_error, obs_error_energy_bins, model_name, energy_key, threshold):
    i = 0
    j = 0

   
    all_bins_error = []
    energy_label = []
    for energy in model_energy_bins:
        bin_error = []
        
        for i in range(len(fluence_error)):
            
            # print(fluence_error[i][6])
            for j in range(len(obs_error_energy_bins[i])):
                if obs_error_energy_bins[i][j] == energy:
                    if fluence_error[i][j] == None:
                        pass
                    else:
                        bin_error.append(fluence_error[i][j])
                            
                        
        
        if bin_error == []:
            continue
        all_bins_error.append(bin_error)
        energy_label.append(energy)
        fig1 = plt.figure(figsize=(8,5))
        ax1 = plt.subplot(111)
        plt.hist(bin_error, alpha=0.5, label=model_name + ' ' +str(energy))
        plt.legend(loc='upper right')
        plt.xlabel('Log Error')
        plt.ylabel('Counts')
        plt.title('Histogram of ' + model_name + ' Fluence Log Error for energy bin ' + str(energy) + ' MeV')
        figname = config.outpath + '/plots/Fluence_LE_histogram_' + model_name + "_" +  energy_key + '_'  + threshold + '_' \
                + str(energy) + "_MeV" ".pdf"
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        


    # Creating the box plots now
    savename = config.outpath + '/plots/Fluence_LE_boxplot_' + model_name + "_" +  energy_key + '_'  + threshold + '.pdf'
    plot_title = 'Fluence Log Error Across all Energy Bins for ' + model_name + ' with energy key/threshold ' + energy_key + ' ' + threshold
    plotting_tools.box_plot(all_bins_error, energy_label, x_label="Energy Bins (MeV)", y_label="Log Error", title=plot_title, \
        save=savename, uselog=False, showplot=False, closeplot=True)


    return



def interpolate_fluence(obs_energy_bin, obs_fluence_array, model_energy_bin, model_fluence_array):
    """ 
    Interpolating the obs. energy bins to match to model energy bins 
    to find fluence error 
    """
    nbin_obs = len(obs_energy_bin)
    nbin_mod = len(model_energy_bin)
    # print(obs_fluence_array)
    new_obs_fluence = []
    new_bins = []
    i = 0
    for i in range(nbin_obs-1):
        # taking care of zeroes since log10(0) is infinity
        if obs_fluence_array[i] == 0.0:
            obs_fluence_array[i] = 1.0
        elif obs_fluence_array[i+1] == 0.0:
            obs_fluence_array[i+1] = 1.0
        rise =np.log10(obs_fluence_array[i+1]) - np.log10(obs_fluence_array[i])
        # rise.imag = 
        run = np.log10(obs_energy_bin[i+1]) - np.log10(obs_energy_bin[i])
        
        slope = rise / run
        k = 0
        # print('Obs Bin ',obs_energy_bin[i], slope)
        for k in range(nbin_mod):
            # print(k)
            if model_energy_bin[k] >= obs_energy_bin[i] and model_energy_bin[k] < obs_energy_bin[i+1]:
                new_bins.append(model_energy_bin[k])
                new_obs_fluence.append(obs_fluence_array[i]*(model_energy_bin[k]/obs_energy_bin[i])**(slope))
                
                
                # print(new_bins[k], new_obs_fluence[k], rise, run)
            k += 1
        i += 1

    # This replaces any 0s with 1s b/c we take the log10 of the fluence elements for
    # the LE calc so log10(0) is undefined whereas log10(1) = 0
    if any(new_obs_fluence) == 0.0:
        for element in new_obs_fluence:
            new_obs_fluence[element] = float(1) 
               
    return new_obs_fluence, new_bins


def error_fluence_calc(obs_energy_bin, obs_fluence, model_energy_bin, model_fluence):
    """
    Use interpolated obs_energy_bins from interpolate_fluence 

    Calculates the Log Error of the Fluence between the model fluence and 
    the observed fluence for each of the model's energy bins
    """ 
    fluence_error = []
    error_bins = []
    i = 0
    for i in range(len(obs_energy_bin)):
        for j in range(len(model_energy_bin)):
            if obs_energy_bin[i] == model_energy_bin[j]:
                # print(model_energy_bin[j])
                error_bins.append(model_energy_bin[j])
                fluence_error.append(np.log10(model_fluence[j])-np.log10(obs_fluence[i]))
                if fluence_error[i] == np.inf or fluence_error[i] == -np.inf:
                    fluence_error[i] = None
        i += 1
    return fluence_error, error_bins



def fluence_validation(sub, model, energy_key, threshold):
    obs = sub['Observed SEP Fluence Spectrum'].to_list()
    pred = sub['Predicted SEP Fluence Spectrum'].to_list()
    events = sub['Prediction Window Start'].to_list()
    obs_units = sub['Observed SEP Fluence Units'].to_list() # useless right now? Everything has the same units... its suspicious
    pred_units = sub['Predicted SEP Fluence Units'].to_list() # useless right now? Everything has the same units... its suspicious, like
    # UMASEP's energy bins look like differential channels but have the units of integral.... 




    do_fitting = True
    fluence_all_events = []
    error_bins_all_events = []

    if len(obs) == len(pred) == len(events):
        for m in range(len(pred)):
            model_fluence_array = []
            model_energy_bin = []
            model_bin_min = []
            model_bin_max = []
            model_bin_width = []
            for n in range(len(pred[m])):
                # Checking that we are comparing the same kind of spectrum for both obs and pred - both need to be integral or both are differential 
                #  (if energy_min =  energy_max or energy_max = -1 then these are integral(?), otherwise they are diff)
                # 
                if pred[m][n]['energy_max'] == pred[m][n]['energy_min'] and obs[m][0]['energy_max'] == -1 \
                    or pred[m][n]['energy_max'] == -1 and obs[m][0]['energy_max'] == -1:
                        model_energy_bin.append(pred[m][n]['energy_min'])
                        model_fluence_array.append(pred[m][n]['fluence'])
                elif pred[m][n]['energy_max'] != -1 and obs[m][n]['energy_max'] != -1:
                    model_energy_bin.append((pred[m][n]['energy_max']+pred[m][n]['energy_min'])/2)
                    model_bin_min.append(model_energy_bin[m][n] - pred[m][n]['energy_min'])
                    model_bin_max.append(pred[m][n]['energy_max'] - model_energy_bin[m][n])
                    model_fluence_array.append(pred[m][n]['fluence'])
            


            
            obs_fluence_array = []
            obs_energy_bin = []
            obs_bin_min = []
            obs_bin_max = []
            obs_bin_width = []
            n = 0
            for n in range(len(obs[m])):
                # if integral flux turn to differential flux (rudimentary for now)
                if obs[m][n]['energy_max'] == obs[m][n]['energy_min'] or obs[m][n]['energy_max'] == -1:
                    obs_energy_bin.append(obs[m][n]['energy_min'])
                    obs_fluence_array.append(obs[m][n]['fluence']) # -obs[m][n+1]['fluence']
                else:
                    obs_energy_bin.append((obs[m][n]['energy_max']+obs[m][n]['energy_min'])/2)
                    # obs_bin_min.append(obs_energy_bin[m][n] - obs[m][n]['energy_min'])
                    # obs_bin_max.append(obs[m][n]['energy_max'] - obs_energy_bin[m][n])
                    obs_fluence_array.append(obs[m][n]['fluence'])
            if len(obs_fluence_array) != len(obs[m]):
                obs_fluence_array.append(obs[m][n+1]['fluence'])
                obs_energy_bin.append(obs[m][n]['energy_min'])
                
            
        
            new_obs_fluence, new_bins = interpolate_fluence(obs_energy_bin, obs_fluence_array, model_energy_bin, model_fluence_array)
            
            
            
            fig = plt.figure(figsize=(20,10))
            ax = plt.subplot(111)
            ax.plot(obs_energy_bin, obs_fluence_array, 'o', label = 'Observations', color = 'Red') 
            ax.plot(model_energy_bin, model_fluence_array, 'P', label = 'Model', color = 'Blue') 
            ax.plot(new_bins, new_obs_fluence, 'o', label = 'Interpolated Obs Fluence', color = 'Black') 
            plt.yscale("log")
            plt.xscale('log')
            ax.legend(loc = 'lower left', fontsize='10', framealpha=1.0)
            # plt.show()
            figname = config.outpath + '/plots/Fluence_' + str(events[m]).rsplit(' ')[0] + '_' + model + "_" +  energy_key + '_'  + threshold + ".pdf"
            plt.savefig(figname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            


            fluence_error, error_bins = error_fluence_calc(new_bins, new_obs_fluence, model_energy_bin, model_fluence_array)
            

            fluence_all_events.append(fluence_error)
            error_bins_all_events.append(error_bins)

    fluence_histograms(model_energy_bin, fluence_all_events, error_bins_all_events, model, energy_key, threshold)
    
    if do_fitting:
        fluence_fitting_stuff(obs, pred, events, model, threshold, energy_key)
        