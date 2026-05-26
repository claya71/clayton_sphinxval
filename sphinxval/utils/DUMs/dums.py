# DUMS
#
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from ..validation_json_handler import zulu_to_time, make_ccmc_zulu_time
import numpy as np
import math
import logging
import logging.config



#Create logger
logger = logging.getLogger(__name__)

def initialize_sphinx_dict():
    """ Set up a dictionary for a pandas df to hold each possible
        quantity, each observed energy channel, and predicted and
        observed values.
        
    """
    #Convert to Pandas dataframe
    #Include triggers with as much flattened info
    #If need multiple dimension, then could be used as tooltip info
    #Last CME, N CMEs, Last speed, last location, Timestamps array of all CMEs used
    

    dict = {"Model": [],
            "Observatory": [],
            "Energy Channel Key": [],
            "Threshold Key": [],
            "Mismatch Allowed": [],
            "Prediction Energy Channel Key": [],
            "Prediction Threshold Key": [],
            "Forecast Source": [],
            "Forecast Path": [],
            "Evaluation Status": [],
            "Forecast Issue Time":[],
            "Prediction Window Start": [],
            "Prediction Window End": [],
            "Observed SEP Event": [], #If an SEP event was matched, list start time for convenience

            #TRIGGER/INPUT SUMMARY TIMING INFORMATION
            "Last Trigger Time": [],
            "Last Input Time": [],
            "Last Eruption Time": [], #Last time for flare/CME

            #USEFUL SUPPLEMENTARY INFORMATION
            "Last Data Time to Issue Time": [],

            #FORECAST TRIGGERS
            "Prediction Number of CMEs": [],
            "Prediction CME Start Time": [], #Timestamp of 1st coronagraph image CME is visible in
            "Prediction CME Liftoff Time": [], #Timestamp of coronagraph
                #image with 1st indication of CME liftoff (used by CACTUS)
            "Prediction CME Latitude": [],
            "Prediction CME Longitude": [],
            "Prediction CME Speed": [],
            "Prediction CME Half Width": [],
            "Prediction CME PA": [],
            "Prediction CME Catalog": [],
            "Prediction CME Catalog ID": [],

            #KNOWN OBSERVED SEP TRIGGERS
            "Observed SEP CME Start Time": [], #Timestamp of 1st coronagraph image CME is visible in
            "Observed SEP CME Liftoff Time": [], #Timestamp of coronagraph
                #image with 1st indication of CME liftoff (used by CACTUS)
            "Observed SEP CME Latitude": [],
            "Observed SEP CME Longitude": [],
            "Observed SEP CME Speed": [],
            "Observed SEP CME Half Width": [],
            "Observed SEP CME PA": [],
            "Observed SEP CME Catalog": [],
            "Observed SEP CME Catalog ID": [],

            #FORECAST TRIGGERS
            "Prediction Number of Flares": [],
            "Prediction Flare Latitude": [],
            "Prediction Flare Longitude": [],
            "Prediction Flare Start Time": [],
            "Prediction Flare Peak Time": [],
            "Prediction Flare End Time": [],
            "Prediction Flare Last Data Time": [],
            "Prediction Flare Intensity": [],
            "Prediction Flare Integrated Intensity": [],
            "Prediction Flare NOAA AR": [],

            #KNOWN OBSERVED SEP TRIGGERS
            "Observed SEP Flare Latitude": [],
            "Observed SEP Flare Longitude": [],
            "Observed SEP Flare Start Time": [],
            "Observed SEP Flare Peak Time": [],
            "Observed SEP Flare End Time": [],
            "Observed SEP Flare Intensity": [],
            "Observed SEP Flare Integrated Intensity": [],
            "Observed SEP Flare NOAA AR": [],

            #MATCHED PREDICTED AND OBSERVED INFORMATION
            "All Clear Match Status": [],
            "Predicted SEP All Clear Probability Threshold": [],
            "Predicted SEP All Clear": [],
            "Observed SEP All Clear": [],

            "Probability Match Status": [],
            "Predicted SEP Probability": [],
            "Observed SEP Probability": [],

            "Threshold Crossing Time Match Status": [],
            "Predicted SEP Threshold Crossing Time": [],
            "Observed SEP Threshold Crossing Time": [],
            
            "Start Time Match Status": [],
            "Predicted SEP Start Time":[],
            "Observed SEP Start Time":[],
 
            "Peak Intensity Match Status": [],
            "Predicted SEP Peak Intensity (Onset Peak)": [],
            "Predicted SEP Peak Intensity (Onset Peak) Units": [],
            "Predicted SEP Peak Intensity (Onset Peak) Time": [],
            "Observed SEP Peak Intensity (Onset Peak)": [],
            "Observed SEP Peak Intensity (Onset Peak) Units": [],
            "Observed SEP Peak Intensity (Onset Peak) Time": [],

            "Peak Intensity Max Match Status": [],
            "Predicted SEP Peak Intensity Max (Max Flux)": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Units": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Time": [],
            "Observed SEP Peak Intensity Max (Max Flux)": [],
            "Observed SEP Peak Intensity Max (Max Flux) Units": [],
            "Observed SEP Peak Intensity Max (Max Flux) Time": [],

            "Observed Max Flux in Prediction Window": [],
            "Observed Max Flux in Prediction Window Units": [],
            "Observed Max Flux in Prediction Window Time": [],

            "End Time Match Status": [],
            "Predicted SEP End Time": [],
            "Observed SEP End Time": [],
            
            "Duration Match Status": [],
            "Predicted SEP Duration": [],
            "Observed SEP Duration": [],
            
            "Fluence Match Status": [],
            "Predicted SEP Fluence": [],
            "Predicted SEP Fluence Units": [],
            "Observed SEP Fluence": [],
            "Observed SEP Fluence Units": [],

            "Fluence Spectrum Match Status": [],
            "Predicted SEP Fluence Spectrum": [],
            "Predicted SEP Fluence Spectrum Units": [],
            "Observed SEP Fluence Spectrum": [],
            "Observed SEP Fluence Spectrum Units": [],

            "Time Profile Match Status": [],
            "Predicted Time Profile": [],
            "Observed Time Profile": [], #string of comma separated filenames

            "Predicted Point Intensity": [],
            "Predicted Point Intensity Units": [],
            "Predicted Point Intensity Time": [],
            "Observed Point Intensity": [],
            "Observed Point Intensity Units": [],
            "Observed Point Intensity Time": [],

            
            #MATCHING INFORMATION
            "Overlapping Observations": [],
            "All Thresholds in Prediction": [],
            "Threshold Crossed in Prediction Window": [],
            "All Threshold Crossing Times": [],
            "Eruption before Threshold Crossed": [],
            "Time Difference between Eruption and Threshold Crossing": [],
            "Farside": [],
            "Is Source Flare": [],
            "All Observation Flare Peak Times": [],
            "All Prediction Flares": [],
            "Is Source CME": [],
            "All Observation CME Start Times": [],
            "All Prediction CMEs": [],
            "Eruption in Range": [],
            "Triggers before Threshold Crossing": [],
            "Inputs before Threshold Crossing": [],
            "Triggers before Peak Intensity": [],
            "Time Difference between Triggers and Peak Intensity": [],
            "Inputs before Peak Intensity": [],
            "Time Difference between Inputs and Peak Intensity": [],
            "Triggers before Peak Intensity Max": [],
            "Time Difference between Triggers and Peak Intensity Max": [],
            "Inputs before Peak Intensity Max": [],
            "Time Difference between Inputs and Peak Intensity Max": [],
            "Triggers before SEP End": [],
            "Time Difference between Triggers and SEP End": [],
            "Inputs before SEP End": [],
            "Time Difference between Inputs and SEP End": [],
            "Prediction Window Overlap with Observed SEP Event": [],
            "Ongoing SEP Event": [],
            "Trigger Advance Time": [],
            "Original Model Short Name": []
            
            }

    return dict



def feeder_from_sphinx(sphinx_df):
    """
    Input Function
    
    

    Inputs:
        df : dataframe
            SPHINX evaluated dataframe


    What we need from SPHINX:
        Observation List
        Model Type?

    
    From Observation list - DUM associated with observation 

    DUM Models:
        Canonical Profile  (Duration, Max/Onset/Timings, Profile)
        Median Peaks (Max/Onset)
        F10.7 Prob (Prob, Contingency)

        

    Outputs:
        New Lines to sphinx_evaluated containing DUM model 'forecasts'


    """
    logger.info("Initiate DUM Models")
    model_names = sphinx_df['Model'].unique()

    # dum_model_dictionary = model_to_DUM_dictionary
    # keywords = dum_model_dictionary.keys()
    # follow_model = False
    # if follow_model:
    #     for model in model_names:
    #         model_df =  df[df['Model'] == model]
    #         foo = [x for x in keywords if x in model][0]
    #         dum_model_type = dum_model_dictionary[foo]    
    #         dum_dict = initiate_dum(model_df, dum_model_type)
    #         df = pd.concat([df, dum_df], ignore_index=True)
    # else:
    dum_df, dum_profs = canonical_prof_dum(sphinx_df)
    
    return dum_df, dum_profs


def model_to_DUM_dictionary():
    dict = {
        'MAG': 'f10.7',
        'SEPSTER': 'peaks',
        'iPATH': 'canonicalprofile',
        'SEPMOD': 'canonicalprofile',
        'SPRINTS': 'f10.7',
        'GSU': 'f10.7',
        'ASPECS': ['f10.7', 'canonicalprofile']
    }
    return dict


def initiate_dum(df, dum_type):
    """ 
    Take model name and observations to generate dum forecast

    This would be for a 'following model DUM' since it can 
    only generate a forecast when the model gives a forecast
    for the observed SEP event
    """
    
    for dums in dum_type:
        df = dum_switch_func(dums, df)


    return df



def dum_switch_func(dum_type, df):
    func ={
        'f10.7': prob_dum,
        'peaks': median_peak_dum,
        'canonicalprofile': canonical_prof_dum
    }.get(dum_type)

    if not callable(func):
        logger.error(str(dum_type) + " is an invalid dum model.")
        raise ValueError(str(dum_type) + " is an invalid dum model.")
    
    df = func(df)

    return df


def prob_dum(df):
    print('This doesnt exist yet, try again later')
    return


def robust_timing(time):
    if type(time) == str and time != 'NaT':
        format_code = "%Y-%m-%d %H:%M:%S"
        time = datetime.strptime(time, format_code)
    return time
    

def canonical_prof_dum(df):
    """
    For each event the original model forecasted for,
    make a 'forecast' for the DUM model
    """

    # energy_channels = df["Energy Channel Key"].unique()
    energy_channels = ['min.10.0.max.-1.0.units.MeV', 'min.100.0.max.-1.0.units.MeV', 'min.30.0.max.-1.0.units.MeV', 'min.50.0.max.-1.0.units.MeV']
    dum_profs = {}
    # print('len of initial df', len(df))
    for available_energies in energy_channels:
        # Loop over energies
        
        sub_df = df[df["Energy Channel Key"] == available_energies]
        # print('len of sub_df for energies', len(sub_df))
        trigger_columns = ['Observed SEP CME Start Time', 'Observed SEP CME Longitude', 'Observed SEP CME Latitude', 'Observed SEP CME Liftoff Time', 'Observed SEP CME Speed', 'Observed SEP CME Half Width', 'Observed SEP CME PA',\
             'Observed SEP Flare Start Time', 'Observed SEP Flare Peak Time', 'Observed SEP Flare End Time', 'Observed SEP Flare Longitude', 'Observed SEP Flare Latitude', 'Observed SEP Flare NOAA AR']
        observed_events = sub_df['Observed SEP Start Time'].unique()
        bounds_data = pd.read_csv('./sphinxval/utils/DUMS/bounds_data.csv')
        canonical_profile_values = canonical_profile_dictionary()[available_energies]
        low_bound = bounds_data['Longitude Low Bound'].iloc[0]
        high_bound = bounds_data['Longitude High Bound'].iloc[0]
        # print('number of observed events', len(observed_events))
        for event in observed_events:
            
            event_block = sub_df[sub_df['Observed SEP Start Time'] == event]
            # print('len of event block', len(event_block))
            
            trigger_subset = event_block[trigger_columns].copy()
            
            unique_triggers = trigger_subset.drop_duplicates()
                
            
            if len(unique_triggers) != 0:
                for i in range(len(unique_triggers)):
                    
                    # print('i for iterations', i)
                    trigger_df = unique_triggers.iloc[i].to_frame().T
                    # print(trigger_df)
                    # print(event_block)
                    # For some reason reading in from a resume file and newly matched runs don't have the same type for CME Start Time???????/??
                    for cols in trigger_columns:
                        # print(cols, event_block[cols].dtype, trigger_df[cols].dtype)
                        if str(trigger_df[cols].dtype) == 'datetime64[ns]':
                            event_block[cols] = pd.to_datetime(event_block[cols])
                    
                    trigger_block = pd.merge(event_block.reset_index(drop = True), trigger_df, how = 'inner')
                    if 'Canonical Profile DUM' in trigger_block['Model'].to_list():
                        # This event and trigger has already been analyzed for DUM Profile - continue on 
                        continue
                    else:
                        current_event = trigger_block.iloc[0].copy() # selecting one of the rows for the selected trigger to copy and use for the DUM information
                       
                    dum_dict = initialize_sphinx_dict()
                    
                    
                
                    
                    dum_dict['Energy Channel Key'] = available_energies
                    dum_dict['Threshold Key'] = current_event['Threshold Key']
                    dum_dict['Mismatch Allowed'] = False
                    dum_dict['Prediction Energy Channel Key'] = current_event['Prediction Energy Channel Key']
                    dum_dict['Prediction Threshold Key'] = current_event['Prediction Threshold Key']
                    dum_dict['Evaluation Status'] = 'Forecast is evaluated'

                    
                    # trigger information
                    dum_dict['Prediction CME Start Time'] = robust_timing(current_event['Prediction CME Start Time'])
                    dum_dict['Prediction CME Longitude'] = current_event['Prediction CME Longitude']
                    dum_dict['Prediction CME Latitude'] = current_event['Prediction CME Latitude']
                    dum_dict['Prediction CME Liftoff Time'] = robust_timing(current_event['Prediction CME Liftoff Time'])
                    dum_dict['Prediction CME Speed'] = current_event['Prediction CME Speed']
                    dum_dict['Prediction CME Half Width'] = current_event['Prediction CME Half Width']
                    dum_dict['Prediction CME PA'] = current_event['Prediction CME PA']
                    dum_dict['Prediction CME Catalog'] = current_event['Prediction CME Catalog']
                    dum_dict['Prediction CME Catalog ID'] = current_event['Prediction CME Catalog ID']
                    
                    
                    
                    
                    dum_dict['Observed SEP CME Start Time'] = robust_timing(current_event['Observed SEP CME Start Time'])
                    dum_dict['Observed SEP CME Longitude'] = current_event['Observed SEP CME Longitude']
                    dum_dict['Observed SEP CME Latitude'] = current_event['Observed SEP CME Latitude']
                    dum_dict['Observed SEP CME Liftoff Time'] = robust_timing(current_event['Observed SEP CME Liftoff Time'])
                    dum_dict['Observed SEP CME Speed'] = current_event['Observed SEP CME Speed']
                    dum_dict['Observed SEP CME Half Width'] = current_event['Observed SEP CME Half Width']
                    dum_dict['Observed SEP CME PA'] = current_event['Observed SEP CME PA']
                    dum_dict['Observed SEP CME Catalog'] = current_event['Observed SEP CME Catalog']
                    dum_dict['Observed SEP CME Catalog ID'] = current_event['Observed SEP CME Catalog ID']




                    dum_dict['Prediction Number of CMEs'] = current_event['Prediction Number of CMEs']
                    dum_dict['Prediction Number of Flares'] = current_event['Prediction Number of Flares']
                    dum_dict['Prediction Flare Latitude'] = current_event['Prediction Flare Latitude']
                    dum_dict['Prediction Flare Longitude'] = current_event['Prediction Flare Longitude']
                    dum_dict['Prediction Flare Start Time'] = robust_timing(current_event['Prediction Flare Start Time'])
                    dum_dict['Prediction Flare Peak Time'] = robust_timing(current_event['Prediction Flare Peak Time'])
                    dum_dict['Prediction Flare End Time'] = robust_timing(current_event['Prediction Flare End Time'])
                    dum_dict['Prediction Flare Last Data Time'] = robust_timing(current_event['Prediction Flare Last Data Time'])
                    dum_dict['Prediction Flare Intensity'] = current_event['Prediction Flare Intensity']
                    dum_dict['Prediction Flare Integrated Intensity'] = current_event['Prediction Flare Integrated Intensity']
                    dum_dict['Prediction Flare NOAA AR'] = current_event['Prediction Flare NOAA AR']


                    # Filling in the observed information
                    dum_dict['Observatory'] = current_event['Observatory']

                    dum_dict['Observed Time Profile'] = current_event['Observed Time Profile']
                    dum_dict['Observed SEP All Clear'] = current_event['Observed SEP All Clear']
                    dum_dict['Observed SEP Probability'] = current_event['Observed SEP Probability']
                    dum_dict['Observed SEP Threshold Crossing Time'] = current_event['Observed SEP Threshold Crossing Time']
                    dum_dict['Observed SEP Start Time'] = robust_timing(current_event['Observed SEP Start Time'])
                    dum_dict['Observed SEP Event'] = robust_timing(current_event['Observed SEP Start Time'])
                    dum_dict['Observed SEP End Time'] = robust_timing(current_event['Observed SEP End Time'])
                    dum_dict['Observed SEP Duration'] = current_event['Observed SEP Duration']
                    dum_dict['Observed SEP Fluence'] = current_event['Observed SEP Fluence']
                    dum_dict['Observed SEP Fluence Units'] = current_event['Observed SEP Fluence Units']
                    dum_dict['Observed SEP Fluence Spectrum'] = current_event['Observed SEP Fluence Spectrum']
                    dum_dict['Observed SEP Fluence Spectrum Units'] = current_event['Observed SEP Fluence Spectrum Units']
                    dum_dict['Observed SEP Peak Intensity (Onset Peak)'] = current_event['Observed SEP Peak Intensity (Onset Peak)']
                    dum_dict['Observed SEP Peak Intensity (Onset Peak) Units'] = "1 / (cm2 s sr)"
                    dum_dict['Observed SEP Peak Intensity (Onset Peak) Time'] = robust_timing(current_event['Observed SEP Peak Intensity (Onset Peak) Time'])
                    dum_dict['Observed SEP Peak Intensity Max (Max Flux)'] = current_event['Observed SEP Peak Intensity Max (Max Flux)']
                    dum_dict['Observed SEP Peak Intensity Max (Max Flux) Units'] = "1 / (cm2 s sr)"
                    dum_dict['Observed SEP Peak Intensity Max (Max Flux) Time'] = robust_timing(current_event['Observed SEP Peak Intensity Max (Max Flux) Time'])
                    dum_dict['Observed Point Intensity'] = current_event['Observed Point Intensity']
                    dum_dict['Observed Point Intensity Units'] = "1 / (cm2 s sr)"
                    dum_dict['Observed Point Intensity Time'] = robust_timing(current_event['Observed Point Intensity Time'])
                    dum_dict['Observed Max Flux in Prediction Window'] = current_event['Observed Max Flux in Prediction Window']
                    dum_dict['Observed Max Flux in Prediction Window Units'] = "1 / (cm2 s sr)"
                    dum_dict['Observed Max Flux in Prediction Window Time'] = robust_timing(current_event['Observed Max Flux in Prediction Window Time'])

                    # known prediction elements
                    dum_dict['Predicted SEP All Clear'] = False
                    dum_dict['Predicted SEP All Clear Probability Threshold'] = np.nan
                    dum_dict['All Clear Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP Probability'] = np.nan
                    dum_dict['Probability Match Status'] = 'SEP Event'
                    
                    trigger_str = ''
                    dum_string = ''
                    event_source_long = None
                    # print('longitude')
                    if not pd.isnull(current_event['Observed SEP CME Longitude']):
                        # print(current_event['CME Longitude'])
                        dum_string += ' CME'
                        event_source_long = current_event['Observed SEP CME Longitude']
                        last_trig = robust_timing(current_event['Observed SEP CME Start Time'])
                        trigger_str += '_CME_' + standard_time_def(current_event['Observed SEP CME Start Time']).replace(':','').replace('/','') + '_long_' + str(event_source_long)
                    if not pd.isnull(current_event['Observed SEP Flare Longitude']) and not pd.isnull(current_event['Observed SEP Flare Start Time']):
                        # print(current_event['Flare Longitude'])
                        dum_string += ' Flare'
                        event_source_long = current_event['Observed SEP Flare Longitude']
                        last_trig = robust_timing(current_event['Observed SEP Flare Start Time'])
                        trigger_str += '_Flare_' + standard_time_def(current_event['Observed SEP Flare Start Time']).replace(':','').replace('/','') + '_long_' + str(event_source_long)
                    if trigger_str == '':
                        trigger_str += '_' + str(i)
                    dum_dict['Last Trigger Time'] = last_trig
                    dum_dict['Last Input Time'] = pd.NaT
                    dum_dict['Last Eruption Time'] = last_trig
                    # print(event_source_long, low_bound, high_bound)
                    if pd.isnull(event_source_long):
                        location_string = 'central'
                    elif event_source_long < low_bound:
                        location_string = "east"
                    elif event_source_long > high_bound:
                        location_string = "west"
                    else:
                        location_string = "central"
                    dum_dict['Model'] = 'Canonical Profile DUM' + dum_string
                    dum_dict['Original Model Short Name'] = 'Canonical Profile DUM ' + location_string

                    canonical_profile = canonical_profile_values[location_string]
                    profile = pd.read_csv(canonical_profile['profile_filename'], index_col = 0)
                    normalized_times = profile.index.to_list()
                    start = robust_timing(current_event['Observed SEP Start Time'])
                    # print(type(start), start)
                    # input()
                    if type(start) == str:
                        if 'T' not in start or 'Z' not in start:
                            # not in zulu time either smh
                            # foo = start.rsplit(' ')
                            # start_time_str = foo[0].replace('/', '-') + 'T' + foo[1] + 'Z'
                            # print(start_time_str)
                            start_time_dt = pd.to_datetime(start)
                            start_time_str = make_ccmc_zulu_time(start_time_dt)
                        else:
                            start_time_str = start
                            start_time_dt = zulu_to_time(start_time_str)
                    else:
                        start_time_dt = start
                        start_time_str = standard_time_def(start)
                            
                    un_normalized_time = []
                    time_profile_time = []
                    flux = profile['Flux']
                
                    # input()
                    time_resolution = 5
                    end_index = 0
                    current_threshold = float(current_event['Threshold Key'].rsplit('.')[1])
                    
                    for times in range(len(normalized_times)):
                    
                        current_time = start_time_dt + float(normalized_times[times]) * timedelta(minutes = time_resolution)
                    
                        un_normalized_time.append(current_time)
                        time_profile_time.append(standard_time_def(current_time))
                        
                        if flux[times] < current_threshold and times > 3:
                            if flux[times - 1] <= current_threshold:
                                if flux[times - 2] <= current_threshold:
                                    end_index = times
                    if end_index == 0:
                        end_index = -1

                    dum_duration = (un_normalized_time[end_index] - un_normalized_time[0]).total_seconds()/(60.*60.)
                  
                    # need to use the unnormalized time to create a profile output file (bleh) so that later parts of the 
                    # validation workflow work properly
                    output_dict = {'dates': time_profile_time, 'fluxes': flux.to_list()}
                    output_df = pd.DataFrame(output_dict)
                    output_df = output_df.set_index('dates')
                    start_time_filename = start_time_str.replace('/', '').replace(':','')
                    output_filename = './model/DUMs/CanonicalProfile/DUM_CanonicalProfile_' + location_string + '_' + available_energies + '_' + start_time_filename + trigger_str + '.txt'
                    # print(output_filename)
                    # output_df.to_csv(output_filename, sep='\t', header = False)

                    dum_dict['Forecast Source'] = output_filename
                    dum_dict['Forecast Path'] = './model/DUMs/CanonicalProfile/'
                    dum_dict['Forecast Issue Time'] = pd.NaT
                    dum_dict['Prediction Window Start'] = un_normalized_time[0]
                    dum_dict['Prediction Window End'] = un_normalized_time[-1]
                    dum_dict['Predicted SEP Threshold Crossing Time'] = un_normalized_time[0]
                    dum_dict['Threshold Crossing Time Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP Start Time'] = un_normalized_time[0]
                    dum_dict['Start Time Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP End Time'] = un_normalized_time[end_index]
                    dum_dict['End Time Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP Duration'] = dum_duration
                    dum_dict['Duration Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP Fluence'] = np.nan
                    dum_dict['Predicted SEP Fluence Units'] = np.nan
                    dum_dict['Fluence Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP Fluence Spectrum'] = np.nan
                    dum_dict['Predicted SEP Fluence Spectrum Units'] = np.nan
                    dum_dict['Fluence Spectrum Match Status'] = 'SEP Event'


                    


                    
                    
                    max_peak_val = max(flux)
                    max_index  = np.where(flux == max_peak_val)[0][0]
                    
                    if canonical_profile['onset_peak_index'] == None:
                        onset_peak_index = max_index
                    else:
                        onset_peak_index = canonical_profile['onset_peak_index']

                    dum_dict['Predicted SEP Peak Intensity (Onset Peak)'] = flux[onset_peak_index]
                    dum_dict['Predicted SEP Peak Intensity (Onset Peak) Time'] = un_normalized_time[onset_peak_index]
                    dum_dict['Predicted SEP Peak Intensity (Onset Peak) Units'] = "1 / (cm2 s sr)"
                    dum_dict['Peak Intensity Match Status'] = 'SEP Event'
                    dum_dict['Predicted SEP Peak Intensity Max (Max Flux)'] = max_peak_val
                    dum_dict['Predicted SEP Peak Intensity Max (Max Flux) Time'] = un_normalized_time[max_index]
                    dum_dict['Predicted SEP Peak Intensity Max (Max Flux) Units'] = "1 / (cm2 s sr)"
                    dum_dict['Peak Intensity Max Match Status'] = 'SEP Event'

                    dum_dict['Predicted Point Intensity'] = None
                    dum_dict['Predicted Point Intensity Units'] = None
                    dum_dict['Predicted Point Intensity Time'] = pd.NaT

                    dum_dict['Predicted Time Profile'] = output_filename
                    dum_dict['Time Profile Match Status'] = 'SEP Event'

                    dum_df = pd.DataFrame([dum_dict])
                    df = pd.concat([df, dum_df], ignore_index=True)
                    dum_profs[output_filename] = output_dict


                    # End Matter of the DataFrame
                    # dum_dict['Overlapping Observations']
                    # dum_dict['All Thresholds in Prediction']
                    # dum_dict['Threshold Crossed in Prediction Window']
                    # dum_dict['All Threshold Crossing Times']
                    # dum_dict['Eruption before Threshold Crossed']
                    # dum_dict['Time Difference between Eruption and Threshold Crossing']
                    # dum_dict['Farside']
                    # dum_dict['Is Source Flare']
                    # dum_dict['All Observation Flare Peak Times']
                    # dum_dict['All Prediction Flares']
                    # dum_dict['Is Source CME']
                    # dum_dict['All Observation CME Start Times']
                    # dum_dict['All Prediction CMEs']
                    # dum_dict['Eruption in Range']
                    # dum_dict['Triggers before Threshold Crossing']
                    # dum_dict['Inputs before Threshold Crossing']
                    # dum_dict['Triggers before Peak Intensity']
                    # dum_dict['Time Difference between Triggers and Peak Intensity']
                    # dum_dict['Inputs before Peak Intensity']
                    # dum_dict['Time Difference between Inputs and Peak Intensity']
                    # dum_dict['Triggers before Peak Intensity Max']
                    # dum_dict['Time Difference between Triggers and Peak Intensity Max']
                    # dum_dict['Inputs before Peak Intensity Max']
                    # dum_dict['Time Difference between Inputs and Peak Intensity Max']
                    # dum_dict['Triggers before SEP End']
                    # dum_dict['Time Difference between Triggers and SEP End']
                    # dum_dict['Inputs before SEP End']
                    # dum_dict['Time Difference between Inputs and SEP End']
                    # dum_dict['Prediction Window Overlap with Observed SEP Event']
                    # dum_dict['Ongoing SEP Event']
                    # dum_dict['Trigger Advance Time']
                    # dum_dict['Original Model Short Name']
                    
    
    return df, dum_profs



def median_peak_dum(df):
    """
    Probably the simplest of the DUMs - gives only a peak flux prediction
    for every model prediction correctly associated with an SEP event

    Needs: dictionary of median peak flux from the benchmark dataset
    for each energy channel

    
    """

    energy_channels = df["Energy Channel Key"].unique()
    for available_energies in energy_channels:
        # Loop over energies
        sub_df = df[df["Energy Channel Key"] == available_energies]
        median_peak_values = median_peak_dictionaries(available_energies)
        onset_peak_val = median_peak_values['Onset Peak']
        max_peak_val = median_peak_values['Max Peak']
        observed_events = sub_df['Observed SEP Start Time'].unique()
        for event in observed_events:
            current_event = sub_df[sub_df['Observed SEP Start Time'] == event]
            # in cases where there are multiple forecasts associated with a single
            # observed SEP event. Just take the first one since the only thing that 
            # matters is some of the observed values 
            if len(current_event) != 0:
                dum_dict = current_event.iloc[0]
            else:
                dum_dict = current_event 
            
            # replacing certain columns of that copied row with DUM Model forecast
            # other columns will remain the same from the initial model forecast

            dum_dict['Model'] = 'Median Peak DUM following ' + current_event['Model']
            dum_dict['Predicted SEP All Clear'] = False
            dum_dict['All Clear Match Status'] = 'SEP Event' 
            dum_dict['Peak Intensity Match Status'] = 'SEP Event'
            dum_dict['Peak Intensity Max Match Status'] = 'SEP Event'
            dum_dict['Original Model Short Name'] = 'Median Peak DUM'
            dum_dict['Forecast Source'] = [current_event['Forecast Source'], 'DUM Model']
            dum_dict['Forecast Path'] = 'DUM Model following ' + current_event['Model'] 
            dum_dict['Evaluation Status'] = 'DUM Model Inserted'
            dum_dict['Forecast Issue Time'] = pd.NaT
            dum_dict['Prediction Window Start'] = current_event['Observed SEP Start Time']
            dum_dict['Prediction Window End'] = current_event['Observed SEP End Time']

            dum_dict['Predicted SEP Peak Intensity (Onset Peak)'] = onset_peak_val
            dum_dict['Predicted SEP Peak Intensity (Onset Peak) Time'] = pd.NaT
            dum_dict['Predicted SEP Peak Intensity Max (Max Flux)'] = max_peak_val
            dum_dict['Predicted SEP Peak Intensity Max (Max Flux) Time'] = pd.NaT
            

            df = pd.concat([df, dum_dict], ignore_index=True)
    
    return df


def median_peak_dictionaries(energy_channel):
    dict = {
        "min.10.0.max.-1.0.units.MeV": {
            'Onset Peak': 49.5215,
            'Max Peak': []
            },
        "min.100.0.max.-1.0.units.MeV": {
            'Onset Peak': 9.66,
            'Max Peak': []
        },
        "min.30.0.max.-1.0.units.MeV": {
            'Onset Peak': 7.7896,
            'Max Peak': []
        },
        "min.50.0.max.-1.0.units.MeV":{
            'Onset Peak': 8.6474,
            'Max Peak': []
        }
    }

    return dict[energy_channel]


def canonical_profile_dictionary():
    dict = {
        "min.10.0.max.-1.0.units.MeV": {
            'east' : {
                'profile_filename': "./sphinxval/utils/DUMs/east_canonical_profile_10.0 MeV 10.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': 61
            },
            'central': {
                'profile_filename': "./sphinxval/utils/DUMs/central_canonical_profile_10.0 MeV 10.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': 53
            },
            'west':{
                'profile_filename': "./sphinxval/utils/DUMs/west_canonical_profile_10.0 MeV 10.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            }
        },
        "min.100.0.max.-1.0.units.MeV": {
            'east' : {
                'profile_filename': "./sphinxval/utils/DUMs/east_canonical_profile_100.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            },
            'central': {
                'profile_filename': "./sphinxval/utils/DUMs/central_canonical_profile_100.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': 6
            },
            'west':{
                'profile_filename': "./sphinxval/utils/DUMs/west_canonical_profile_100.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            }
        },
        "min.30.0.max.-1.0.units.MeV": {
            'east' : {
                'profile_filename': "./sphinxval/utils/DUMs/east_canonical_profile_30.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            },
            'central': {
                'profile_filename': "./sphinxval/utils/DUMs/central_canonical_profile_30.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            },
            'west':{
                'profile_filename': "./sphinxval/utils/DUMs/west_canonical_profile_30.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            }
        },
        "min.50.0.max.-1.0.units.MeV": {
            'east' : {
                'profile_filename': "./sphinxval/utils/DUMs/east_canonical_profile_50.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': 40
            },
            'central': {
                'profile_filename': "./sphinxval/utils/DUMs/central_canonical_profile_50.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': None
            },
            'west':{
                'profile_filename': "./sphinxval/utils/DUMs/west_canonical_profile_50.0 MeV 1.0 pfu_SEP Start Time_N-deg Poly.csv",
                'onset_peak_index': 40
            }
        }

    }

    return dict


def extract_profile(profilename, start_time):
    profile = pd.read_csv(profilename, index_col = False)


    return profile_data


def standard_time_def(time):
    # Input: Time as a datetime
    # Output: converted time to standard form of YYYY-MM-DDTHH:MM:SSZ
    # print(time, type(time))
    str_time = str(time).rsplit(" ")[0] + "T" + str(time).rsplit(" ")[1] + "Z"
    return str_time