import argparse
import pandas as pd
import os
import re
import logging

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'


def str2bool(v):
    # converts typical true false strings to boolean True False
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_filepath(relFilepathList):
  
  # filepath of this script
  dirName = os.path.dirname(__file__)
  
  # joining this filepath with that of each file
  filepaths = [os.path.join(dirName, f) for f in relFilepathList]
  
  return filepaths


def read_pickle(filepath):
  
  print('reading in ' + filepath)
  
  # reading pkl file into df
  obj = pd.read_pickle(filepath)
  
  return obj


def pkl_to_df(filepaths):
  
  # reading all files into a list of dfs
  dfs = [read_pickle(f) for f in filepaths]
  
  # concatting all dfs into one
  df = pd.concat(dfs, ignore_index=True)
  
  # resetting index
  df = df.reset_index(drop=True)
  
  return df


def convert_model(model):
  
  match model:
    case 'MAG4_SHARP_HMI':
      return 'MAG4 (SHARP)'
    case 'SEPMOD':
      return 'SEPMOD'
    case 'UMASEP-10':
      return 'UMASEP-10'
    case 'UMASEP-100':
      return 'UMASEP-100'
    case 'COMESEP flare only':
      return 'COMESEP (f)'
    case 'COMESEP flare+CME ':
      return 'COMESEP (f+CME)'
    case 'SEPSTER2D':
      return 'SEPSTER2D'
    case 'SEPSTER2D CME':
      return 'SEPSTER2D'
    case 'SEPSTER (Parker Spiral)':
      return 'SEPSTER (PS)'
    case 'MagPy_SHARP_HMI_CEA':
      return 'MagPy'
    case 'MFLAMPA':
      return 'M-FLAMPA'
    case 'SFS-Update':
      return 'PPS'
    case 'STAT':
      return 'STAT'
    case 'ADEPT-AFRL':
      return 'ADEPT'
    case 'ADEPT-AFRL 1hr':
      return 'ADEPT (1hr)'
    case 'ADEPT-AFRL 6hr':
      return 'ADEPT (6hr)'
    case 'ZEUS+iPATH_CME':
      return 'IPATH'
    case 'MEMPSEP_Mean':
      return 'MEMPSEP (Mean)'
    case 'MEMPSEP_Median':
      return 'MEMPSEP (Median)'
    case 'SAWS-ASPECS CME':
      return 'ASPECS (CME)'
    case 'SAWS-ASPECS CME 50%':
      return 'ASPECS (CME 50)'
    case 'SAWS-ASPECS CME 90%':
      return 'ASPECS (CME 90)'
    case 'SAWS-ASPECS CME electrons':
      return 'ASPECS (CME+e)'
    case 'SAWS-ASPECS CME electrons 50%':
      return 'ASPECS (CME+e 50)'
    case 'SAWS-ASPECS CME electrons 90%':
      return 'ASPECS (CME+e 90)'
    case 'SAWS-ASPECS CME_SOHO':
      return 'ASPECS (sCME)'
    case 'SAWS-ASPECS CME_SOHO 50%':
      return 'ASPECS (sCME 50)'
    case 'SAWS-ASPECS CME_SOHO 90%':
      return 'ASPECS (sCME 90)'
    case 'SAWS-ASPECS CME_SOHO electrons':
      return 'ASPECS (sCME+e)'
    case 'SAWS-ASPECS CME_SOHO electrons 50%':
      return 'ASPECS (sCME+e 50)'
    case 'SAWS-ASPECS CME_SOHO electrons 90%':
      return 'ASPECS (sCME+e 90)'
    case 'SAWS-ASPECS flare':
      return 'ASPECS (f)'
    case 'SAWS-ASPECS flare 50%':
      return 'ASPECS (f 50)'
    case 'SAWS-ASPECS flare 90%':
      return 'ASPECS (f 90)'
    case 'SAWS-ASPECS flare + CME':
      return 'ASPECS (f+CME)'
    case 'SAWS-ASPECS flare + CME 50%':
      return 'ASPECS (f+CME 50)'
    case 'SAWS-ASPECS flare + CME 90%':
      return 'ASPECS (f+CME 90)'
    case 'SAWS-ASPECS flare + CME electrons':
      return 'ASPECS (f+CME+e)'
    case 'SAWS-ASPECS flare + CME electrons 50%':
      return 'ASPECS (f+CME+e 50)'
    case 'SAWS-ASPECS flare + CME electrons 90%':
      return 'ASPECS (f+CME+e 90)'
    case 'SAWS-ASPECS flare + CME_SOHO':
      return 'ASPECS (f+sCME)'
    case 'SAWS-ASPECS flare + CME_SOHO 50%':
      return 'ASPECS (f+sCME 50)'
    case 'SAWS-ASPECS flare + CME_SOHO 90%':
      return 'ASPECS (f+sCME 90)'
    case 'SAWS-ASPECS flare + CME_SOHO electrons':
      return 'ASPECS (f+sCME+e)'
    case 'SAWS-ASPECS flare + CME_SOHO electrons 50%':
      return 'ASPECS (f+sCME+e 50)'
    case 'SAWS-ASPECS flare + CME_SOHO electrons 90%':
      return 'ASPECS (f+sCME+e 90)'
    case 'SAWS-ASPECS flare electrons':
      return 'ASPECS (f+e)'
    case 'SAWS-ASPECS flare electrons 50%':
      return 'ASPECS (f+e 50)'
    case 'SAWS-ASPECS flare electrons 90%':
      return 'ASPECS (f+e 90)'
    case 'Lavasa':
      return 'Lavasa'
    case 'SPREAdFAST':
      return 'SPREAdFAST'
    case 'cRT+AE10':
      return 'CRT+AE10'
    case 'SEPSAT':
      return 'SEPSAT'
    case 'UNSPELL flare':
      return 'UNSPELL (f)'
    case 'SPRINTS Post Eruptive 0-24 hrs':
      return 'SPRINTS (post 0-24)'
    case 'SPRINTS Post Eruptive 24-48 hrs':
      return 'SPRINTS (post 24-48)'
    case 'SPRINTS Post Eruptive 48-72 hrs':
      return 'SPRINTS (post 48-72)'
    case 'SPRINTS Post Eruptive 72-96 hrs':
      return 'SPRINTS (post 72-96)'
    case _:
      print(str(model) + ' not found')


def convert_energy(estring):
  
  # min.10.0.max.-1.0.units.MeV
  
  # for UNSPELL, min.10.0.max.-1.0.units.MeV_min.5.0.max.-1.0.units.MeV
  
  # emin
  try:
    s = str(re.escape('min.'))
    e = str(re.escape('.max'))
    emin = float(re.findall(s+"(.*)"+e, estring)[0])
  except:
    s = str(re.escape('min.'))
    e = str(re.escape('.max.-1.0.units.MeV_'))
    emin = float(re.findall(s+"(.*)"+e, estring)[0])
  
  return emin


def convert_threshold(tstring):
  
  # threshold.10.0.units.1 / (cm2 s sr)
  
  s = str(re.escape('threshold.'))
  e = str(re.escape('.units'))
  thresh = float(re.findall(s+"(.*)"+e, tstring)[0])
  
  return thresh


def convert_dt(dtstring):
  
  if not pd.isnull(dtstring):
    # convert string to pd timestamp
    dtts = pd.to_datetime(dtstring)
    
    # convert pd timestamp to formatted string
    dtstr = dtts.strftime('%Y-%m-%d %H:%M:%S')
  else:
    dtstr = None
  
  return dtstr


def build_model_profile_df(files):
  
  if isinstance(files, str):
    # splitting files by comma
    files = files.split(',')
    
    # reading profiles into list of dfs
    dfs = [readtxt_model(f) for f in files]
    
    dfall = pd.concat(dfs, ignore_index=True)
    dfall = dfall.sort_values(by=['dt'])
    dfall = dfall.drop_duplicates(subset=['dt'])
    
    # resetting index
    dfall = dfall.reset_index(drop=True)
  
  else:
    
    dfall = pd.DataFrame(columns=['dt', 'flux'])
  
  return dfall


def readtxt_model(file):
  
  # open the text file in the row
  # load into pandas df
  # use the row's observed start date
  
  try:
    df = pd.read_csv(file, sep='    ', names=['dt', 'flux'], header=None,  engine='python')
    df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%dT%H:%M:%SZ')#.dt.tz_convert(None)
  except:
    df = pd.read_csv(file, sep=' ', names=['dt', 'flux'], header=None,  engine='python')
    df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%dT%H:%M:%SZ')#.dt.tz_convert(None)
  
  return df


def build_profile_df(files, dtstart, dtend):
  if isinstance(files, str):
    # splitting files by comma
    files = files.split(',')
    
    # reading profiles into list of dfs
    dfs = [readtxt(f, dtstart, dtend) for f in files]
    
    dfall = pd.concat(dfs, ignore_index=True)
    dfall = dfall.sort_values(by=['dt'])
    dfall = dfall.drop_duplicates(subset=['dt'])
    # resetting index
    dfall = dfall.reset_index(drop=True)
  else:
    dfall = pd.DataFrame(columns=['dt', 'flux'])
  return dfall


def readtxt(file, dtstart, dtend):
  
  # open the text file in the row
  # load into pandas df
  # use the row's observed start date
  df = pd.read_csv(file, sep='    ', names=['dt', 'flux'], header=None,  engine='python')
  
  df['dt'] = pd.to_datetime(df['dt']).dt.tz_convert(None)
  
  # converting start and ends dates to dts
  dtstart = pd.to_datetime(dtstart)
  dtend = pd.to_datetime(dtend)
  
  # 1 day prior to start and 1 day after end
  dtstart -= pd.Timedelta(hours=12)
  dtend += pd.Timedelta(hours=12)
  
  # filtering df by start and end dts
  mask = (df['dt'] >= dtstart) & (df['dt'] <= dtend)
  df = df.loc[mask]
  
  return df


def convert_dts_to_strings(df):
  
  # converting datetimes to strings
  df_dt = df.select_dtypes(include=['datetime64'])
  dt_keys = list(df_dt.keys())
  for dt in dt_keys:
    df[dt] = df[dt].apply(lambda x: convert_dt(x))
  
  return df


def populate_time_profiles(df):
  
  print('reading in time profiles')
  
  # setting up df columns
  df[['forecasted time profile x', 'forecasted time profile y', 'observed time profile x', 'observed time profile y']] = None
  
  for index, row in df.iterrows():
    if row['Predicted Time Profile'] is not None:
      
      filepath_model = row['Predicted Time Profile']
      
      # reading model time profile into df
      df_model = build_model_profile_df(filepath_model)
      
      # putting model time profiles into the full df
      df.loc[:, 'forecasted time profile x'].loc[index] = None if df_model.empty else df_model['dt'].astype(str).tolist()
      df.loc[:, 'forecasted time profile y'].loc[index] = None if df_model.empty else df_model['flux'].tolist()
      
      # getting observation time profile
      df_obs = build_profile_df(row['Observed Time Profile'], row['observed start time'], row['observed end time'])
      
      # putting observed time profiles into the full df
      df.loc[:, 'observed time profile x'].loc[index] = None if df_obs.empty else df_obs['dt'].astype(str).tolist()
      df.loc[:, 'observed time profile y'].loc[index] = None if df_obs.empty else df_obs['flux'].tolist()
  
  df = df.drop(columns=['Forecast Source', 'Forecast Path', 'Observed Time Profile', 'Predicted Time Profile'])
  
  return df


def drop_unwanted_cols(df):
  
  # dropping unwanted columns
  unwanted_keys = ['Mismatch Allowed']
  unwanted_triggers = ['CME Catalog',
                       'Number of CMEs',
                       'Number of Flares',
                       'CME Start Time',
                       'CME Liftoff Time',
                       'Flare Start Time',
                       'Flare Peak Time',
                       'Flare End Time',
                       'Flare Last Data Time',
                       'Flare Integrated Intensity',
                       'Flare NOAA AR',
                       'CME PA']
  unwanted_observations = ['Observatory',
                           'Observed SEP Fluence Units',
                           'Observed SEP Fluence Spectrum',
                           'Observed SEP Fluence Spectrum Units',
                           'Observed Max Flux in Prediction Window',
                           'Observed Max Flux in Prediction Window Time',
                           'Observed Max Flux in Prediction Window Units',
                           'Observed Point Intensity',
                           'Observed Point Intensity Time',
                           'Observed Point Intensity Units',
                           'Observed SEP Peak Intensity (Onset Peak) Units',
                           'Observed SEP Peak Intensity Max (Max Flux) Units']
  unwanted_predictions = ['Predicted SEP Threshold Crossing Time',
                          'Predicted SEP Fluence Units',
                          'Predicted SEP Fluence Spectrum',
                          'Predicted SEP Fluence Spectrum Units',
                          'Prediction Energy Channel Key',
                          'Prediction Threshold Key',
                          'Predicted Point Intensity',
                          'Predicted Point Intensity Time',
                          'Predicted Point Intensity Units',
                          'Predicted SEP Peak Intensity (Onset Peak) Units',
                          'Predicted SEP Peak Intensity Max (Max Flux) Units']
  unwanted_status_keys = [k for k in list(df.keys()) if 'Status' in k]
  all_unwanted_keys = [*unwanted_keys, *unwanted_triggers, *unwanted_observations, *unwanted_predictions, *unwanted_status_keys]
  df = df.drop(columns=all_unwanted_keys, errors='ignore')
  
  return df


def rename_cols(df):
  
  # renaming keys so they're shorter and easier to use
  keys_to_rename = {'Observed SEP All Clear': 'observed all-clear',
                    'Observed SEP Probability': 'observed probability',
                    'Observed SEP Threshold Crossing Time': 'sep date',
                    'Observed SEP Start Time': 'observed start time',
                    'Observed SEP End Time': 'observed end time',
                    'Observed SEP Duration': 'observed duration',
                    'Observed SEP Fluence': 'observed fluence',
                    'Observed SEP Peak Intensity (Onset Peak)': 'observed onset peak intensity',
                    'Observed SEP Peak Intensity (Onset Peak) Time': 'observed onset peak time',
                    'Observed SEP Peak Intensity Max (Max Flux)': 'observed max intensity',
                    'Observed SEP Peak Intensity Max (Max Flux) Time': 'observed max time',
                    'Predicted SEP All Clear': 'forecasted all-clear',
                    'Predicted SEP Probability': 'forecasted probability',
                    'Predicted SEP Start Time': 'forecasted start time',
                    'Predicted SEP End Time': 'forecasted end time',
                    'Predicted SEP Duration': 'forecasted duration',
                    'Predicted SEP Fluence': 'forecasted fluence',
                    'Predicted SEP Peak Intensity (Onset Peak)': 'forecasted onset peak intensity',
                    'Predicted SEP Peak Intensity (Onset Peak) Time': 'forecasted onset peak time',
                    'Predicted SEP Peak Intensity Max (Max Flux)': 'forecasted max intensity',
                    'Predicted SEP Peak Intensity Max (Max Flux) Time': 'forecasted max time',
                    'Prediction Window Start': 'forecast window start',
                    'Prediction Window End': 'forecast window end',
                    'Forecast Issue Time': 'forecast issue time',
                    'CME Latitude': 'cme latitude',
                    'CME Longitude': 'cme longitude',
                    'CME Speed': 'cme speed',
                    'CME Half Width': 'cme half width',
                    'Flare Latitude': 'flare latitude',
                    'Flare Longitude': 'flare longitude',
                    'Flare Intensity': 'flare intensity'}
  df = df.rename(columns=keys_to_rename)
  
  return df


def save_output(df, outFile='database', perModel=False):
  
  if perModel:
    # saving a json per model so a second code can combine whichever models
    all_models = list(df['model'].unique())
    for m in all_models:
      mask = (df['model'] == m)
      df_model = df.loc[mask]
      save = m + '.json'
      logger.info('saving ' + save)
      df_model.to_json(save, orient='records', indent=2)
  else:
    save = outFile if '.json' in outFile else outFile + '.json'
    logger.info('saving ' + save)
    df.to_json(save, orient='records', indent=2)


def setup_df(dataframe, inFile='output/pkl/SPHINX_dataframe.pkl', outFile='database', perModel=False):
  
  filepaths = [inFile]
  
  # setting up initial df from pkl file
  # df = pkl_to_df(filepaths)
  df = dataframe
  
  # main workflow for converting SPHINX df to VIVID JSON
  convert_df(df, outFile, perModel)


def convert_df(df=pd.DataFrame(), outFile='database', perModel=False):
  
  # if the df is empty for some reason, read it in from the SPHINX output file
  if df.empty:
    df = setup_df()
  
  # converting model names
  df['model'] = df['Model'].apply(lambda x: convert_model(x))
  df = df.drop(columns=['Model'])
  
  # converting energy
  df['energy'] = df['Energy Channel Key'].apply(lambda x: convert_energy(x))
  df = df.drop(columns=['Energy Channel Key'])
  
  # converting threshold
  df['threshold'] = df['Threshold Key'].apply(lambda x: convert_threshold(x))
  df = df.drop(columns=['Threshold Key'])
  
  # using index as event id
  df = df.rename_axis('event id').reset_index()
  
  # dropping unwanted columns
  df = drop_unwanted_cols(df)
  
  # renaming keys so they're shorter and easier to use
  df = rename_cols(df)
  
  # reading in time profiles from model and observation TXT files
  df = populate_time_profiles(df)
  
  # converting datetimes to strings
  df = convert_dts_to_strings(df)
  
  # saving to json
  save_output(df, outFile, perModel)
  


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Converts SPHINX output to ' \
                                   'VIVID input')
  
  parser.add_argument('-i', '--input', help='Relative filepath to SPHINX ' \
                      'output file', default='output/pkl/SPHINX_dataframe.pkl',
                      required=False)
  
  parser.add_argument('-o', '--output', help='File name for saving VIVID ' \
                      'input', default='database', required=False)
  
  parser.add_argument('-p', '--permodel', help='Boolean for saving a JSON ' \
                      'per model instead of one large JSON input',
                      type=str, default=False, required=False)
  
  args = parser.parse_args()
  
  setup_df(args.input, args.output, str2bool(args.permodel))
