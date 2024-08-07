import argparse
import subprocess
import os
import logging


logger = logging.getLogger(__name__)



# def run_vivid(path):
#   """
#   Runs VIVID locally from the command line and opens it in a browser
  
#   Parameters
#   ----------
#   path : string
#     Absolute filepath to VIVID
  
#   Returns
#   -------
#   None
#   """
  
#   try:
#     command = 'npm run dev -- --open'
#     p = subprocess.Popen([command], shell=True, cwd=path)
#     p.wait()
#   except:
#     print('Check filepath to VIVID: ' + str(path))



def push_vivid_local():
    """
    push vivid for local installation of sphinx and vivid. Will copy database filepath to 
    vivid's location. Meant to be part of the normal sphinx workflow - as soon as sphinx
    is done running it'll put the database where you need it to be for vivid.
    """
    is_default =  input('Is the sphinx_to_vivid_database.json file in its default location (default location is in ./outputs/? True/False')
    if not is_default:
        database_filepath = input('Input the filepath to your sphinx_to_vivid database')
    else:
        database_filepath = './output/'        
    # Checking that the file does indeed exist in the location specified
    database_exists = os.path.isfile(database_filepath + 'sphinx_to_vivid_database.json')
    
    if not database_exists:
        logger.info('No sphinx_to_vivid database found, check that the database_filepath is correct')
        return
        
    else:
        logger.info('sphinx_to_vivid databse found')
        vivid_location = input('Where is vivid located? Put in complete filepath for your local installation of vivid.')
        vivid_exists = os.path.isdir(vivid_location)
        if vivid_exists:
            logger.info('moving the database to your designated vivid locaton (specifically /src/database/ folder)')
            os.rename(database_filepath + 'sphinx_to_vivid_database.json', vivid_location + 'database.json')
        else:
            logger.info('local install of vivid (and its database directory) not found')



    
    



def push_vivid(push_vivid_toggle, inFile = './outputs/sphinx_to_vivid_database.json'):
    """
    Titular function. Will be the core that determines the proper distribution location of the
    database for vivid. 
    """


    





"""
2 'versions': local install version, AWS version (gonna look different depending)
Find database file 
push database to vivid location

be able to run from command line and from sphinx workflow

command line arguments:
    database location/filename
    vivid location


Create a SPHINX to Vivid local staging folder that is the designated location for vivid to grab the proper output database
local folder for now
SPHINX not on the same installation location as VIVID currently (will be determined then we will accomadate for whatever happens)  - either a shared folder mounted in the sphinx location shared b/t
sphinx and vivid or we will set up something that will grab the database from the staging location and moves it to where vivid can see it
 
"""


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description='Pushes the sphinx to vivid database to the Vivid location')
  
#   parser.add_argument('-p', '--path', help='Absolute filepath to VIVID',
#                       required=False)
  
    parser.add_argument('-database_path', '-dbp', help = 'filepath to sphinx_to_vivid database, by default its located in ./outputs/', required = False,
            default = './outputs/')

    parser.add_argument('-vivid_path', '-vp', help = 'Absolute filepath to your local installation of Vivid. Example: C:\\Users\\cfalliso\\Documents\\Codes\\SPHINX\\Dev SPHINX\\vivid-main\\src\\database\\database.json',
            required = True)

    args = parser.parse_args()
  
#   run_vivid(args.path)
  