import argparse
import subprocess
import os


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

def push_vivid_local(database_filepath, vivid_filepath):
    """
    push vivid for local installation of sphinx and vivid. Will copy database filepath to 
    vivid's location. Meant to be part of the normal sphinx workflow - as soon as sphinx
    is done running it'll put the database where you need it to be for vivid.
    """
    is_default =  input('Is the sphinxtovivid_database.json file in its default location (default location is in ./outputs/? True/False')
    if not is_default:
        database_filepath = input()

    # Checking that the file does indeed exist in the location specified
    database_exists = os.path.isfile(database_filepath)
    
    if not database_exists:
        


    # Now that the database file is located
    



def push_vivid_toggle():
    """
    simple toggle that is intended to be used when push_vivid is on the AWS server so that
    the code will know if its there or locally installed. Will then take you to the correct
    push_vivid subroutine.
    """




"""
2 'versions': local install version, AWS version (gonna look different depending)
Find database file 
push database to vivid location

be able to run from command line and from sphinx workflow

command line arguments:
    database location/filename
    vivid location

 
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
  