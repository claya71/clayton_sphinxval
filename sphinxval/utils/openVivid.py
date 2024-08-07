import argparse
import subprocess





def run_vivid(path):
  """
  Runs VIVID locally from the command line and opens it in a browser
  
  Parameters
  ----------
  path : string
    Absolute filepath to VIVID
  
  Returns
  -------
  None
  """
  
  try:
    command = 'npm run dev -- --open'
    p = subprocess.Popen([command], shell=True, cwd=path)
    p.wait()
  except:
    print('Check filepath to VIVID: ' + str(path))




if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='push_vivid started')
  
  # parser.add_argument('-p', '--path', help='Absolute filepath to VIVID',
                      # required=False)

              
  
  args = parser.parse_args()
  
  run_vivid(args.path)
  