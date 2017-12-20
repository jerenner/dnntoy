from __future__ import print_function

import argparse
from time import sleep
import magic
from subprocess import call
import sys, os
from datetime import datetime
from os.path import dirname
from glob import glob

# ------------------------------------------------------------------------------
# prod_voxelstosipm.py
# Modified from irene_prod.py
#
# To check:
# - input file name
# - number of events

script_name = "/home/jrenner/jerenner/dnntoy/nexustosipm/nexustosipm.py"
JOBSDIR = '/home/jrenner/analysis/MC/descape/jobsEPEM'

def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Script to produce HDF5 files')
    parser.add_argument('-j','--jobs',
                        action='store',
                        help='jobs',
                        required='True')
    parser.add_argument('-n','--nevts',
                        action='store',
                        help='number of events',
                        required='True')
    parser.add_argument('-f','--ifile',
                        action='store',
                        help='voxels input file',
                        required='True')

    return parser


#get options
args = get_parser().parse_args()
opts = vars(args) # dict

jobs = int(args.jobs)
nevts = int(args.nevts)
ifile = args.ifile

#----------- check and make dirs
def checkmakedir( path ):
    if os.path.isdir( path ):
        print('hey, directory already exists!:\n' + path)
    else:
        os.makedirs( path )
        print('creating directory...\n' + path)

#IO dirs

checkmakedir(JOBSDIR)

exec_template_file = '/home/jrenner/production/templates/voxelstosipm.sh'
exec_template = open(exec_template_file).read()
exec_params = {'jobsdir' : JOBSDIR}

#remove old jobs
jobs_files = glob(JOBSDIR + '/voxelstosipm_*.sh')
map(os.remove, jobs_files)

jobfilename = JOBSDIR + '/' + 'voxelstosipm_0.sh'
jobfile = open(jobfilename, 'w')
jobfile.write(exec_template.format(**exec_params))

count_jobs = 0
for i in range(jobs):
    if(i > 0): 
        jobfile.write('\n\necho date\ndate\n')
        jobfile.close()
        count_jobs += 1
        jobfilename = JOBSDIR + '/voxelstosipm_{}.sh'.format(count_jobs)
        jobfile = open(jobfilename, 'w')
        jobfile.write(exec_template.format(**exec_params))

    cmd = 'python {0} -j {1} -i {2} -n {3} -f {4}\n'.format(script_name,jobs,count_jobs,nevts,ifile)
    jobfile.write(cmd)
jobfile.write('\n\necho date\ndate\n')
jobfile.close()

#send jobs
for i in range(0, count_jobs+1):
    cmd = 'qsub {}/voxelstosipm_{}.sh'.format(JOBSDIR, i)
    print(cmd)
    #call(cmd, shell=True, executable='/bin/bash')
    os.system(cmd)
    sleep(0.5)

sys.exit()
