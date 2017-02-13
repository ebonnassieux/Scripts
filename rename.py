import os
import sys
import glob


# AUTHOR: J.B.R. OONK  (ASTRON/LEIDEN UNIV. 2015)
# - changes LTA retrieval filename to standard filename
# - run in the directory where LTA files are located 


# FILE DIRECTORY
path = "./"  #DIRECTORY

filelist = glob.glob(path+'*.tar')
print 'LIST:', filelist

#FILE STRING SEPARATORS
sp1d='%'
sp2d='2F'
extn='.MS'
extt='.tar'

#LOOP
print '#####  STARTING THE LOOP  #####'
for infile_orig in filelist:

  #GET FILE
  infiletar  = os.path.basename(infile_orig)
  infile     = infiletar
  print 'doing file: ', infile

  spl1=infile.split(sp1d)[11]
  spl2=spl1.split(sp2d)[1]
  spl3=spl2.split(extn)[0]
  newname = spl3+extn+extt

  # SPECIFY FILE MV COMMAND
  command='mv ' + infile + ' ' +newname
  print command

  # CARRY OUT FILENAME CHANGE !!!
  # - COMMENT FOR TESTING OUTPUT
  # - UNCOMMENT TO PERFORM FILE MV COMMAND
  #os.system(command)

  print 'finished rename of: ', newname
