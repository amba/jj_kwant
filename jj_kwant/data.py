
from datetime import datetime
import pathlib
import shutil
import os.path
import sys
import json

class datafile:
    def __init__(
            self,
            file : str = "data.dat",
            params = [],
            folder : str = "",
            number_format = '%.10g',
            number_delimiter = "\t\t",
            args={}
    ):
        self.number_format = number_format
        self.number_delimiter = number_delimiter
        
        # make new datafolder
        date_string = datetime.now()
        data_folder = date_string.strftime("%Y-%m-%d_%H-%M-%S")
        
        if folder:
            data_folder = data_folder + "_" + folder

        print("creating new datafolder: ", data_folder)
        pathlib.Path(data_folder).mkdir()
        script = sys.argv[0]
        print("script: ", script)
        # copy script into
        # copy script into output folder
        
        shutil.copyfile(script, data_folder + '/' + os.path.basename(script))


        # generate datafile
        data_filename = data_folder + "/" + file
        self.datafile = open(data_filename, "w")
        self.params = params

        # write datafile header
        header = "# " + number_delimiter.join(params) + \
            number_delimiter +  "eigenvalue\n"
        self.datafile.write(header)
        
        # generate metadata file
        # copy script args into metadata

        with open(data_folder + '/args.json', 'w') as outfile:
            json.dump(args, outfile)
        
        
    def log(self, evs, params={}):
        number_format = self.number_format
        number_delimiter = self.number_delimiter
        
        params_list = []
        values = []
        line = ''
        for param in self.params:
            values.append(number_format % params[param])
        params_line = number_delimiter.join(values)

        # evs
        # one line for each eigenvalue
        for ev in evs:
            ev_value = number_format % ev
            line = params_line + number_delimiter + ev_value + "\n"
            self.datafile.write(line)
        # new block
        self.datafile.write("\n")

        # flush buffers
        self.datafile.flush()
        os.fsync(self.datafile)
            
            
            
                
        
        
        
        
        
        
        
