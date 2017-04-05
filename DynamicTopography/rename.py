#!/bin/python

import os, re, datetime, sys
from subprocess import call, check_output

run_in_ipython = 'IPython' in sys.modules

if run_in_ipython:
    basedir ='/mnt/volume_1/DynamicTopography/Models/'
else:
    basedir = os.path.dirname(os.path.abspath(__file__))+'/'
    
    
model_names = [
    'gld106',
    'gld107',
    'gld108', 
    'gld112',
    'gld115',
    'gld118',
    'gld119',
    'gld123',
    'gld130',
    'gld134',
    'gld136',
    'gld162',
    'gld179',
    'gld214',
    'gld215',
    'gld216',
    'gld217',
    'gld224',
    'gld225',
    'gld226',
    'gld230',
    'gld89',
    'gld91',
    'gld95',
    'gld98',
    'gld241',
    ]

if not run_in_ipython:
    if len(sys.argv) >= 2:
        model_names =  sys.argv[1:]
    else:
        print "missing arguments"
        print sys.argv[1:]
        sys.exit()

plate_frame_folder = 'PlateFrame'
mantle_frame_folder = 'MantleFrame'

#model_names = get_models(basedir)
#model_names = ['M1']

def main():
    for model_name in model_names:
        print 'renaming '+model_name+'...............'
        model_root = basedir+model_name
        if not os.path.isdir(model_root):
            print 'Unable to find folder: '+ model_root

        plate_frame_root = basedir+model_name+'/'+plate_frame_folder
        if not os.path.isdir(plate_frame_root):
            print 'Unable to find folder: '+ plate_frame_root

        mantle_frame_root = basedir+model_name+'/'+mantle_frame_folder
        if not os.path.isdir(mantle_frame_root):
            print 'Unable to find folder: '+ mantle_frame_root

        time_list = []

        for f in os.listdir(plate_frame_root):
            if not f.endswith(".nc"):
                #print f
                continue
            time_list.append(get_time(f, model_name))

        time_list = sorted(set(time_list),key=lambda x: float(x))

        print time_list
        
        lines = ['#!/bin/bash\n', '#'+str(datetime.datetime.now())+'\n', '#'+str(time_list)+'\n']
        fo = open(model_root+"/rename.sh", "w+")

        lines.append('#Plate Frame: .nc \n')
        lines += generate_cmd_lines(plate_frame_root, 'nc', time_list, model_name)

        lines.append('#Plate Frame: .jpg \n')
        lines += generate_cmd_lines(plate_frame_root, 'jpg', time_list, model_name)


        lines.append('#Mantle Frame \n')
        lines += generate_cmd_lines(mantle_frame_root, 'jpg', time_list, model_name, True)

        fo.writelines(lines)
        fo.close()
        check_output(["chmod","+x", model_root+"/rename.sh"])
        check_output([model_root+"/rename.sh"])
    print str(datetime.datetime.now()) +': done!'

def get_time(f,model_name):
    f = re.sub(model_name, '', f)#remove model name
    #print f, model_name
    numbers = re.findall(r'\d*\.\d+|\d+',f)
    #print "numbers: "
    #print numbers
    if len(numbers) == 1:
        return numbers[0]
    elif len(numbers) > 1:
        return numbers[-1]
    else:
        print 'No time found in the file name: ' + f
        return None

    
def get_models(basedir):
    ret = []
    for d in os.listdir(basedir):
        if (os.path.isdir(basedir+d) and 
            os.path.isdir(basedir+d+'/PlateFrame') and 
            os.path.isdir(basedir+d+'/MantleFrame')):
            
            ret.append(d)
    return ret
    
    
def generate_cmd_lines(folder, ext, time_list, model_name, mantle_frame=False):
    lines_ = []
    for f in os.listdir(folder):
        time = get_time(f,model_name)
        if time not in time_list:
            print 'Invalid time: ' + str(time), model_name, f
        if 'Masked' in f:
            new_name = 'Masked_' + time + '.' + ext
        else:
            new_name = time+'.'+ext
            
        if f.endswith('.'+ext) and f != new_name:
            lines_.append(
                (time,
                 'mv '+folder+'/'+f+' '+ folder+'/'+new_name+'\n'))
        else:
            continue
       
    lines_.sort(key=lambda tup: tup[0])    
    return [x[1] for x in lines_]

if __name__ == "__main__":
    main()
    #print get_models(basedir)

