import json
import multiprocessing
import subprocess
import os
import tqdm
import logging
import numpy as np
from functools import partial

# provide in increasing order of level of optimization
opt_levels = ['O1', 'O2', 'O3']
dataset_name = 'llvm'
num_sub_datasets = 10
get_labels = False # set True to compare functions and get the dataset labels 

# path to compiler
llvm_build_path = '/mnt/disks/data/tarindu/llvm-build'
# llvm_build_path = '/Users/tarindujayatilaka/Documents/LLVM/llvm-build'

llvm_extract_path = os.path.join(llvm_build_path, 'bin', 'llvm-extract')
llc_path = os.path.join(llvm_build_path, 'bin', 'llc')

num_workers = multiprocessing.cpu_count()

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.DEBUG):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, 'w+')     
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

debug_log_path = os.path.join(os.path.dirname(os.getcwd()), f'{dataset_name}-debug.log')   
debug_logger = setup_logger('debug_logger', debug_log_path)

error_log_path = os.path.join(os.path.dirname(os.getcwd()), f'{dataset_name}-error.log')   
error_logger = setup_logger('error_logger', error_log_path)

"""
takes a json object as input and returns a list of commands to create .ll files for each optimization level, and dump code features.

input is the compile commands for O1, O2, and O3
Input:  [
            {   "directory": .., 
                "command": ..,
                "file": ..         },
            {   "directory": ..,
                "command": ..,
                "file": ..         },
            {   "directory": ..,
                "command": ..,
                "file": ..         }
        ]

Output: {
            'O1':               [command, output_dir, code_feature_dump_path],
            'O2':               [command, output_dir, code_feature_dump_path],
            'O3':               [command, output_dir, code_feature_dump_path]
        }

"""
def get_data_dump_commands(objs):
    commands = {}

    # change output path and optimization level
    for index, opt_level in enumerate(opt_levels):
        obj = objs[index]
        cmd = obj['command']
        cmd_list = cmd.split()

        # use -o and opt-level flags as anchors to modify the command
        output_anchor = cmd_list.index('-o')
        opt_anchor = cmd_list.index(f'-{opt_level}')

        cmd_list.insert(output_anchor, '-emit-llvm')
        prev_output_path = cmd_list[output_anchor+2]

        cmd_list_opt_level = cmd_list.copy()
        output_path = os.path.join(os.path.dirname(os.getcwd()), dataset_name, 'ir', f'{opt_level}', prev_output_path[:-1] + 'll')
        cmd_list_opt_level[output_anchor+2] = output_path
        
        code_feature_dump_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), dataset_name, 'code-features', f'{opt_level}', prev_output_path[:-1] + 'txt'))
        # cmd_list_opt_level[opt_anchor] = '-' + opt_level
        cmd_list_opt_level.insert(opt_anchor, f'-mlpm-feature-dump-path={code_feature_dump_path}')
        cmd_list_opt_level.insert(opt_anchor, '-mllvm')
        cmd_list_opt_level.insert(opt_anchor, '-dump-mlpm-data')
        cmd_list_opt_level.insert(opt_anchor, '-mllvm')
        
        output_dir = output_path[:output_path.rfind(os.path.sep)]
        
        commands[opt_level] = [cmd_list_opt_level, output_dir, code_feature_dump_path]

    return commands

def run_data_dump_commands(command_output_dir_dict):
    # crete directories
    for opt_level in opt_levels: 
        output_dir = command_output_dir_dict[opt_level][1]

        debug_logger.debug('mkdir -p ' + output_dir +'\n')
        subprocess.run(['mkdir', '-p', output_dir], check=True)
        
        code_feature_path = command_output_dir_dict[opt_level][2]
        code_feature_dir = code_feature_path[:code_feature_path.rfind(os.path.sep)]
        
        debug_logger.debug(code_feature_path + '\n')
        subprocess.run(['mkdir', '-p', code_feature_dir], check=True)

    # create the .ll files for all optimization levels and dump code features
    try:
        for opt_level in opt_levels: 
            command = command_output_dir_dict[opt_level][0]
            debug_logger.debug(" ".join(command) + '\n')
            subprocess.run(command, check=True) 
    
    except subprocess.CalledProcessError as e:
        error_logger.warning(e)


"""
compares the assembly files and returns the minimum optimization level for function

module_dir_paths =  {
                        'O1':               [module_path, module_sub_dir],
                        'O2':               [module_path, module_sub_dir],
                        'O3':               [module_path, module_sub_dir]
                    }

"""
def get_optimization_level_label(function, module_dir_paths):
    asm_files = []

    for opt_level in opt_levels: 
        module_dir_path = module_dir_paths[opt_level]
        module_path = module_dir_path[0]
        ir_output_path = os.path.join(module_dir_path[1], function + '.ll')
        as_output_path = os.path.join(module_dir_path[1], function + '.s')
        
        command = [llvm_extract_path, '-S', '-func', function, module_path, '-o', ir_output_path]
        debug_logger.debug(" ".join(command) + '\n')
        try:
            subprocess.run(command, check=True, stderr=subprocess.DEVNULL)

            command = [llc_path, ir_output_path, '-o', as_output_path]
            debug_logger.debug(" ".join(command) + '\n')
            subprocess.run(command, check=True)

            with open(as_output_path, 'r+') as f:
                asm_files.append(f.read())
        
        # flag asm file as -1 if function not found
        except subprocess.CalledProcessError as e:
            error_logger.warning(e)
            asm_files.append(-1)


    ## note: in case of multiple optimization levels, replace the following logic with performance
    ## for now, assume the performance always increase with the optimization level

    # if function not found, assign the lowest optimization level without the function
    opt_level_label = ''
    for i in range(len(opt_levels)):
        if asm_files[i] == -1:
            opt_level_label += opt_levels[i] + '-'
            
    if (opt_level_label):
         opt_level_label += 'FNF'
    # if all of the optimization levels had the function, assign the label as follows
    else:
    # start from the end and compare adjacent optimization levels
    # if two adjacents optimization levels produce the same asm, assign the lower optimization level
        index = len(opt_levels) - 1
        opt_level_label = opt_levels[index]
        for _ in range(len(opt_levels) - 1):
            if asm_files[index] == asm_files[index-1]:
                index = index-1
                opt_level_label = opt_levels[index]
            else:
                break

    return opt_level_label

# takes a module and returns a csv containing training features and labels
"""
command_output_dir_dict =   {
                                'O1':   [command, output_dir, code_feature_dump_path],
                                'O2':   [command, output_dir, code_feature_dump_path],
                                'O3':   [command, output_dir, code_feature_dump_path]
                            }
"""
def get_training_dataset(command_output_dir_dict, dataset_opt_level):
    code_features_path = command_output_dir_dict[dataset_opt_level][2]
    print(dataset_opt_level)
    print(code_features_path)
    module_name = code_features_path[code_features_path.rfind('/') + 1 : code_features_path.rfind('.')]

    dataset = ''
    try: 
        with open(code_features_path, 'r+') as f:
            function_code_features_list = list(map(str.strip, list(filter(None, f.read().split('####')))))

        # only get labels for the first opt level to prevent redundancy
        if (get_labels and dataset_opt_level==opt_levels[0]):
            # create sub directories for the module to extract functions
            module_dir_paths = {}
            for opt_level in opt_levels: 
                parent_dir = command_output_dir_dict[opt_level][1]
                module_path = os.path.join(parent_dir, module_name + '.ll')
                module_sub_dir = os.path.join(parent_dir, module_name + '.ll.dir')
                debug_logger.debug('mkdir -p ' + module_sub_dir +'\n')
                subprocess.run(['mkdir', '-p', module_sub_dir], check=True)
                module_dir_paths[opt_level] = [module_path, module_sub_dir]
                
            # creates a row in the format: module path, function, [code features], optimization level label
            for function_code_features in function_code_features_list:
                function_code_features = function_code_features.split('\n')
                
                # index
                function = function_code_features[0]
                
                # code features list
                features = ''
                for code_feature in function_code_features[1:]:
                    features += code_feature.split(':')[-1] + ','
                features = features[:-1]

                # get the training label
                label = get_optimization_level_label(function, module_dir_paths)
                
                row = code_features_path + ', ' + function + ', ' + features + ', ' + label
                dataset += row + '\n'
        
        else:
            # creates a row in the format: module path, function, [code features]
            for function_code_features in function_code_features_list:
                function_code_features = function_code_features.split('\n')
                
                # index
                function = function_code_features[0]
                
                # code features list
                features = ''
                for code_feature in function_code_features[1:]:
                    features += code_feature.split(':')[-1] + ','
                features = features[:-1]
                
                row = code_features_path + ', ' + function + ', ' + features
                dataset += row + '\n'

    except FileNotFoundError as e:
        error_logger.warning(e)
    
    return dataset

def get_dataset_header(command_output_dir_dict):
    code_features_path = command_output_dir_dict[opt_levels[0]][2]

    with open(code_features_path, 'r+') as f:
        function_code_features_list = list(map(str.strip, list(filter(None, f.read().split('####')))))

    features = ''
    for code_feature in function_code_features_list[0].split('\n')[1:]:
        features += code_feature.split(':')[0] + ','
    
    header = 'module_path, function, ' + features + 'label' + '\n'
    return header

if __name__ == '__main__': 
    data = []
    for opt_level in opt_levels:
        json_path = os.path.join(os.path.dirname(os.getcwd()), dataset_name, f'{opt_level}-build', 'compile_commands.json')
        with open(json_path)as f:
            data.append(json.load(f))

    data = np.array(data).T.tolist()
   
    # FIX ME: remove this later
    # data = data[:1]
    # get_data_dump_commands(data[0])

    with multiprocessing.Pool(num_workers) as pool:
        command_output_dir_dict_list = list(tqdm.tqdm(pool.imap(get_data_dump_commands, data), total=len(data)))
 
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.imap(run_data_dump_commands, command_output_dir_dict_list), total=len(command_output_dir_dict_list))) 

    print("\nCreating CSVs.\n")

    header = get_dataset_header(command_output_dir_dict_list[0])

    # load balancing
    num_modules = len(command_output_dir_dict_list)
    if (num_modules > num_sub_datasets):
        sub_dataset_size = int(num_modules / num_sub_datasets)
        if (num_modules % sub_dataset_size):
            num_sub_datasets = num_sub_datasets + 1
    else:
        sub_dataset_size = num_modules
        num_sub_datasets = 1
    
    print(f'Number of Modules: {num_modules}')
    print(f'Number of CSV Files: {num_sub_datasets}')
    print(f'Modules per CSV File: {sub_dataset_size}')

    with multiprocessing.Pool(num_workers) as pool:
        for opt_level in opt_levels:
            for i in range(num_sub_datasets):
                sub_command_output_dir_dict_list = command_output_dir_dict_list[i*sub_dataset_size: min((i+1)*sub_dataset_size, num_modules)]
                
                print(f"\nCreating CSV {i}.\n")
                dataset_dir = os.path.join(os.path.dirname(os.getcwd()), dataset_name, 'datasets', opt_level)
                subprocess.run(['mkdir', '-p', dataset_dir], check=True)
                
                with open(os.path.join(os.path.dirname(os.getcwd()), dataset_name, 'datasets', opt_level, f'dataset-{i}.csv'), 'w+') as f:
                    f.write(header)
                    # setting partial function to specify the opt level
                    get_training_dataset_partial = partial(get_training_dataset, dataset_opt_level=opt_level)
                    for sub_dataset in list(tqdm.tqdm(pool.imap(get_training_dataset_partial, sub_command_output_dir_dict_list), total=len(sub_command_output_dir_dict_list))):
                        f.write(sub_dataset)


