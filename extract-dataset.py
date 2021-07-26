import json
import multiprocessing
import subprocess
import os
import tqdm
import logging
import numpy as np

# provide in increasing order of level of optimization
opt_levels = ['O1', 'O2', 'O3']
dataset_name = 'llvm'
num_sub_datasets = 3

# path to compiler
#llvm_build_path = '/mnt/disks/data/tarindu/llvm-build'
llvm_build_path = '/Users/tarindujayatilaka/Documents/LLVM/llvm-build'

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

Input:  {
            "directory": ..,
            "command": ..,
            "file": ..
        }

Output: {
            'O1':               [command, output_dir],
            'O2':               [command, output_dir],
            'O3':               [command, output_dir],
            'code-features':    [command, output_path]
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
        output_path = os.path.join(os.path.dirname(os.getcwd()), dataset_name, f'{opt_level}-ll{os.path.sep}', prev_output_path[:-1] + 'll')
        cmd_list_opt_level[output_anchor+2] = output_path
        cmd_list_opt_level[opt_anchor] = '-' + opt_level
        output_dir = output_path[:output_path.rfind(os.path.sep)]
        commands[opt_level] = [cmd_list_opt_level, output_dir]
    
    #command to dump code features
    code_feature_dump_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), dataset_name, f'code-features{os.path.sep}', prev_output_path[:-1] + 'txt'))
    code_feature_dump_command = commands[opt_levels[0]][0].copy()
    opt_anchor = code_feature_dump_command.index(f'-{opt_levels[0]}')
    code_feature_dump_command.insert(opt_anchor, f'-mlpm-feature-dump-path={code_feature_dump_path}')
    code_feature_dump_command.insert(opt_anchor, '-mllvm')
    code_feature_dump_command.insert(opt_anchor, '-dump-mlpm-data')
    code_feature_dump_command.insert(opt_anchor, '-mllvm')
    commands['code-features'] = [code_feature_dump_command, code_feature_dump_path]

    return commands

def run_data_dump_commands(command_output_dir_dict):
    #crete directories
    output_path = command_output_dir_dict['code-features'][1]
    output_dir = output_path[:output_path.rfind(os.path.sep)]
    debug_logger.debug(output_path + '\n')
    subprocess.run(['mkdir', '-p', output_dir], check=True)

    for opt_level in opt_levels: 
        output_dir = command_output_dir_dict[opt_level][1]
        debug_logger.debug('mkdir -p ' + output_dir +'\n')
        subprocess.run(['mkdir', '-p', output_dir], check=True)

    # run the code feature dump command first (needs to replace function annotations creted by data dump by running O1 again)
    try:
        command = command_output_dir_dict['code-features'][0]
        debug_logger.debug(" ".join(command) + '\n')
        subprocess.run(command, check=True)

        # create the .ll files for all optimization levels
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

# takes a module and returns a csv file containing training features and labels
def get_training_dataset(command_output_dir_dict):
    code_features_path = command_output_dir_dict['code-features'][1]
    module_name = code_features_path[code_features_path.rfind('/') + 1 : code_features_path.rfind('.')]

    # create sub directories for the module
    module_dir_paths = {}
    for opt_level in opt_levels: 
        parent_dir = command_output_dir_dict[opt_level][1]
        module_path = os.path.join(parent_dir, module_name + '.ll')
        module_sub_dir = os.path.join(parent_dir, module_name + '.ll.dir')
        debug_logger.debug('mkdir -p ' + module_sub_dir +'\n')
        subprocess.run(['mkdir', '-p', module_sub_dir], check=True)
        module_dir_paths[opt_level] = [module_path, module_sub_dir]

    dataset = ''
    try: 
        with open(code_features_path, 'r+') as f:
            function_code_features_list = list(map(str.strip, list(filter(None, f.read().split('####')))))

        # creates a row in the format: function, [code features], optimization level
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
            
            row = function + ', ' + features + ', ' + label + ', ' + code_features_path
            dataset += row + '\n'
    except FileNotFoundError as e:
        error_logger.warning(e)
    return dataset

def get_dataset_header(command_output_dir_dict):
    code_features_path = command_output_dir_dict['code-features'][1]

    with open(code_features_path, 'r+') as f:
        function_code_features_list = list(map(str.strip, list(filter(None, f.read().split('####')))))

    features = ''
    for code_feature in function_code_features_list[0].split('\n')[1:]:
        features += code_feature.split(':')[0] + ','
    
    header = 'function, ' + features + 'label, module_path' + '\n'
    return header

if __name__ == '__main__': 
    data = []
    for opt_level in opt_levels:
        json_path = os.path.join(os.path.dirname(os.getcwd()), dataset_name, f'{opt_level}-build', 'compile_commands.json')
        with open(json_path)as f:
            data.append(json.load(f))

    data = np.array(data).T.tolist()
   
    # FIX ME: remove this later
    data = data[:4]
    # get_data_dump_commands(data[0])

    with multiprocessing.Pool(num_workers) as pool:
        command_output_dir_dict_list = list(tqdm.tqdm(pool.imap(get_data_dump_commands, data), total=len(data)))
 
    # with multiprocessing.Pool(num_workers) as pool:
    #     list(tqdm.tqdm(pool.imap(run_data_dump_commands, command_output_dir_dict_list), total=len(command_output_dir_dict_list)))

    print("\nCreating CSVs.\n")
    header = get_dataset_header(command_output_dir_dict_list[0])
    num_modules = len(command_output_dir_dict_list)
    sub_dataset_size = int(num_modules / num_sub_datasets)
    if (num_modules / sub_dataset_size):
        num_sub_datasets = num_sub_datasets + 1
    
    print(f'Number of Modules: {num_modules}')
    print(f'Number of CSV Files: {num_sub_datasets}')
    print(f'Modules per CSV File: {sub_dataset_size}')

    with multiprocessing.Pool(num_workers) as pool:
        for i in range(num_sub_datasets):
            sub_command_output_dir_dict_list = command_output_dir_dict_list[i*sub_dataset_size: min((i+1)*sub_dataset_size, num_modules)]
            print(f"\nCreating CSV {i}.\n")
            with open(os.path.join(os.path.dirname(os.getcwd()), dataset_name, f'dataset-{i}.csv'), 'w+') as f:
                f.write(header)
                for sub_dataset in list(tqdm.tqdm(pool.imap(get_training_dataset, sub_command_output_dir_dict_list), total=len(sub_command_output_dir_dict_list))):
                    f.write(sub_dataset)



    # with open('results.txt', 'w') as f:
    #     for result in p.imap(mp_worker, filenames):
    #         # (filename, count) tuples from worker
    #         f.write('%s: %d\n' % result)

    # print(subprocess.run(['ls']))
    # subprocess.run(cmd_list, check=True)
    # command = " ".join(cmd_list)
    # print(command)

    # with open('../O1-build/compile_commands.json', 'w+') as f:
    #     json.dump(modified_data, f, indent=4)
        

# import csv   
# fields=['first','second','third']
# with open(r'name', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(fields)
