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
create_builds_option = True

# path to compiler
llvm_build_path = '/mnt/disks/data/tarindu//llvm-build'

# path to the target to compile
target_dir = '/mnt/disks/data/tarindu/llvm-project/llvm'

llvm_extract_path = os.path.join(llvm_build_path, 'bin', 'llvm-extract')
llc_path = os.path.join(llvm_build_path, 'bin', 'llc')
clang_path = os.path.join(llvm_build_path, 'bin', 'clang')
clangpp_path = os.path.join(llvm_build_path, 'bin', 'clang++')

num_workers = multiprocessing.cpu_count()

debug_file_path = os.path.join(os.path.dirname(os.getcwd()), 'debug.log')
logging.basicConfig(filename=debug_file_path, filemode='w+', level=logging.DEBUG)

def create_builds():
    script_dir = str(os.getcwd())
    
    for opt_level in opt_levels:
        build_dir = os.path.join(os.path.dirname(script_dir), dataset_name, f'{opt_level}-build')
        cmake_command = [
            'cmake', '-DLLVM_ENABLE_LIBCXX=ON',  '-DLLVM_USE_NEWPM=ON', '-DLLVM_ENABLE_PROJECTS="clang;debuginfo-tests"',
            f'-DCMAKE_C_COMPILER={clang_path}', f'-DCMAKE_CXX_COMPILER={clangpp_path}', f'-DCMAKE_C_FLAGS_RELEASE="-{opt_level}"',
            f'-DCMAKE_CXX_FLAGS_RELEASE="-{opt_level}"', '-DLLVM_TARGETS_TO_BUILD=X86', '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=1', '-DLLVM_ENABLE_ASSERTIONS=1', f'{target_dir}' 
            ]
        llvm_headers_command = ['make', 'install-llvm-headers']
        make_command = ['make', f'-j{num_workers}']
        
        logging.debug('mkdir -p ' + build_dir +'\n')
        subprocess.run(['mkdir', '-p', build_dir], check=True)
        logging.debug('cd ' + build_dir +'\n')
        subprocess.run(['cd', build_dir], check=True)
        
        logging.debug(" ".join(cmake_command) + '\n')
        subprocess.run(cmake_command, check=True) 
        logging.debug(" ".join(llvm_headers_command) + '\n')
        subprocess.run(llvm_headers_command, check=True) 
        logging.debug(" ".join(make_command) + '\n')
        subprocess.run(make_command, check=True) 

    logging.debug('cd ' + script_dir +'\n')
    subprocess.run(['cd', script_dir], check=True)

        
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
    logging.debug(output_path + '\n')
    subprocess.run(['mkdir', '-p', output_dir], check=True)

    for opt_level in opt_levels: 
        output_dir = command_output_dir_dict[opt_level][1]
        logging.debug('mkdir -p ' + output_dir +'\n')
        subprocess.run(['mkdir', '-p', output_dir], check=True)

    # run the code feature dump command first (needs to replace function annotations creted by data dump by running O1 again)
    command = command_output_dir_dict['code-features'][0]
    logging.debug(" ".join(command) + '\n')
    subprocess.run(command, check=True)

    # create the .ll files for all optimization levels
    for opt_level in opt_levels: 
        command = command_output_dir_dict[opt_level][0]
        logging.debug(" ".join(command) + '\n')
        subprocess.run(command, check=True) 

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
        logging.debug(" ".join(command) + '\n')
        try:
            subprocess.run(command, check=True, stderr=subprocess.DEVNULL)

            command = [llc_path, ir_output_path, '-o', as_output_path]
            logging.debug(" ".join(command) + '\n')
            subprocess.run(command, check=True)

            with open(as_output_path, 'r+') as f:
                asm_files.append(f.read())
        
        # flag asm file as -1 if function not found
        except subprocess.CalledProcessError as e:
            logging.debug(e)
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
        logging.debug('mkdir -p ' + module_sub_dir +'\n')
        subprocess.run(['mkdir', '-p', module_sub_dir], check=True)
        module_dir_paths[opt_level] = [module_path, module_sub_dir]

    with open(code_features_path, 'r+') as f:
        function_code_features_list = list(map(str.strip, list(filter(None, f.read().split('####')))))

    dataset = ''

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
        
        row = function + ',' + features + ', ' + label
        dataset += row + '\n'
    
    return dataset

def get_dataset_header(command_output_dir_dict):
    code_features_path = command_output_dir_dict['code-features'][1]

    with open(code_features_path, 'r+') as f:
        function_code_features_list = list(map(str.strip, list(filter(None, f.read().split('####')))))

    features = ''
    for code_feature in function_code_features_list[0].split('\n')[1:]:
        features += code_feature.split(':')[0] + ','
    
    header = 'function, ' + features + 'label' + '\n'
    return header

if __name__ == '__main__': 
    if (create_builds_option): 
        create_builds()

    data = []
    for opt_level in opt_levels:
        json_path = os.path.join(os.path.dirname(os.getcwd()), dataset_name, f'{opt_level}-build', 'compile_commands.json')
        with open(json_path)as f:
            data.append(json.load(f))

    data = np.array(data).T.tolist()
   
    # FIX ME: remove this later
    # data = data[:4]
    # get_data_dump_commands(data[0])

    with multiprocessing.Pool(num_workers) as pool:
        command_output_dir_dict_list = list(tqdm.tqdm(pool.imap(get_data_dump_commands, data), total=len(data)))
 
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.imap(run_data_dump_commands, command_output_dir_dict_list), total=len(command_output_dir_dict_list)))

    with multiprocessing.Pool(num_workers) as pool:
        with open(os.path.join(os.path.dirname(os.getcwd()), dataset_name, 'dataset.csv'), 'w+') as f:
            header = get_dataset_header(command_output_dir_dict_list[0])
            f.write(header)
            for sub_dataset in list(tqdm.tqdm(pool.imap(get_training_dataset, command_output_dir_dict_list), total=len(command_output_dir_dict_list))):
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