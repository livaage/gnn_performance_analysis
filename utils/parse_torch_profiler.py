import pandas as pd
import re

# the memory part has units an numbers in the same 
def split_number_unit(value):
    number = re.findall(r'[-+]?\d*\.\d+|\d+', value)[0]  # Extract the numeric part
    unit = re.findall(r'\D+', value)[-1]  # Extract the unit part
    return number, unit


def txt_to_pd(file_path: str):
    """

    :param file_path: the path for a .txt output file from a torch profiler
    :return: df: the .txt read into a pandas dataframe with float and timestamp values

    This mostly AI generated script will fail if the format of the profiler output changes

    """
    profiling_file = open(file_path, 'r')
    profiling_report = profiling_file.read()
    profiling_file.close()

    # Split the text data into lines

    lines = profiling_report.strip().split('\n')

    headers_with_empty = lines[3].split('  ')
    headers_with_spaces = list(filter(None, headers_with_empty))
    headers = [h.strip() for h in headers_with_spaces]
    # Read the data into a list of lists
    data = []
    for line in lines[5:]:
        # skip the final lines that don't have function calls, we don't know how long the function call stack is
        if line.strip() == '' or line.startswith('------') or line.startswith('Self'):
            continue
        split_line = line.split('   ')
        row = [x.strip() for x in split_line if x]
        data.append(row)
    # Create the DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Apply the function to each column and create new columns
    for col in ['CPU Mem', 'Self CPU Mem', 'CUDA Mem', 'Self CUDA Mem']:
        df[f'{col}_number'], df[f'{col}_unit'] = zip(*df[col].apply(split_number_unit))

    # Convert columns to appropriate types
    df['Self CPU %'] = df['Self CPU %'].str.rstrip('%').astype(float)
    df['Self CPU'] = pd.to_timedelta(df['Self CPU'])
    df['CPU total %'] = df['CPU total %'].str.rstrip('%').astype(float)
    df['CPU total'] = pd.to_timedelta(df['CPU total'])
    df['CPU time avg'] = pd.to_timedelta(df['CPU time avg'])
    df['Self CUDA'] = pd.to_timedelta(df['Self CUDA'])
    df['Self CUDA %'] = df['Self CUDA %'].str.rstrip('%').astype(float)
    df['CUDA total'] = pd.to_timedelta(df['CUDA total'])
    df['CUDA time avg'] = pd.to_timedelta(df['CUDA time avg'])
    df['# of Calls'] = df['# of Calls'].astype(int)
    df['CPU Mem_number'] = df['CPU Mem_number'].astype(float)
    df['Self CPU Mem_number'] = df['Self CPU Mem_number'].astype(float)
    df['CUDA Mem_number'] = df['CUDA Mem_number'].astype(float)
    df['Self CUDA Mem_number'] = df['Self CUDA Mem_number'].astype(float)

    df = df[~df['Name'].str.contains('_MultiProcessingDataLoaderIter')]
    df = df[~df['Name'].str.contains('TQDM')]

    #df = df[~df['Name'].str.contains('[pl][profile]')]
    

    return df
