import psutil
import time
import datetime
import os

path = 'parameter_studies/memory_log.csv'
exists = os.path.exists('parameter_studies/memory_log.csv')
file = open(path, 'a')
if not file:
    print('ERROR')
if not exists:
    file.write('date,time,memory (%), memory (byte), swap memory (%), swap memory (byte)' + 
               ''.join([f',process[{i}].name,process[{i}].ID,process[{i}].memory (byte)' for i in range(10)]) + '\n')


while True:
    now = datetime.datetime.now()
    now = now.strftime("%Y.%m.%d.,%H:%M:%S")

    virtual_mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()
    memory = f'{virtual_mem.percent},{virtual_mem.used},{swap_mem.percent},{swap_mem.used}'

    all_processes = psutil.process_iter(attrs=['pid', 'name', 'memory_info'])
    sorted_processes = sorted(all_processes, key=lambda x: x.info['memory_info'].rss, reverse=True)
    processes = ''.join([f'{process.info["name"]},{process.info["pid"]},{process.info["memory_info"].rss},' for process in sorted_processes[:10]])

    file.write(f'{now},{memory},{processes}\n')
    file.flush()
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")}\t{(virtual_mem.used / 1024 / 1024): .2f} MB ({virtual_mem.percent: .2f} %)')
    time.sleep(60)
