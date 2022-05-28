import os
import csv
log_dir = "log"
out_csv = open("reslut.csv", "w")
csv_writer = csv.writer(out_csv)
csv_writer.writerow(["ID","TASK","MODEL","DEVICE","EPOCH_NUM","BATCH_SIZE","Adversarial_Training_type","Adversarial_init_epsilon","Adversarial_init_type","Sampling_times_theta","Sampling_times_delta","Sampling_noise_theta","Sampling_noise_delta","Sampling_step_theta","Sampling_step_delta","lambda","beta_s","beta_p"])
id = 0
for _,_,logs in  os.walk(log_dir):
    for log in logs:
        id += 1
        result = [id]
        f = open(log_dir+"/"+log)
        f_lines = f.readlines()
        if len(f_lines) != 0:
            for line in f_lines:
                if ": " in line and "EPOCH: " not in line and "Metric" not in line:
                    result.append(line[line.find(": ")+2:-1])
                elif "Best Metric:" in line:
                    result.append(line[line.find("': ")+3:-2])
            csv_writer.writerow(result)
        f.close()
out_csv.close()