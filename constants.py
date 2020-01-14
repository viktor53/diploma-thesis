from os import sep

DATA_DIR = "data_csv"

CONF_DIR = "conf"

PATH_TO_DATA = ".." + sep + DATA_DIR

ALL_ORIGINAL_FILES = [
     "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
     "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
     "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
     "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
     "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
     "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
     "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
     "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
     "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
     "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
]

ORIGINAL_FILES_WITH_SAME_HEADER = [
    [
        "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
        "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
    ],
    [
        "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
    ]
]

SPLIT_FILES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter_0.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter_1.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter_dropped_0.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_0.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_1.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_2.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_3.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_4.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_5.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_6.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_7.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_8.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_9.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_10.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_11.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_12.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_13.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_14.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_15.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_16.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_17.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_18.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_19.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_20.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_21.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_22.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_23.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_24.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_25.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_0.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_1.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_2.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_3.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_4.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_5.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_6.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_7.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_8.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_9.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_10.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_11.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_12.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_13.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_14.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_15.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_16.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_17.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_18.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_19.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_20.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_21.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_22.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_23.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_24.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_25.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_26.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_27.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_28.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_29.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_30.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_31.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_32.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_33.csv"
]

FULL_ANALYSIS_CONF = "full_analysis.yaml"

CLASSES_ANALYSIS_CONF = "classes_analysis.yaml"

ANALYSIS_BY_PARTS = [
    "analysis_part_1.yaml",
    "analysis_part_2.yaml",
    "analysis_part_3.yaml",
    "analysis_part_4.yaml",
    "analysis_part_5.yaml",
    "analysis_part_6.yaml",
    "analysis_part_7.yaml",
    "analysis_part_8.yaml",
]

CLASSES = [
    "Benign",
    "Bot",
    "Infilteration",
    "SQL Injection",
    "Brute Force -Web",
    "Brute Force -XSS",
    "FTP-BruteForce",
    "SSH-Bruteforce",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "DoS attacks-GoldenEye"
]

FILES_WITH_RESULTS = [
    "Benign/Benign_results.csv",
    "Bot/Bot_results.csv",
    "Brute_Force_-Web/Brute_Force_-Web_results.csv",
    "Brute_Force_-XSS/Brute_Force_-XSS_results.csv",
    "DDOS_attack-HOIC/DDOS_attack-HOIC_results.csv",
    "DDOS_attack-LOIC-UDP/DDOS_attack-LOIC-UDP_results.csv",
    "DDoS_attacks-LOIC-HTTP/DDoS_attacks-LOIC-HTTP_results.csv",
    "DoS_attacks-GoldenEye/DoS_attacks-GoldenEye_results.csv",
    "DoS_attacks-SlowHTTPTest/DoS_attacks-SlowHTTPTest_results.csv",
    "DoS_attacks-Slowloris/DoS_attacks-Slowloris_results.csv",
    "FTP-BruteForce/FTP-BruteForce_results.csv",
    "Infilteration/Infilteration_results.csv",
    "SQL_Injection/SQL_Injection_results.csv",
    "SSH-Bruteforce/SSH-Bruteforce_results.csv"
]
