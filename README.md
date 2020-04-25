# CSE-CIC-IDS2018 dataset

### Dataset Description

**classes:**

* Benign
* Bot
* Infilteration
* SQL Injection
* Brute Force -Web
* Brute Force -XSS
* FTP-BruteForce
* SSH-Bruteforce
* DDOS attack-HOIC
* DDOS attack-LOIC-UDP
* DDoS attacks-LOIC-HTTP
* DoS attacks-Hulk
* DoS attacks-SlowHTTPTest
* DoS attacks-Slowloris
* DoS attacks-GoldenEye

**files:**

* Friday-02-03-2018_TrafficForML_CICFlowMeter.csv (Benign, Bot)
* Friday-16-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DoS attacks-SlowHTTPTest, DoS attacks-Hulk)
* Friday-23-02-2018_TrafficForML_CICFlowMeter.csv (Benign, SQL Injection, Brute Force -XSS, Brute Force -Web)
* Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DDoS attacks-LOIC-HTTP)
* Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv (Benign, Infilteration)
* Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DoS attacks-GoldenEye, DoS attacks-Slowloris)
* Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv (Benign, SQL Injection, Brute Force -XSS, Brute Force -Web)
* Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv (Benign, FTP-BruteForce, SSH-Bruteforce)
* Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DDOS attack-HOIC, DDOS attack-LOIC-UDP)

**features:**

* all except thuesday file:
  * Dst Port
  * Protocol
  * Timestamp
  * Flow Duration
  * Tot Fwd Pkts
  * Tot Bwd Pkts
  * TotLen Fwd Pkts
  * TotLen Bwd Pkts
  * Fwd Pkt Len Max
  * Fwd Pkt Len Min
  * Fwd Pkt Len Mean
  * Fwd Pkt Len Std
  * Bwd Pkt Len Max
  * Bwd Pkt Len Min
  * Bwd Pkt Len Mean
  * Bwd Pkt Len Std
  * Flow Byts/s
  * Flow Pkts/s
  * Flow IAT Mean
  * Flow IAT Std
  * Flow IAT Max
  * Flow IAT Min
  * Fwd IAT Tot
  * Fwd IAT Mean
  * Fwd IAT Std
  * Fwd IAT Max
  * Fwd IAT Min
  * Bwd IAT Tot
  * Bwd IAT Mean
  * Bwd IAT Std
  * Bwd IAT Max
  * Bwd IAT Min
  * Fwd PSH Flags
  * Bwd PSH Flags
  * Fwd URG Flags
  * Bwd URG Flags
  * Fwd Header Len
  * Bwd Header Len
  * Fwd Pkts/s
  * Bwd Pkts/s
  * Pkt Len Min
  * Pkt Len Max
  * Pkt Len Mean
  * Pkt Len Std
  * Pkt Len Var
  * FIN Flag Cnt
  * SYN Flag Cnt
  * RST Flag Cnt
  * PSH Flag Cnt
  * ACK Flag Cnt
  * URG Flag Cnt
  * CWE Flag Count
  * ECE Flag Cnt
  * Down/Up Ratio
  * Pkt Size Avg
  * Fwd Seg Size Avg
  * Bwd Seg Size Avg
  * Fwd Byts/b Avg
  * Fwd Pkts/b Avg
  * Fwd Blk Rate Avg
  * Bwd Byts/b Avg
  * Bwd Pkts/b Avg
  * Bwd Blk Rate Avg
  * Subflow Fwd Pkts
  * Subflow Fwd Byts
  * Subflow Bwd Pkts
  * Subflow Bwd Byts
  * Init Fwd Win Byts
  * Init Bwd Win Byts
  * Fwd Act Data Pkts
  * Fwd Seg Size Min
  * Active Mean
  * Active Std
  * Active Max
  * Active Min
  * Idle Mean
  * Idle Std
  * Idle Max
  * Idle Min
  * Label
* thuesday file header has 4 more columns:
  * Flow ID
  * Src IP
  * Src Port
  * Dst IP
  
**features transformation:**

* features that have only one value were removed:
    * Bwd Blk Rate Avg, Bwd Byts/b Avg, Bwd PSH Flags, Bwd Pkts/b Avg, Bwd URG Flags, Fwd Blk Rate Avg, Fwd Byts/b Avg, Fwd Pkts/b Avg
* also Timestamp was removed
* some features contain "inf" and "nan" value:
    * "nan" was replaced by value that has the biggest occurrence
    * "inf" was replaced by the biggest value
    * Flow Byts/s - "nan" was replaced by 0 and "inf" by 1806642857.14286
    * Flow Pkts/s - "nan" was replaced by 1000000 and "inf" by 6000000.0
* all features were normalized by mean and standard deviation   
  
**classes ratio:**

* Number of samples: 16 232 943
* Benign: 13 484 708 (83.0700139%)
* Bot: 286 191 (1.7630260%)
* Infilteration: 161 934 (0.9975640%)
* SQL Injection: 87 (0.0005359%)
* Brute Force -Web: 611 (0.0037640%)
* Brute Force -XSS: 230 (0.0014169%)
* FTP-BruteForce: 193 360 (1.1911580%)
* SSH-Bruteforce: 187 589 (1.1556068%)
* DDOS attack-HOIC: 686 012 (4.2260482%)
* DDOS attack-LOIC-UDP: 1 730 (0.0106573%)
* DDoS attacks-LOIC-HTTP: 576 191 (3.5495166%)
* DoS attacks-Hulk: 461 912 (2.8455222%)
* DoS attacks-SlowHTTPTest: 139 890 (0.8617661%)
* DoS attacks-Slowloris: 10 990 (0.0677018%)
* DoS attacks-GoldenEye: 41 508 (0.2557022%)

**Train/test/validation split:**

* full train dataset contains 80% of data (12 986 236 rows)
* test dataset contains 20% of data (3 246 707 rows)
* full train was split on train (80%, 10 388 967) and validation (20%, 2 597 269 rows)

**Explained variance:**

* for x < 19 components - not very high 
* 19 components - 0.886
* 21 components - 0.910
* 27 components - 0.960
* 29 components - 0.972
* 31 components - 0.983
* 33 components - 0.992
* 35 components - 0.995
* for x > 35 components - almost 1.0

**feature selection (Decision tree):**
* class Benign
    * best features and only features - 'Idle Min', 'Fwd IAT Tot', 'Bwd IAT Std', 'Bwd IAT Mean',
      'PSH Flag Cnt', 'Active Std', 'Flow Pkts/s', 'Fwd Pkt Len Max', 'Bwd Pkts/s', 'Bwd IAT Tot',
      'URG Flag Cnt', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd IAT Mean', 'Init Bwd Win Byts',
      'Fwd Pkts/s', 'Fwd Header Len', 'Flow Duration', 'Bwd Seg Size Avg', 'Fwd IAT Std',
      'Bwd Pkt Len Mean', 'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Dst Port', 'Subflow Fwd Byts'
    * 25 features indexes - 69, 21, 28, 27, 45, 63, 16, 7, 36, 26, 47, 29, 30, 22,
      59, 35, 33, 2, 53, 23, 13, 61, 58, 0, 55
    * t-test p-value - 0.5310
* class Bot
    * best features and only features - 'Bwd Pkt Len Mean', 'Dst Port'
    * 2 features indexes - 13, 0
    * t-test p-value - 0.3764
* class Brute Force -Web
    * best features and only features - 'Fwd IAT Min', 'Flow Pkts/s', 'Down/Up Ratio',
     'Fwd Pkts/s', 'Bwd IAT Mean', 'Bwd Pkts/s', 'Flow IAT Max', 'Dst Port', 'Init Fwd Win Byts',
    * 9 features indexes - 25, 16, 50, 35, 27, 36, 19, 0, 58
    * t-test p-value - 0.4482
* class Brute Force -XSS
    * best features and only features - 'Fwd IAT Max', 'Flow IAT Min', 'Bwd Pkt Len Max',
      'Idle Min', 'Init Bwd Win Byts', 'Fwd IAT Min', 'Fwd Act Data Pkts',
      'Fwd Header Len', 'Fwd IAT Std', 'Bwd Pkt Len Std', 'Bwd IAT Min',
      'Fwd Seg Size Avg', 'Flow IAT Max', 'Flow Byts/s', 'Bwd IAT Tot',
      'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd IAT Max', 'Dst Port', 'Bwd Pkts/s'
    * 20 features indexes - 24, 20, 11, 69, 59, 25, 60, 33, 23, 14, 30, 52, 19, 15, 26, 61, 58, 29, 0, 36
    * t-test p-value - 0.3571
* class DDOS attack-HOIC
    * best features and only features - 'ACK Flag Cnt', 'Flow IAT Max', 'Fwd Pkts/s',
     'Init Fwd Win Byts', 'Dst Port'
    * 5 features indexes - 46, 19, 35, 58, 0
    * t-test p-value - 0.5799
* class DDOS attack-LOIC-UDP
    * best features and only features - 'Fwd Act Data Pkts'
    * 1 feature indexe - 60
    * t-test p-value - 0.9521
* class DDoS attacks-LOIC-HTTP
    * best features and only features - 'Init Fwd Win Byts', 'Dst Port', 'Flow IAT Min',
      'Flow Duration', 'Bwd Pkt Len Std'
    * 5 features indexes - 58, 0, 20, 2, 14
    * t-test p-value - 0.3553
* class DoS attacks-GoldenEye
    * best features and only features - 'Pkt Len Var', 'TotLen Fwd Pkts', 'Pkt Len Max',
      'Subflow Bwd Byts', 'Fwd Seg Size Min', 'Bwd IAT Std', 'Init Fwd Win Byts',
      'Flow IAT Mean', 'Flow IAT Min', 'Bwd Pkt Len Std'
    * 10 features indexes - 41, 5, 38, 57, 61, 28, 58, 17, 20, 14
    * t-test p-value - 0.4692
* class DoS attacks-Hulk
    * best features and only features - 'Fwd Pkt Len Max', 'Dst Port', 'Bwd Pkt Len Std',
      'Subflow Bwd Byts', 'Flow Byts/s', 'Fwd IAT Min', 'Fwd Seg Size Min', 'Tot Bwd Pkts'
    * 8 features indexes - 7, 0, 14, 57, 15, 25, 61, 4
    * t-test p-value - 0.4069
* class DoS attacks-SlowHTTPTest
    * best features and only features - 'Bwd IAT Std', 'Bwd IAT Max', 'Init Fwd Win Byts',
      'Dst Port', 'Bwd Pkts/s', 'Flow Pkts/s', 'Fwd Pkts/s', 'Fwd Seg Size Min'
    * 8 features indexes - 28, 29, 58, 0, 36, 16, 35, 61
    * t-test p-value - 0.9995
* class DoS attacks-Slowloris
    * best features and only features - 'Fwd Pkt Len Mean', 'Flow Byts/s', 'Active Min',
      'Flow IAT Std', 'Pkt Len Var', 'Fwd Header Len', 'Fwd Act Data Pkts', 'Flow Duration',
      'Pkt Len Std', 'Subflow Bwd Pkts', 'Fwd Pkts/s', 'Bwd Pkt Len Std',
      'Bwd Pkt Len Max', 'Pkt Size Avg', 'Init Fwd Win Byts', 'Fwd PSH Flags',
      'Idle Min', 'Fwd IAT Mean', 'Fwd IAT Min', 'Dst Port', 'Pkt Len Mean',
      'Fwd Seg Size Min', 'Bwd IAT Max'
    * 23 features indexes - 9, 15, 65, 18, 41, 33, 60, 2, 40, 56, 35, 14, 11, 51, 58, 31, 69,
      22, 25, 0, 39, 61, 29
    * t-test p-value - 0.7071
* class FTP-BruteForce
    * best features and only features - 'Bwd Pkts/s', 'Flow Pkts/s', 'Fwd Seg Size Min'
    * 3 features indexes - 36, 16, 61
    * t-test p-value - 0.9997
* class Infilteration
    * best features and only features - 'Active Std', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
      'Idle Max', 'Subflow Bwd Pkts', 'Idle Mean', 'Idle Std', 'Subflow Bwd Byts',
      'Active Max', 'Down/Up Ratio', 'Active Mean', 'URG Flag Cnt', 'Idle Min',
      'Bwd Header Len', 'Subflow Fwd Byts', 'Active Min', 'PSH Flag Cnt', 'Pkt Len Min',
      'TotLen Fwd Pkts', 'Fwd Pkt Len Min', 'Fwd IAT Min', 'Flow IAT Std',
      'Fwd IAT Std', 'Fwd Act Data Pkts', 'Flow IAT Max', 'Fwd Seg Size Avg',
      'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Pkt Len Max', 'Fwd Pkt Len Max',
      'Fwd Pkt Len Mean', 'Flow Duration', 'Flow IAT Min', 'Pkt Len Var',
      'Fwd Pkt Len Std', 'Bwd Seg Size Avg', 'Pkt Len Std', 'Subflow Fwd Pkts',
      'Fwd IAT Mean', 'TotLen Bwd Pkts', 'Init Bwd Win Byts', 'Pkt Len Mean,'
      'Flow Byts/s', 'Bwd Pkt Len Std', 'Pkt Size Avg', 'Bwd IAT Tot',
      'Bwd IAT Std', 'Bwd IAT Mean', 'Fwd Header Len', 'Flow IAT Mean',
      'RST Flag Cnt', 'ACK Flag Cnt', 'Bwd IAT Min', 'Flow Pkts/s', 'Fwd IAT Max',
      'Bwd Pkt Len Max', 'Bwd IAT Max', 'Fwd Pkts/s', 'Fwd Seg Size Min',
      'Fwd IAT Tot', 'Bwd Pkts/s', 'Init Fwd Win Byts', 'Dst Port'
    * 63 features indexes - 63, 3, 4, 68, 56, 66, 67, 57, 64, 50, 62, 47, 69, 34, 55, 65, 45,
      37, 5, 8, 25, 18, 23, 60, 19, 52, 12, 13, 38, 7, 9, 2, 20, 41, 10, 53, 40, 54, 22, 6,
      59, 39, 15, 14, 51, 26, 28, 27, 33, 17, 44, 46, 30, 16, 24, 11, 29, 35, 61, 21, 36,
      58, 0
    * t-test p-value - 0.9991
* class SQL Injection
    * best features and only features - 'Pkt Size Avg', 'Fwd IAT Tot', 'ECE Flag Cnt',
      'Init Fwd Win Byts', 'Fwd Seg Size Min', 'Dst Port', 'Flow IAT Std', 'Flow Duration',
      'Fwd Pkts/s', 'Bwd Pkt Len Std'
    * 10 features indexes - 51, 21, 49, 58, 61, 0, 18, 2, 35, 14
    * t-test p-value - 0.3148
* class SSH-Bruteforce
    * best features and only features - 'URG Flag Cnt', 'Fwd Pkt Len Min', 'Bwd Pkts/s',
      'Fwd Header Len', 'Dst Port', 'Bwd Header Len'
    * 6 features indexes - 47, 8, 36, 33, 0, 34
    * t-test p-value - 0.3147

### Scripts

Each script contains runnable code, but the code is commented to not run
everything. Running everything takes too much time, so user can uncomment
only what is needed.

The project consists of 8 scripts:
* [constants.py](./constants.py)
* [data.py](./data.py)
* [data_analysis.py](./data_analysis.py)
* [data_transformation.py](./data_transformation.py)
* [data_visualization.py](./data_visualization.py)
* [graphs.py](./graphs.py)
* [model_training.py](./model_training.py)
* [rules.py](./rules.py)

The script [constants.py](./constants.py) contains definitions of constants.
There are names of CSV files and names of folders for results etc. To download
the dataset you can use script [data.py](./data.py). It downloads only the
CSV files. If you want run the analysis fo the dataset, you can use the script
[data_analysis.py](./data_analysis.py). You can also configure your own analysis
(see already prepared [configurations](./conf)). The scripts [data_transformation.py](./data_transformation.py)
and [data_visualization.py](./data_visualization.py) are used for data transformation
and their visualization. You can also define your own transformation. Graphs
are created with the script [graphs.py](./graphs.py). The training is
implemented in the script [model_training.py](./model_training.py). However,
the rules creation is implemented in separated script [rules.py](./rules.py).