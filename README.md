# IDS-2018

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

**files**:

* Friday-02-03-2018_TrafficForML_CICFlowMeter.csv (Benign, Bot)
* Friday-16-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DoS attacks-SlowHTTPTest, DoS attacks-Hulk)
* Friday-23-02-2018_TrafficForML_CICFlowMeter.csv (Benign, SQL Injection, Brute Force -XSS, Brute Force -Web)
* Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DDoS attacks-LOIC-HTTP)
* Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv (Benign, Infilteration)
* Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DoS attacks-GoldenEye, DoS attacks-Slowloris)
* Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv (Benign, SQL Injection, Brute Force -XSS, Brute Force -Web)
* Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv (Benign, FTP-BruteForce, SSH-Bruteforce)
* Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv (Benign, DDOS attack-HOIC, DDOS attack-LOIC-UDP)

**headers**:

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

![pie chart](graphs/classes_ratio.png)