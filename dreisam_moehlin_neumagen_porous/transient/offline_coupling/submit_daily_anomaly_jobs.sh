#!/bin/bash

sbatch -p compute calculate_daily_anomalies_dmn.sh
sbatch -p compute calculate_daily_anomalies_wsg_hausen.sh
sbatch -p compute calculate_daily_anomalies_wsg_zartener_becken.sh
sbatch -p compute calculate_daily_anomalies_wsg_boetzingen.sh
sbatch -p compute calculate_daily_anomalies_wsg_breisach.sh
sbatch -p compute calculate_daily_anomalies_wsg_ebringen.sh
sbatch -p compute calculate_daily_anomalies_wsg_eichstetten.sh
sbatch -p compute calculate_daily_anomalies_wsg_gottenheim.sh
sbatch -p compute calculate_daily_anomalies_wsg_krozinger_berg.sh
sbatch -p compute calculate_daily_anomalies_wsg_march.sh
sbatch -p compute calculate_daily_anomalies_wsg_schlatt.sh
sbatch -p compute calculate_daily_anomalies_wsg_tuniberg.sh
sbatch -p compute calculate_daily_anomalies_wsg_umkirch.sh
