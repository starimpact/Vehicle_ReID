export DMLC_INTERFACE=p15p1; python distribution/launch.py -n 4 -s 4 \
   -H distribution/hosts --sync-dst-dir /tmp/mxnet \
   python Train_Distribution_Batch_Plate.py

