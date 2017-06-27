export MXNET_ENABLE_GPU_P2P=0; python distribution/launch.py -n 2 -s 12 \
   -H distribution/hosts --sync-dst-dir /tmp/mxnet \
   python Train_Distribution_Batch_Plate.py


