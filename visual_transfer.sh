#!/bin/bash
python VisualTopN.py 
dest="mingzhang@192.168.0.101:~/Desktop"
echo "transfering to <${dest}>..."
scp -r ImageResult ${dest} 
