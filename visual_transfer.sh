#!/bin/bash
python VisualTopN.py 
dest="mzhang@192.168.0.107:~/Desktop"
echo "transfering to <${dest}>..."
scp -r ImageResult ${dest} 
