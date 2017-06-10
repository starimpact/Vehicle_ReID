folder=~/work/dmlc/mxnet_v0.8.0
echo "copy mxnet from ${folder}..."
cp -r ${folder}/dmlc-core/tracker ./dmlc-core/
cp -r ${folder}/python/mxnet ./
cp -r ${folder}/lib/libmxnet.so ./mxnet
