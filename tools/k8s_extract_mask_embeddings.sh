cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import json_parser;print json_parser.parse(\"$cluster_spec\", \"worker\")"
echo "worker list command is $worker_list_command"
eval worker_list=`python2 -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import json_parser;print json_parser.parse(\"$cluster_spec\", \"index\")"
eval node_rank=`python2 -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 \
--node_rank=${node_rank} --master_addr=${master_addr} \
--master_port=23452 \
-m mldec.extract_mask_embeddings extract_mask_embeddings configs/mldec/extract_mask_embeddings.py --k8s --override mini_batch_size:1000
