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

#--node_rank=${node_rank} --master_addr=${master_addr} \
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=3 \
--master_port=23452 \
-m mldec.extract_vild_embeddings extract_vild_embeddings configs/mldec/extract_vild_embeddings.py --k8s 
#--override .train.dataloader.num_workers:0 .val.dataloader.num_workers:0
