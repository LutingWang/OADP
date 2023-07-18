#! /bin/sh
#
# train_pai.sh
# Copyright (C) 2021 biaolong.cbl <biaolong.cbl@a74e07347.et15>
#
# Distributed under terms of the MIT license.
#

rm -rf pai_immf.tar.gz
cd ../
tar  --exclude='**/__pycache__' --exclude='./pretrain_models' --exclude='./cache' --exclude='./experiments' --exclude='*.pkl'  --exclude='./data'  --exclude='*.pth'  --exclude='./pai'  --exclude='./work_dir' --exclude='./work_dirs'   --exclude='./jiuding' --exclude='./log_tmp' -zcvf ./pai/pai_immf.tar.gz .

# -Doversubscription=true
# set odps.algo.hybrid.deploy.info=LABEL:V100M32:OPER_EQUAL;
# set odps.algo.hybrid.deploy.info=LABEL:P100:OPER_EQUAL;
# -Doversubscription=true
# use search_algo_quality_dev;
# set odps.algo.hybrid.deploy.info=LABEL:V100M32:OPER_EQUAL;


cmd_oss='''
use search_algo_quality_dev;
set odps.algo.hybrid.deploy.info=LABEL:V100M32:OPER_EQUAL;
pai -name pytorch180
-Dscript="file:///Users/lutingwang/Developer/ConditionalDETR/pai/pai_immf.tar.gz"
-DentryFile="tools/launch.py"
-Dbuckets="oss://mvap-public-data/chenbiaolong/datasets/?role_arn=acs:ram::1367265699002728:role/searchalgo4pai&host=cn-zhangjiakou.oss.aliyuncs.com"
-DuserDefinedParameters="--nproc_per_node 4  distill.py  --backbone resnet50 --batch_size 2 --enc_layers 0 --fpn  --dec_layers 6 --coco_path=/data/oss_bucket_0/coco2017/  --output_dir oss://mvap-data/zhax/wangluting/ConditionalDETR/conddetr_r50_fpn_dis_d3etr_g8/ --refine_level 2 --cfg cfg.py --group_detr 2 --dec_distill --backbone_teacher resnet101 --teacher oss://mvap-public-data/chenbiaolong/models/checkpoints/ConditionalDETR_r101_epoch50.pth --resume oss://mvap-data/zhax/wangluting/ConditionalDETR/conddetr_r50_fpn_dis_d3etr/checkpoint.pth"
-Dcluster="{\"worker\":{\"gpu\":400,\"cpu\":3200,\"memory\":100000}}"
-DworkerCount=2;
'''
# -Doversubscription=true

# -Dbuckets="oss://mvap-public-data/chenbiaolong/projects/clip_bert/data/?role_arn=acs:ram::1367265699002728:role/imac4pai&host=cn-zhangjiakou.oss.aliyuncs.com"
echo $cmd_oss
odpscmd -e "$cmd_oss"

#-Doversubscription=true
