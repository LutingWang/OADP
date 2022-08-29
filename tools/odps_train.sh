#!/usr/bin/env bash

set -x

grep -n -r "ipdb\|breakpoint" \
    --exclude-dir .git \
    --exclude-dir data \
    --exclude-dir lvis \
    --exclude-dir mmdet \
    --exclude-dir pretrained \
    --exclude-dir todd \
    --exclude-dir work_dirs \
    --exclude ./tools/odps_train.sh \
    --exclude ./tools/odps_test.sh \
    --exclude requirements.txt \
    .
if [[ $? -eq 0 ]]; then
    echo "ipdb and breakpoint are not allowed in this repo."
    exit 1
fi

set -e

JOB_NAME=$1
CONFIG=$2
GPUS=$3
PY_ARGS=${@:4}

PROJECT_NAME=${PROJECT_NAME:-mldec}
ENTRY_FILE=${ENTRY_FILE:-tools/train.py}
WORKBENCH=${WORKBENCH:-search_algo_quality_dev}  # search_algo_quality_dev, imac_dev
ROLEARN=${ROLEARN:-searchalgo}  # searchalgo, imac

tar -zchf /tmp/${PROJECT_NAME}.tar.gz \
    --exclude .git \
    --exclude data \
    --exclude lvis \
    --exclude mmdet \
    --exclude pretrained \
    --exclude todd \
    --exclude work_dirs \
    .

cmd_oss="
use ${WORKBENCH};
pai -name pytorch180
    -Dscript=\"file:///tmp/${PROJECT_NAME}.tar.gz\"
    -DentryFile=\"${ENTRY_FILE}\"
    -DworkerCount=${GPUS}
    -DuserDefinedParameters=\"${CONFIG} ${JOB_NAME} --odps ${PY_ARGS}\"
    -Dbuckets=\"oss://mvap-data/zhax/wangluting/?role_arn=acs:ram::1367265699002728:role/${ROLEARN}4pai&host=cn-zhangjiakou.oss.aliyuncs.com\";
"
# set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
    # -Dcluster=\"{\\\"worker\\\":{\\\"gpu\\\":${GPUS}00}}\"
odpscmd -e "${cmd_oss}"
