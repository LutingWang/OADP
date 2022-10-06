#!/usr/bin/env bash

set -x

grep -n -r "ipdb\|breakpoint" \
    --exclude tools/odps_train.sh \
    --exclude tools/odps_test_multilabel.sh \
    cafe clip mldec tools
if [[ $? -eq 0 ]]; then
    echo "ipdb and breakpoint are not allowed in this repo."
    exit 1
fi

set -e

CONFIG=$1
PY_ARGS=${@:2}
PY_ARGS="${PY_ARGS} --odps GIT_COMMIT_ID:\\\'$(git rev-parse --short HEAD)\\\'"

if [[ ! -z "${DEBUG}" ]]; then
    PY_ARGS="${PY_ARGS} TRAIN_WITH_VAL_DATASET:\\\'1\\\' DRY_RUN:\\\'1\\\'"
fi

PROJECT_NAME=${PROJECT_NAME:-mldec}
ENTRY_FILE=${ENTRY_FILE:-tools/test_multilabel.py}
WORKBENCH=${WORKBENCH:-search_algo_quality_dev}  # search_algo_quality_dev, imac_dev
ROLEARN=${ROLEARN:-searchalgo}  # searchalgo, imac

tar -zchf /tmp/${PROJECT_NAME}.tar.gz cafe clip configs mldec tools requirements.txt

cmd_oss="
use ${WORKBENCH};
pai -name pytorch180
    -Dscript=\"file:///tmp/${PROJECT_NAME}.tar.gz\"
    -DentryFile=\"${ENTRY_FILE}\"
    -DworkerCount=1
    -DuserDefinedParameters=\"${CONFIG} ${PY_ARGS}\"
    -Dbuckets=\"oss://mvap-data/zhax/wangluting/?role_arn=acs:ram::1367265699002728:role/${ROLEARN}4pai&host=cn-zhangjiakou.oss.aliyuncs.com\";
"
# set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
    # -Dcluster=\"{\\\"worker\\\":{\\\"gpu\\\":${GPUS}00}}\"
odpscmd -e "${cmd_oss}"