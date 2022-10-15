import sys
import time

import requests
import todd

sys.path.insert(0, '')
import mldec


root = 'http://api.conceptnet.io'
uri = '/related/c/en/{}?filter=/c/en'

logger = todd.get_logger()

extra_classnames = []
for i, classname in enumerate(mldec.COCO_48):
    timer = time.time()
    url = root + uri.format(classname)
    logger.info(f'Getting {i}: {url}')
    obj = requests.get(url).json()
    logger.info(f'{url} parsed')
    for related in obj['related'][:20]:
        related_uri = related['@id']
        assert related_uri.startswith('/c/en/')
        extra_classname = related_uri[len('/c/en/'):].replace('_', ' ')
        if any(c in extra_classname for c in mldec.COCO):
            logger.debug(f'Skipping related {extra_classname}')
            continue
        logger.info(f'Adding related {extra_classname}')
        extra_classnames.append(extra_classname)
    # time.sleep(max(0, timer + 1 - time.time()))

with open('data/coco/annotations/extra_classnames.txt', 'w') as f:
    for extra_classname in extra_classnames:
        f.write(extra_classname + '\n')
