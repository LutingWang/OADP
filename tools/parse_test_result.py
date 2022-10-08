"""Parse mmdet classwise test results.

MMDet classwise test results look like this

    +------------+-------+--------------+-------+------------+-------+
    | category   | AP    | category     | AP    | category   | AP    |
    +------------+-------+--------------+-------+------------+-------+
    | person     | 0.487 | bicycle      | 0.204 | car        | 0.368 |
    | motorcycle | 0.313 | train        | 0.457 | truck      | 0.243 |
    | boat       | 0.167 | bench        | 0.168 | bird       | 0.268 |
    | horse      | 0.415 | sheep        | 0.332 | bear       | 0.490 |
    | zebra      | 0.582 | giraffe      | 0.575 | backpack   | 0.105 |
    | handbag    | 0.097 | suitcase     | 0.202 | frisbee    | 0.507 |
    | skis       | 0.137 | kite         | 0.337 | surfboard  | 0.215 |
    | bottle     | 0.302 | fork         | 0.158 | spoon      | 0.041 |
    | bowl       | 0.304 | banana       | 0.142 | apple      | 0.129 |
    | sandwich   | 0.247 | orange       | 0.244 | broccoli   | 0.185 |
    | carrot     | 0.131 | pizza        | 0.421 | donut      | 0.328 |
    | chair      | 0.173 | bed          | 0.306 | toilet     | 0.431 |
    | tv         | 0.442 | laptop       | 0.453 | mouse      | 0.498 |
    | remote     | 0.164 | microwave    | 0.436 | oven       | 0.228 |
    | toaster    | 0.208 | refrigerator | 0.359 | book       | 0.111 |
    | clock      | 0.451 | vase         | 0.255 | toothbrush | 0.067 |
    | airplane   | 0.000 | bus          | 0.000 | cat        | 0.000 |
    | dog        | 0.000 | cow          | 0.000 | elephant   | 0.000 |
    | umbrella   | 0.000 | tie          | 0.000 | snowboard  | 0.000 |
    | skateboard | 0.001 | cup          | 0.005 | knife      | 0.000 |
    | cake       | 0.000 | couch        | 0.024 | keyboard   | 0.000 |
    | sink       | 0.000 | scissors     | 0.000 | None       | None  |
    +------------+-------+--------------+-------+------------+-------+

This script attempts to compute mAP with regard to interested splits. For
instance, the following example computs mAP for `COCO_17`

    Please paste the results below (enter twice to stop):
        +------------+-------+--------------+-------+------------+-------+
        | category   | AP    | category     | AP    | category   | AP    |
        +------------+-------+--------------+-------+------------+-------+
        |                              ...                               |
        +------------+-------+--------------+-------+------------+-------+

    Please input the split name: coco 17
    0.001764705882352941
"""

import re
import sys

sys.path.insert(0, '')
import mldec


print("Please paste the results below (enter twice to stop):")
results = []
while True:
    input_ = input()
    if input_ == '':
        break
    results.append(input_)
result = '\n'.join(results)

split = input('Please input the split name: ').strip().upper().replace(' ', '_')
classes = getattr(mldec, split)
classwise_results = {
    c: float(re.search(f'\\| {c} +\\| (0.\\d{{3}}) \\|', result).group(1))
    for c in classes
}
print(classwise_results)
mAP = sum(classwise_results.values()) / len(classes)
print(mAP)
