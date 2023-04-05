_base_ = [
    '../base.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(
            output_dir='data/lvis_v1/oake/objects/train2017',
            # TODO: select oln
            proposal_file='data/lvis_v1/proposals/debug_train.pkl',
            proposal_sorted=True,
            # proposal_file='data/lvis_v1/proposals/rpn_r101_fpn_coco_train.pkl',
            # proposal_sorted=False,
            # proposal_file='data/lvis_v1/proposals/oln_r50_fpn_coco_train.pkl',
            # proposal_sorted=True,
        ),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            output_dir='data/lvis_v1/oake/objects/val2017',
            # TODO: select oln
            proposal_file='data/lvis_v1/proposals/debug_val.pkl',
            proposal_sorted=True,
            # proposal_file='data/lvis_v1/proposals/rpn_r101_fpn_coco_val.pkl',
            # proposal_sorted=False,
            # proposal_file='data/lvis_v1/proposals/oln_r50_fpn_coco_val.pkl',
            # proposal_sorted=True,
        ),
    ),
)
mini_batch_size = 512
