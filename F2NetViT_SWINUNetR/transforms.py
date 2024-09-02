from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    DivisiblePadd
)

# Debugging function to print shapes
def print_shape(data, label):
    print(f"(transforms.py) After transform - Image shape: {data.shape}, Label shape: {label.shape if label is not None else 'None'}")
    return data, label

# Transforms to be applied on training instances (with labels)
train_transform = Compose(
    [   
        EnsureChannelFirstd(keys="image", channel_dim=0),
        EnsureChannelFirstd(keys="label", channel_dim=0),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'])
    ]
)

# Cuda version of "train_transform" (with labels)
train_transform_cuda = Compose(
    [   
        EnsureChannelFirstd(keys="image", channel_dim=0),
        EnsureChannelFirstd(keys="label", channel_dim=0),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)

# Transforms to be applied on validation instances (without labels)
val_transform = Compose(
    [   
        EnsureChannelFirstd(keys="image", channel_dim=0),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys="image"),
        ToTensord(keys='image')
    ]
)

# Cuda version of "val_transform" (without labels)
val_transform_cuda = Compose(
    [   
        EnsureChannelFirstd(keys="image", channel_dim=0),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys="image"),
        ToTensord(keys='image', device='cuda')
    ]
)
