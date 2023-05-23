from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    Resize,
    ScaleIntensity,
    Spacing,
    NormalizeIntensity
)

monai_transforms = Compose(
    [
        # I'm using these because they are common in Monai examples,
        # you may not use them if you want.
        ScaleIntensity(),
        EnsureChannelFirst(),
        # Voxel spacing
        Spacing(
            pixdim=(1.75, 1.75, 1.75)
        ),
        # Matrix size
        Resize((128, 128, 128)),
        # Normalization
        NormalizeIntensity(nonzero=True, channel_wise=True)

    ]
)