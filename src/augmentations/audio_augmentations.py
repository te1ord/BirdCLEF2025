from audiomentations import AddGaussianNoise, TimeStretch, PitchShift

KEY2AUDIO_AUGMENTATION = {
    "gaussian_noise": AddGaussianNoise,
    "time_stretch": TimeStretch,
    "pitch_shift": PitchShift,
    # Add more augmentations as needed
}