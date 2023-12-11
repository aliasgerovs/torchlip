class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        # In 'train' mode, we apply both resizing and a variety of data augmentation techniques.
        return A.Compose([
            A.Resize(CFG.size, CFG.size, always_apply=True),
            A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
            A.VerticalFlip(p=0.5),  # Randomly flip images vertically
            A.RandomRotate90(p=0.5),  # Randomly rotate images by 90 degrees
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),  # Random shifts, scales, and rotations
            A.Perspective(scale=(0.05, 0.1), p=0.5),  # Random perspective transformations
            A.Blur(blur_limit=(3, 5), p=0.5),  # Apply Gaussian blur with a kernel size between 3 and 5
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add Gaussian noise to the images
            A.Normalize(max_pixel_value=255.0, always_apply=True),
            ToTensorV2(always_apply=True),  # Convert image to PyTorch tensor format
        ])
    else:
        # In 'test' or other modes, we only apply the necessary resizing and normalization.
        return A.Compose([
            A.Resize(CFG.size, CFG.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
            ToTensorV2(always_apply=True),  # Convert image to PyTorch tensor format
        ])
