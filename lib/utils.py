class DB(object):
    """
    This class is used to load the dataset and perform some preprocessing.

    """


class Data(object):
    """
    This class is used to load the dataset and perform some preprocessing.

    """

    def __init__(self, data_dir, data_name, split, img_size=256, transform=None):
        """
        :param data_dir: path to the dataset
        :param data_name: name of the dataset
        :param split: split, one of 'train', 'val', or 'test'
        :param img_size: size of the output image
        :param transform: image transform pipeline
        """
        self.data_dir = data_dir
        self.data_name = data_name
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(data_dir, data_name, split) + '/*.*'))

    def __getitem__(self, index):
        """
        Read an image from a file and preprocesses it and returns.
        """
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.files)