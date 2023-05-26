import torch
import torchvision
import numpy as np
import PIL
import collections
import random
import cv2
import os
from pytorch_openpose.src import util

__all__ = [
    "VideoFilePathToTensor",
    "VideoFolderPathToTensor",
    "VideoResize",
    "VideoRandomCrop",
    "VideoCenterCrop",
    "VideoRandomHorizontalFlip",
    "VideoRandomVerticalFlip",
    "VideoGrayscale",
]

# working from code from FROM https://github.com/YuxinZhaozyx/pytorch-VideoDataset


class VideoFilePathToTensor(object):
    """load video at given file path to torch.Tensor (C x L x H x W, C = 3)
        It can be composed with torchvision.transforms.Compose().

    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames.
        fps (int): sample frame per seconds. It must lower than or equal the origin video fps.
            Default is None.
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len=None, fps=None, padding_mode=None):
        self.max_len = max_len
        self.fps = fps
        assert padding_mode in (None, "zero", "last")
        self.padding_mode = padding_mode
        self.channels = 3  # only available to read 3 channels video

    def __call__(self, path):
        """
        Args:
            path (str): path of video file.

        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """

        # open video file
        cap = cv2.VideoCapture(path)
        assert cap.isOpened()

        # calculate sample_factor to reset fps
        sample_factor = 1
        if self.fps:
            old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
            sample_factor = int(old_fps / self.fps)
            assert sample_factor >= 1

        # init empty output frames (C x L x H x W)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(int(num_frames / sample_factor), self.max_len)
        else:
            # time length is unlimited
            time_len = int(num_frames / sample_factor)

        frames = torch.FloatTensor(self.channels, time_len, height, width)

        for index in range(time_len):
            frame_index = sample_factor * index

            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == "zero":
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == "last":
                    # fill the rest frames with the last frame
                    assert index > 0
                    frames[:, index:, :, :] = frames[:, index - 1, :, :].view(
                        self.channels, 1, height, width
                    )
                break

        frames /= 255
        cap.release()
        # device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        # frames = frames.to(device)
        return frames


class VideoFolderPathToTensor(object):
    """load video at given folder path to torch.Tensor (C x L x H x W)
        It can be composed with torchvision.transforms.Compose().

    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames.
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len=None, padding_mode=None):
        self.max_len = max_len
        assert padding_mode in (None, "zero", "last")
        self.padding_mode = padding_mode

    def __call__(self, path):
        """
        Args:
            path (str): path of video folder.

        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """

        # get video properity
        frames_path = sorted(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
        )
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)

        # init empty output frames (C x L x H x W)
        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(num_frames, self.max_len)
        else:
            # time length is unlimited
            time_len = num_frames

        frames = torch.FloatTensor(channels, time_len, height, width)

        # load the video to tensor
        for index in range(time_len):
            if index < num_frames:
                # frame exists
                # read frame
                frame = cv2.imread(frames_path[index])
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == "zero":
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == "last":
                    # fill the rest frames with the last frame
                    assert index > 0
                    frames[:, index:, :, :] = frames[:, index - 1, :, :].view(
                        channels, 1, height, width
                    )
                break

        frames /= 255
        return frames


class VideoResize(object):
    """resize video tensor (C x L x H x W) to (C x L x h x w)

    Args:
        size (sequence): Desired output size. size is a sequence like (H, W),
            output size will matched to this.
        interpolation (int, optional): Desired interpolation. Default is 'PIL.Image.BILINEAR'
    """

    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        assert isinstance(size, collections.abc.Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to be scaled (C x L x H x W)

        Returns:
            torch.Tensor: Rescaled video (C x L x h x w)
        """

        h, w = self.size
        C, L, H, W = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)

        # use torchvision implemention to resize video frames
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(self.size, self.interpolation),
                torchvision.transforms.ToTensor(),
            ]
        )

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            rescaled_video[:, l, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class VideoRandomCrop(object):
    """Crop the given Video Tensor (C x L x H x W) at a random location.

    Args:
        size (sequence): Desired output size like (h, w).
    """

    def __init__(self, size):
        assert len(size) == 2
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top : top + h, left : left + w]

        return video


class VideoCenterCrop(object):
    """Crops the given video tensor (C x L x H x W) at the center.

    Args:
        size (sequence): Desired output size of the crop like (h, w).
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped Video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = int((H - h) / 2)
        left = int((W - w) / 2)

        video = video[:, :, top : top + h, left : left + w]

        return video


class VideoRandomHorizontalFlip(object):
    """Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.

        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video


class VideoRandomVerticalFlip(object):
    """Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.

        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([2])

        return video


class VideoGrayscale(object):
    """Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output video
    """

    def __init__(self, num_output_channels=1):
        assert num_output_channels in (1, 3)
        self.num_output_channels = num_output_channels

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (3 x L x H x W) to be converted to grayscale.

        Returns:
            torch.Tensor: Grayscaled video (1 x L x H x W  or  3 x L x H x W)
        """

        C, L, H, W = video.size()
        grayscaled_video = torch.FloatTensor(self.num_output_channels, L, H, W)

        # use torchvision implemention to convert video frames to gray scale
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Grayscale(self.num_output_channels),
                torchvision.transforms.ToTensor(),
            ]
        )

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            grayscaled_video[:, l, :, :] = frame

        return grayscaled_video


class NormalizeVideoFrames(object):
    """
    Normalize a tensor video with mean and standard deviation.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # per channel mean and standard deviation in RGB order
        self.mean = mean
        self.std = std

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor):  video (torch.Tensor): Video (3 x L x H x W) to be normalized.

        Returns:
            torch.Tensor: Normalized video.
        """

        C, L, H, W = video.size()
        normalized_video = torch.zeros(C, L, H, W, dtype=torch.float32)

        frames = video.permute(1, 0, 2, 3)  # Shape: (L, C, H, W)

        normalized_frames = torchvision.transforms.functional.normalize(
            frames, self.mean, self.std
        )

        normalized_video = normalized_frames.permute(1, 0, 2, 3)

        return normalized_video


# Helper function to proc image for openpose
def process_image(x):
    # Convert Torch tensor to numpy array
    numpy_images = (x * 255).cpu().numpy()

    # Transpose the image array if necessary
    if len(numpy_images.shape) == 4:
        numpy_images = np.transpose(numpy_images, (0, 2, 3, 1))
    else:
        numpy_images = np.transpose(numpy_images, (1, 2, 0))

    oriImgs = []
    for numpy_image in numpy_images:
        oriImg = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        oriImgs.append(oriImg)
    oriImgs = np.array(oriImgs)

    scale_search = 0.5
    boxsize = 368
    stride = 8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.05

    # Calculate the multiplier for scaling
    multipliers = scale_search * boxsize / oriImgs.shape[1]

    # Resize images
    resized_images = []
    for oriImg in oriImgs:
        resized_img = cv2.resize(
            oriImg,
            (0, 0),
            fx=multipliers,
            fy=multipliers,
            interpolation=cv2.INTER_CUBIC,
        )
        resized_images.append(resized_img)
    resized_images = np.array(resized_images)

    # Pad images
    padded_images = []
    pads = []
    for i in range(x.shape[0]):
        padded_image, pad = util.padRightDownCorner(resized_images[i], stride, padValue)
        padded_images.append(padded_image)
        pads.append(pad)
    padded_images = np.stack(padded_images)
    pads = np.stack(pads)

    # Transpose and normalize images
    im = np.transpose(np.float32(padded_images), (0, 3, 1, 2)) / 256 - 0.5
    im = np.ascontiguousarray(im)

    return torch.from_numpy(im.astype(np.float32))
