# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import os
from pathlib import Path
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


_DISFA_MODULE = None


def _get_disfa_module():
    global _DISFA_MODULE
    if _DISFA_MODULE is None:
        module_path = Path(__file__).resolve().parent / 'dataset' / 'DISFA' / 'disfa_pytorch_dataset.py'
        spec = importlib.util.spec_from_file_location('disfa_pytorch_dataset', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load DISFA dataset module from {module_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DISFA_MODULE = module
    return _DISFA_MODULE


def _parse_disfa_selected_aus(selected_aus):
    if not selected_aus:
        return None
    return tuple(int(au.strip()) for au in selected_aus.split(',') if au.strip())

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == 'DISFA':
        disfa_module = _get_disfa_module()

        manifest_path = args.disfa_manifest_path or os.path.join(
            args.data_path, 'prepared', 'disfa_aligned_manifest.csv')
        split = 'train' if is_train else args.disfa_eval_split
        selected_aus = _parse_disfa_selected_aus(args.disfa_selected_aus) or disfa_module.AUS
        dataset = disfa_module.DISFAAlignedDataset(
            manifest_path=manifest_path,
            dataset_root=args.data_path,
            split=split,
            target_mode=args.disfa_target_mode,
            selected_aus=selected_aus,
            image_transform=transform,
        )
        nb_classes = len(selected_aus)
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    if args.data_set == 'DISFA':
        # DISFA images are already aligned facial crops, so keep resizing simple.
        t = [transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC)]
        if is_train:
            t.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            ])
        t.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        return transforms.Compose(t)

    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
