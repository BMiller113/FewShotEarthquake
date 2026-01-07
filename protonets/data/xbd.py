import os
import random
import re
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

from protonets.data.base import convert_dict, CudaTransform


# -----------------------------
# Naming convention (your dataset)
# -----------------------------
PRE_RE = re.compile(r"^(?P<stem>.+)_pre_disaster\.(png|tif|tiff)$", re.IGNORECASE)
POST_RE = re.compile(r"^(?P<stem>.+)_post_disaster\.(png|tif|tiff)$", re.IGNORECASE)
POST_MASK_RE = re.compile(r"^(?P<stem>.+)_post_disaster_target\.(png|tif|tiff)$", re.IGNORECASE)
PRE_MASK_RE = re.compile(r"^(?P<stem>.+)_pre_disaster_target\.(png|tif|tiff)$", re.IGNORECASE)


def _map_by_stem(folder: str, regex: re.Pattern) -> Dict[str, str]:
    m = {}
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        mo = regex.match(fn)
        if mo:
            m[mo.group("stem")] = p
    return m


def _pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pil_mask(path: str) -> Image.Image:
    # Keep as-is; we'll convert to int tensor
    return Image.open(path)


def _to_tensor_img(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3
    arr = np.transpose(arr, (2, 0, 1))              # 3,H,W
    return torch.from_numpy(arr)


def _to_tensor_mask(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.int64)  # H,W
    return torch.from_numpy(arr)


def _normalize_img(img: torch.Tensor,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    m = torch.tensor(mean, dtype=img.dtype)[:, None, None]
    s = torch.tensor(std, dtype=img.dtype)[:, None, None]
    return (img - m) / s


@dataclass(frozen=True)
class XbdSample:
    stem: str
    pre_path: str
    post_path: str
    post_mask_path: str
    pre_mask_path: Optional[str] = None  # optional


def _build_index_from_as_is(root: str) -> List[XbdSample]:
    """
    Expects:
      root/
        img_pre/
        img_post/
        gt_post/
        gt_pre/   (optional)
    """
    img_pre = os.path.join(root, "img_pre")
    img_post = os.path.join(root, "img_post")
    gt_post = os.path.join(root, "gt_post")
    gt_pre = os.path.join(root, "gt_pre")

    if not (os.path.isdir(img_pre) and os.path.isdir(img_post) and os.path.isdir(gt_post)):
        raise FileNotFoundError(
            "Expected folders under data.root:\n"
            f"  {img_pre}\n  {img_post}\n  {gt_post}\n"
            "Optional:\n"
            f"  {gt_pre}\n"
        )

    pre_map = _map_by_stem(img_pre, PRE_RE)
    post_map = _map_by_stem(img_post, POST_RE)
    post_m_map = _map_by_stem(gt_post, POST_MASK_RE)

    pre_m_map = {}
    if os.path.isdir(gt_pre):
        pre_m_map = _map_by_stem(gt_pre, PRE_MASK_RE)

    stems = sorted(set(pre_map.keys()) & set(post_map.keys()) & set(post_m_map.keys()))
    if not stems:
        raise RuntimeError("No matched stems found across img_pre/img_post/gt_post. Check naming patterns.")

    items = []
    for stem in stems:
        items.append(
            XbdSample(
                stem=stem,
                pre_path=pre_map[stem],
                post_path=post_map[stem],
                post_mask_path=post_m_map[stem],
                pre_mask_path=pre_m_map.get(stem, None),
            )
        )
    return items


def _random_crop_coords(H: int, W: int, ph: int, pw: int) -> Tuple[int, int]:
    if H < ph or W < pw:
        raise ValueError(f"Patch size {ph}x{pw} larger than image {H}x{W}")
    y = random.randint(0, H - ph)
    x = random.randint(0, W - pw)
    return y, x


def _crop(t: torch.Tensor, y: int, x: int, ph: int, pw: int) -> torch.Tensor:
    if t.ndim == 3:
        return t[:, y:y + ph, x:x + pw]
    return t[y:y + ph, x:x + pw]


def _pick_patch(mask_for_constraint: torch.Tensor,
                constraint_cls: Optional[int],
                ph: int, pw: int,
                min_pixels: int,
                max_tries: int) -> Optional[Tuple[int, int]]:
    """
    If constraint_cls is None: accept any random crop.
    Else: require >= min_pixels pixels of constraint_cls in the crop.
    """
    H, W = mask_for_constraint.shape
    for _ in range(max_tries):
        y, x = _random_crop_coords(H, W, ph, pw)
        if constraint_cls is None:
            return y, x
        patch = _crop(mask_for_constraint, y, x, ph, pw)
        if int((patch == constraint_cls).sum().item()) >= min_pixels:
            return y, x
    return None


def _extract_episode_for_fixed_classes(items: List[XbdSample],
                                      chosen_classes: List[int],
                                      n_support: int,
                                      n_query: int,
                                      patch_size: int,
                                      min_class_pixels: int,
                                      max_tries: int,
                                      use_two_stream: bool,
                                      use_pre_gt_for_sampling: bool,
                                      pre_gt_building_value: int) -> Dict:
    """
    Attempt to build an episode for an explicit list of classes.
    Raises RuntimeError if it cannot satisfy one of the classes.
    """
    ph = pw = patch_size
    xs_img, xs_msk, xq_img, xq_msk = [], [], [], []

    for cls in chosen_classes:
        sup_imgs, sup_msks = [], []
        qry_imgs, qry_msks = [], []
        needed = n_support + n_query
        collected = 0

        # Shuffle scenes to diversify sampling
        scene_order = random.sample(items, k=len(items))

        for sample in scene_order:
            if collected >= needed:
                break

            pre = _normalize_img(_to_tensor_img(_pil_rgb(sample.pre_path)))
            post = _normalize_img(_to_tensor_img(_pil_rgb(sample.post_path)))
            post_mask = _to_tensor_mask(_pil_mask(sample.post_mask_path))

            pre_mask = None
            if use_pre_gt_for_sampling and sample.pre_mask_path is not None:
                pre_mask = _to_tensor_mask(_pil_mask(sample.pre_mask_path))

            coords = _pick_patch(post_mask, cls, ph, pw, min_class_pixels, max_tries)
            if coords is None:
                continue

            y, x = coords

            # Optional filter: only sample where pre-gt indicates building presence
            if pre_mask is not None:
                pre_patch = _crop(pre_mask, y, x, ph, pw)
                if int((pre_patch == pre_gt_building_value).sum().item()) < 1:
                    continue

            pre_p = _crop(pre, y, x, ph, pw)
            post_p = _crop(post, y, x, ph, pw)
            m_p = _crop(post_mask, y, x, ph, pw)

            if use_two_stream:
                img_p = {"pre": pre_p, "post": post_p}
            else:
                img_p = torch.cat([pre_p, post_p], dim=0)

            if collected < n_support:
                sup_imgs.append(img_p)
                sup_msks.append(m_p)
            else:
                qry_imgs.append(img_p)
                qry_msks.append(m_p)

            collected += 1

        if collected < needed:
            raise RuntimeError(f"Could not form episode for class {cls}: got {collected}/{needed}")

        xs_img.append(sup_imgs)
        xs_msk.append(sup_msks)
        xq_img.append(qry_imgs)
        xq_msk.append(qry_msks)

    def _stack_imgs(img_list_2d):
        if isinstance(img_list_2d[0][0], dict):
            pre_t = torch.stack([torch.stack([x["pre"] for x in row], 0) for row in img_list_2d], 0)
            post_t = torch.stack([torch.stack([x["post"] for x in row], 0) for row in img_list_2d], 0)
            return {"pre": pre_t, "post": post_t}
        return torch.stack([torch.stack(row, 0) for row in img_list_2d], 0)

    return {
        "class_ids": torch.tensor(chosen_classes, dtype=torch.long),
        "xs_img": _stack_imgs(xs_img),
        "xs_msk": torch.stack([torch.stack(row, 0) for row in xs_msk], 0),
        "xq_img": _stack_imgs(xq_img),
        "xq_msk": torch.stack([torch.stack(row, 0) for row in xq_msk], 0),
    }


def _extract_episode_xbd(n_support: int, n_query: int, n_way: int,
                         patch_size: int,
                         class_ids: List[int],
                         min_class_pixels: int,
                         max_tries: int,
                         use_two_stream: bool,
                         use_pre_gt_for_sampling: bool,
                         pre_gt_building_value: int,
                         episode_max_retries: int,
                         allow_way_drop: bool,
                         min_way: int,
                         d: Dict) -> Dict:
    """
    Robust episodic sampler for imbalanced segmentation.

    Behavior:
      - Tries to sample n_way classes from `class_ids`
      - If a class fails consistently (e.g., rare class in a split), removes it and retries
      - If remaining classes < n_way:
          - if allow_way_drop: reduces n_way down to len(remaining), but not below min_way
          - else: raises
    """
    items: List[XbdSample] = d["items"]

    # Work on a mutable candidate list
    candidates = list(class_ids)
    requested_way = int(n_way)

    last_err = None
    for _ in range(int(episode_max_retries)):
        if len(candidates) < requested_way:
            if allow_way_drop and len(candidates) >= int(min_way):
                requested_way = len(candidates)
            else:
                raise RuntimeError(
                    f"Not enough feasible classes to sample an episode: "
                    f"requested_way={requested_way}, candidates={candidates}."
                )

        chosen = random.sample(candidates, k=requested_way)

        try:
            return _extract_episode_for_fixed_classes(
                items=items,
                chosen_classes=chosen,
                n_support=n_support,
                n_query=n_query,
                patch_size=patch_size,
                min_class_pixels=min_class_pixels,
                max_tries=max_tries,
                use_two_stream=use_two_stream,
                use_pre_gt_for_sampling=use_pre_gt_for_sampling,
                pre_gt_building_value=pre_gt_building_value,
            )
        except RuntimeError as e:
            last_err = e

            # If the error specifies a class, remove it from candidates (common case)
            msg = str(e)
            m = re.search(r"class\s+(\d+)", msg)
            if m:
                bad_cls = int(m.group(1))
                if bad_cls in candidates and len(candidates) > int(min_way):
                    candidates = [c for c in candidates if c != bad_cls]
            continue

    raise RuntimeError(
        f"Failed to form an episode after {episode_max_retries} retries. "
        f"Last error: {last_err}. "
        "Try lowering data.min_class_pixels, increasing data.max_tries, "
        "reducing data.query, or allow way-dropping in eval."
    )


def load(opt, splits):
    """
    AS-IS loader. No re-pack required.

    Required:
      data.root -> folder containing img_pre/img_post/gt_post/(optional gt_pre)

    Episode params:
      data.way, data.shot, data.query, data.train_episodes, data.test_episodes
      data.test_way/test_shot/test_query optionally override

    Sampling params:
      data.patch_size (default 128)
      data.min_class_pixels (default 64)
      data.max_tries (default 50)
      data.class_ids "0,1,2,3,4"
      data.use_two_stream (recommended True)
      data.episode_max_retries (default 50)

    Eval robustness params (important for rare classes 3/4):
      data.allow_way_drop (default True)
      data.min_way (default 2)

    Optional pre-gt filter:
      data.use_pre_gt_for_sampling (default False)
      data.pre_gt_building_value (default 1)
    """
    root = opt.get("data.root", "")
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"data.root not found: {root}")

    items_all = _build_index_from_as_is(root)

    class_ids = [int(x) for x in opt.get("data.class_ids", "0,1,2,3,4").split(",")]
    patch_size = int(opt.get("data.patch_size", 128))
    min_class_pixels = int(opt.get("data.min_class_pixels", 64))
    max_tries = int(opt.get("data.max_tries", 50))
    use_two_stream = bool(opt.get("data.use_two_stream", False)) or bool(opt.get("data.use_two_stream", True))

    use_pre_gt_for_sampling = bool(opt.get("data.use_pre_gt_for_sampling", False))
    pre_gt_building_value = int(opt.get("data.pre_gt_building_value", 1))

    episode_max_retries = int(opt.get("data.episode_max_retries", 50))

    allow_way_drop = bool(opt.get("data.allow_way_drop", True))
    min_way = int(opt.get("data.min_way", 2))

    # Split config (random split by stem)
    split_seed = int(opt.get("data.split_seed", 1234))
    train_frac = float(opt.get("data.split_train_frac", 0.8))
    val_frac = float(opt.get("data.split_val_frac", 0.1))

    rng = random.Random(split_seed)
    stems = [it.stem for it in items_all]
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_set = set(stems[:n_train])
    val_set = set(stems[n_train:n_train + n_val])
    test_set = set(stems[n_train + n_val:])

    ret = {}
    for split in splits:
        if split == "train":
            allowed = train_set
        elif split == "val":
            allowed = val_set
        elif split == "test":
            allowed = test_set
        elif split == "trainval":
            allowed = train_set | val_set
        else:
            raise ValueError(f"Unknown split name: {split}")

        split_items = [it for it in items_all if it.stem in allowed]
        if not split_items:
            raise RuntimeError(f"No items available for split '{split}'. Check split settings.")

        # Episode sizes
        if split in ["val", "test"] and opt.get("data.test_way", 0) != 0:
            n_way = int(opt["data.test_way"])
        else:
            n_way = int(opt["data.way"])

        if split in ["val", "test"] and opt.get("data.test_shot", 0) != 0:
            n_support = int(opt["data.test_shot"])
        else:
            n_support = int(opt["data.shot"])

        if split in ["val", "test"] and opt.get("data.test_query", 0) != 0:
            n_query = int(opt["data.test_query"])
        else:
            n_query = int(opt["data.query"])

        n_episodes = int(opt["data.test_episodes"]) if split in ["val", "test"] else int(opt["data.train_episodes"])

        transforms = [
            partial(convert_dict, "items"),
            partial(
                _extract_episode_xbd,
                n_support, n_query, n_way,
                patch_size,
                class_ids,
                min_class_pixels,
                max_tries,
                use_two_stream,
                use_pre_gt_for_sampling,
                pre_gt_building_value,
                episode_max_retries,
                allow_way_drop,
                min_way,
            ),
        ]
        if opt.get("data.cuda", False):
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        ds = TransformDataset(ListDataset([split_items]), transforms)

        class OneEpisodeSampler:
            def __len__(self):
                return n_episodes

            def __iter__(self):
                for _ in range(n_episodes):
                    yield torch.LongTensor([0])

        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=OneEpisodeSampler(), num_workers=0)

    return ret
